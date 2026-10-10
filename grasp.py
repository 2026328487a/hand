"""自适应抓取 —— 四指同时合拢,逐路电流分析,谁抵住物体谁独立停。

设计要求(来自实际使用需求):

  ① 先把拇指张到最开,让拇指与四指的张角尽可能大,物体才进得了手心
  ② 抓取时四指**同时**动,但电流要**逐路**分析 —— 物体可能不规则,每根手指
     需要的弯曲程度不同,逐路停才能让每根都真正抵住物体
  ③ 拇指最后压上来固定

为什么 ③ 必须在 ② 之后(真机实测,不是设计偏好):同一个终点(四指 1.0 /
拇指旋转 1.0 / 弯曲 0.85),

    六路同时冲 → 食指卡在 0.52、拇指只到 0.54,峰值 1072mA / 777mA
    分步到位   → 食指 1.00、拇指 0.85,峰值   47mA /  89mA

差 23 倍。拇指在四指还没就位时插进去就是互相卡死。SimSync 里的顺序互锁已经
把这个约束固化了,所以这里即使一次性下发,时序也会被自动排开。

逐路停靠的是 FollowSender 的「顶住就钉住」:哪一路电流超阈值,就把**那一路**
的下发位置钉在它当前实际位置上,不再加力,其余各路照常继续。这正是 ② 要的
行为 —— 不规则物体上每根手指停在不同深度。

阈值分两档,别混:

    保护档 450mA   自由运动峰值实测 145mA,出厂单路上限 1199mA。防机械干涉
    接触档 300mA   抓取时要的是"轻轻抵住"。gestures.py 的捏合/抓握用 350

合拢速度也要降。检测周期 40ms,slew=1.0 时接触后还会多压进去约 0.045 归一化
量;降到 0.5 就只有 0.018,接触更轻。

抓完的报告有实用价值:每根手指停在哪个深度 + 接触电流,是物体轮廓的间接测量。

用法::

    python grasp.py                      # 抓 → 停 3 秒 → 松开
    python grasp.py --close 0.85         # 四指最多闭到 0.85
    python grasp.py --contact 250        # 更灵敏的接触判据
    python grasp.py --hold 8             # 抓住后保持 8 秒
    python grasp.py --no-hand            # 只看仿真时序,不连真机
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import Calibration                              # noqa: E402
from rohand import MOTOR_CN, N_MOTORS, ROHand                     # noqa: E402
from sim_sync import (DOF_CN, SIM_IX, SIM_LT, SIM_TB, SIM_TR,     # noqa: E402
                      SIM_TO_REAL, FollowSender, SimSync,
                      THUMB_BEND_MAX, align_sim_to_hand)

CONTACT_MA = 300         # 接触判据(见模块头)
GRASP_SLEW = 0.5         # 合拢速度,归一化/秒
ARRIVE_TOL = 0.03        # 指令到位容差

# 四指的仿真序下标 → 真机电机下标
FINGER_SIM = tuple(range(SIM_IX, SIM_LT + 1))
FINGER_REAL = tuple(SIM_TO_REAL[s] for s in FINGER_SIM)


@dataclass
class GraspReport:
    """抓取结果。fingers/thumb_* 都是真机实测的归一化深度。"""
    fingers: list = field(default_factory=list)   # 四指(仿真序)
    thumb_rot: float = 0.0
    thumb_bend: float = 0.0
    contacts: dict = field(default_factory=dict)  # 真机电机下标 → 接触时电流
    timeout_phases: list = field(default_factory=list)

    def text(self) -> str:
        rows = ["  部位        深度    接触"]
        for k, s in enumerate(FINGER_SIM):
            real = FINGER_REAL[k]
            c = self.contacts.get(real)
            rows.append(f"  {DOF_CN[s]:8s} {self.fingers[k]:7.2f}   "
                        + (f"{c}mA" if c else "—（走完全程，没碰到）"))
        for name, val, real in (("拇指旋转", self.thumb_rot, SIM_TO_REAL[SIM_TR]),
                                ("拇指弯曲", self.thumb_bend, SIM_TO_REAL[SIM_TB])):
            c = self.contacts.get(real)
            rows.append(f"  {name:8s} {val:7.2f}   "
                        + (f"{c}mA" if c else "—"))
        spread = max(self.fingers) - min(self.fingers)
        rows.append(f"  四指深度差 {spread:.2f}"
                    + ("（差异明显 → 物体不规则）" if spread > 0.08
                       else "（四指基本同深 → 物体规则或空抓）"))
        return "\n".join(rows)


def _pump(sync, sender, hand, cond, timeout: float, hz: float = 30.0) -> bool:
    """按 hz 下发,直到 cond() 为真或超时。返回是否满足条件。"""
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if not sync.alive():
            return False
        if sender is not None and hand is not None:
            sender.tick(hand)
        if cond():
            return True
        time.sleep(1.0 / hz)
    return False


def _arrived(sync, idx, want, tol=ARRIVE_TOL):
    return abs(float(sync.action()[idx]) - want) <= tol


def grasp(sync: SimSync, sender: FollowSender | None, hand: ROHand | None,
          cal: Calibration, close: float = 1.0, rot: float = 0.85,
          bend: float = 0.75, contact_ma: int = CONTACT_MA,
          slew: float = GRASP_SLEW, on_log=print) -> GraspReport:
    """按 ①②③ 的时序抓一次。sender/hand 为 None 时只跑仿真。"""
    rep = GraspReport()
    old_slew, sync.slew = sync.slew, slew
    old_stall = None
    if sender is not None:
        old_stall, sender.stall_ma = sender.stall_ma, contact_ma
        sender.resync()                 # 清掉上一次的钉住状态

    def note_contacts():
        if sender is None:
            return
        for i in sender.stalled:
            rep.contacts.setdefault(i, sender.currents[i])

    try:
        # ① 张到最大开口。拇指旋转 0 = 张开位,与四指张角最大,物体才进得去。
        on_log("① 张开到最大开口（拇指让到最外）")
        sync.set_target([0.0] * N_MOTORS)
        if not _pump(sync, sender, hand,
                     lambda: _arrived(sync, SIM_TR, 0.0)
                     and _arrived(sync, SIM_TB, 0.0)
                     and all(_arrived(sync, s, 0.0) for s in FINGER_SIM), 6.0):
            rep.timeout_phases.append("张开")
        if hand is not None:
            on_log(f"   到位，可以放物体了。真机 {hand.read_pos()}")

        # ② 四指同时合拢。谁抵住物体谁自己停 —— 不规则物体上每根停在不同深度。
        on_log(f"② 四指同时合拢（目标 {close:.2f}，接触阈值 {contact_ma}mA）")
        tgt = [close] * 4 + [0.0, 0.0]
        sync.set_target(tgt)

        def fingers_done():
            note_contacts()
            if sender is not None:
                pinned = set(sender.stalled)
                if all(r in pinned for r in FINGER_REAL):
                    return True         # 四根全抵住了
            return all(_arrived(sync, s, close) for s in FINGER_SIM)

        if not _pump(sync, sender, hand, fingers_done, 12.0):
            rep.timeout_phases.append("合拢")
        note_contacts()
        if sender is not None:
            got = [MOTOR_CN[i] for i in sender.stalled if i in FINGER_REAL]
            on_log(f"   抵住的手指: {got or '无（都走完了全程）'}")

        # ③ 拇指对掌 + 压紧。SimSync 的顺序互锁保证它一定排在四指之后。
        on_log(f"③ 拇指对掌到 {rot:.2f}")
        tgt[SIM_TR] = rot
        sync.set_target(tgt)
        if not _pump(sync, sender, hand,
                     lambda: (note_contacts() or _arrived(sync, SIM_TR, rot)), 8.0):
            rep.timeout_phases.append("对掌")

        bend = min(bend, THUMB_BEND_MAX)
        on_log(f"④ 拇指压紧到 {bend:.2f}（上限 {THUMB_BEND_MAX}）")
        tgt[SIM_TB] = bend
        sync.set_target(tgt)
        tb_real = SIM_TO_REAL[SIM_TB]

        def thumb_done():
            note_contacts()
            if sender is not None and tb_real in sender.stalled:
                return True
            return _arrived(sync, SIM_TB, bend)

        if not _pump(sync, sender, hand, thumb_done, 8.0):
            rep.timeout_phases.append("压紧")
        note_contacts()
    finally:
        sync.slew = old_slew
        if sender is not None and old_stall is not None:
            sender.stall_ma = old_stall

    # 报告用**真机实测**位置,不是指令 —— 指令永远等于目标,看不出物体形状。
    # 采样前先稳一下:阶段结束判据看的是仿真 ctrl 到位,而真机滞后约 70ms,
    # 紧接着采样会偏浅(实测拇指命令 0.75 报出 0.69)。报告是物体轮廓的测量
    # 结果,这点偏差不能留。
    if hand is not None:
        _pump(sync, sender, hand, lambda: False, 0.5)
        note_contacts()
        pos = hand.read_pos()
        norms_real = [cal.norm_from_pos(i, p) for i, p in enumerate(pos)]
    else:
        a = sync.action()
        norms_real = [0.0] * N_MOTORS
        for s in range(N_MOTORS):
            norms_real[SIM_TO_REAL[s]] = float(a[s])
    rep.fingers = [norms_real[r] for r in FINGER_REAL]
    rep.thumb_rot = norms_real[SIM_TO_REAL[SIM_TR]]
    rep.thumb_bend = norms_real[SIM_TO_REAL[SIM_TB]]
    return rep


def release(sync: SimSync, sender: FollowSender | None, hand: ROHand | None,
            on_log=print):
    """松开。先松拇指再张四指 —— 拇指压在四指上时直接张四指会顶到拇指。"""
    on_log("松开：先退拇指，再张四指")
    if sender is not None:
        sender.resync()                         # 解除所有钉住
    a = sync.action()
    sync.set_target([a[0], a[1], a[2], a[3], a[SIM_TR], 0.0])
    _pump(sync, sender, hand, lambda: _arrived(sync, SIM_TB, 0.0), 5.0)
    sync.set_target([0.0] * N_MOTORS)
    _pump(sync, sender, hand,
          lambda: all(_arrived(sync, s, 0.0) for s in range(N_MOTORS)), 8.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--close", type=float, default=1.0, help="四指闭合上限")
    ap.add_argument("--rot", type=float, default=0.85, help="拇指对掌位")
    ap.add_argument("--bend", type=float, default=0.75, help="拇指压紧量")
    ap.add_argument("--contact", type=int, default=CONTACT_MA,
                    help="接触判据 mA,越小越灵敏")
    ap.add_argument("--slew", type=float, default=GRASP_SLEW, help="合拢速度")
    ap.add_argument("--hold", type=float, default=3.0, help="抓住后保持秒数")
    ap.add_argument("--no-hand", action="store_true", help="只跑仿真")
    ap.add_argument("--left", action="store_true")
    args = ap.parse_args()

    cal = Calibration.load()
    sync = SimSync(left=args.left, on_event=lambda *a: print("  ·", *a))
    sync.start()
    if not sync.wait_ready():
        print("✗ 仿真没能启动")
        return 1
    watchdog = threading.Timer(240, sync.stop)
    watchdog.daemon = True
    watchdog.start()

    hand = sender = None
    if not args.no_hand:
        hand = ROHand().open()
        print(f"已连接真机 {hand.versions()}")
        sender = FollowSender(cal, sync,
                              on_warn=lambda m: print(f"  ⚠ {m}"))
        align_sim_to_hand(sync, cal, hand)

    try:
        rep = grasp(sync, sender, hand, cal, close=args.close, rot=args.rot,
                    bend=args.bend, contact_ma=args.contact, slew=args.slew)
        print()
        print("抓取结果：")
        print(rep.text())
        if rep.timeout_phases:
            print(f"  ⚠ 超时阶段: {rep.timeout_phases}")
        print()
        print(f"保持 {args.hold:.0f} 秒 ...")
        _pump(sync, sender, hand, lambda: False, args.hold)
        release(sync, sender, hand)
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        sync.stop()
        if hand is not None:
            try:
                hand.stop()
                hand.move(cal.pose_from_norm([0.0] * N_MOTORS), speed=140)
                hand.wait_settled()
                print("已回到张开姿态:", hand.read_pos())
            except Exception:                               # noqa: BLE001
                pass
            hand.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
