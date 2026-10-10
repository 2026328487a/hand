"""把 MuJoCo 仿真的 action 输出接到真机上。

仿真和真机的通道顺序**不一样**,这是接线时第一个要处理的:

    仿真 XML 执行器顺序          真机 GetFingerPosAll 顺序
    0 index_finger              0 thumb_bend
    1 middle_finger             1 index
    2 ring_finger               2 middle
    3 little_finger             3 ring
    4 thumb_root   (旋转)        4 little
    5 thumb        (弯曲)        5 thumb_rot

两套坐标怎么对齐,有两种做法,这里都实现了:

**normalized(默认)** —— 双方都用 0=张开 1=打满,直接按比例映射。简单、稳,
    仿真里训出来的归一化 action 可以直接发。代价是两边的"打满"物理量不同:
    仿真 index 的 ctrlrange 是 1.57 rad(90°),真机实测行程只有 73.36°,
    所以仿真的满弯比真机多约 17°。

**angle** —— 按实际关节角对齐。仿真的关节角是"从伸直开始弯了多少"(0 = 伸直),
    真机报的是绝对角(伸直 ≈ 173.83°),所以 real_deg = open_deg - sim_rad。
    物理上更诚实,但**仿真能弯到的角度真机到不了**,超出部分会被夹到限位。

用法::

    python sim_bridge.py --list                    # 看仿真侧的手势
    python sim_bridge.py --gesture 握拳             # 单个手势下发真机
    python sim_bridge.py --mirror                  # 跑仿真轨迹,实时镜像到真机
    python sim_bridge.py --compare                 # 量化 sim2real 差距
    python sim_bridge.py --mirror --dry-run        # 只打印,不动手
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from calibration import Calibration                     # noqa: E402
from lesson1_gestures import (DOF, GESTURES, action_of,  # noqa: E402
                              ctrl_limits, load)
from rohand import MOTORS, N_MOTORS, HandError, ROHand   # noqa: E402

# 仿真 DOF index → 真机电机 index。
# 仿真的 thumb_root 是**旋转**、thumb 是**弯曲**,别被名字骗了。
SIM_TO_REAL = [
    1,   # index_finger  → index
    2,   # middle_finger → middle
    3,   # ring_finger   → ring
    4,   # little_finger → little
    5,   # thumb_root    → thumb_rot
    0,   # thumb         → thumb_bend
]
REAL_TO_SIM = [SIM_TO_REAL.index(i) for i in range(N_MOTORS)]


def reorder_sim_to_real(vec):
    """长度 6 的仿真序向量 → 真机序。"""
    out = [0.0] * N_MOTORS
    for sim_i, real_i in enumerate(SIM_TO_REAL):
        out[real_i] = vec[sim_i]
    return out


def pose_normalized(cal: Calibration, action) -> list:
    """仿真归一化 action(0..1,仿真序) → 真机指令位置。"""
    return cal.pose_from_norm(reorder_sim_to_real(action))


def pose_by_angle(cal: Calibration, action, hi) -> list:
    """按实际关节角对齐。action 归一化,hi 是各执行器 ctrlrange 上限(rad)。

    仿真角是相对伸直的弯曲量,真机报绝对角。四指和拇指弯曲的真机角随弯曲
    **减小**,拇指旋转**增大**,所以方向由标定里的 open/closed 决定,不能写死。
    """
    sim_rad = np.asarray(action) * np.asarray(hi)
    sim_deg = np.degrees(sim_rad)
    degs = [0.0] * N_MOTORS
    for sim_i, real_i in enumerate(SIM_TO_REAL):
        open_deg, closed_deg = cal.deg_span(real_i)
        sign = 1.0 if closed_deg > open_deg else -1.0   # 弯曲时角度的走向
        degs[real_i] = open_deg + sign * sim_deg[sim_i]
    return cal.pose_from_deg(degs)


def build_pose(cal, action, hi, mode: str) -> list:
    return (pose_by_angle(cal, action, hi) if mode == "angle"
            else pose_normalized(cal, action))


# ---------------- 各个模式 ----------------

def cmd_gesture(args, cal, model, hi):
    wanted = [(n, g) for n, g in GESTURES if n == args.gesture]
    if not wanted:
        print(f"未知手势 {args.gesture};可用: {sorted({n for n, _ in GESTURES})}")
        return 2
    label, gesture = wanted[0]
    action = action_of(gesture)
    pose = build_pose(cal, action, hi, args.mode)
    print(f"{label}: 仿真 action(仿真序) {np.round(action, 2).tolist()}")
    print(f"     → 真机 pos(真机序) {pose}")
    if args.dry_run:
        return 0
    with ROHand() as hand:
        hand.move(cal.pose_from_norm([0.0] * N_MOTORS), speed=140)
        hand.wait_settled()
        hand.move(pose, speed=args.speed)
        hand.wait_settled()
        time.sleep(0.6)
        report_gap(hand, cal, action, hi)
        hand.move(cal.pose_from_norm([0.0] * N_MOTORS), speed=140)
        hand.wait_settled()
    return 0


def cmd_mirror(args, cal, model, hi):
    """跑仿真,把每一帧的 ctrl 实时下发真机。

    仿真步长是 2ms,而真机一次 SetFingerPosAll + 读回要几十毫秒 —— 全量下发
    会把仿真拖成慢动作。所以按 --rate 抽帧,只发得过来的那部分。
    """
    from lesson1_gestures import make_trajectory
    import mujoco

    data = mujoco.MjData(model)
    ctrl, labels = make_trajectory(model, hold=args.hold, blend=args.blend)
    lo, hi_arr = ctrl_limits(model)
    span = np.where(hi_arr - lo == 0, 1, hi_arr - lo)
    dt = model.opt.timestep
    every = max(1, round(1.0 / args.rate / dt))
    print(f"仿真 {len(ctrl)} 步 × {dt * 1000:.0f}ms,每 {every} 步下发一次 "
          f"(约 {args.rate}Hz),映射模式 {args.mode}")

    hand = None
    sent = 0
    if not args.dry_run:
        hand = ROHand().open()
        hand.move(cal.pose_from_norm([0.0] * N_MOTORS), speed=140)
        hand.wait_settled()
    try:
        last_label = None
        last_check = 0.0
        peak = [0] * N_MOTORS
        t_start = time.perf_counter()
        for i in range(len(ctrl)):
            data.ctrl[:] = ctrl[i]
            mujoco.mj_step(model, data)
            if i % every:
                continue
            action = (ctrl[i] - lo) / span          # ctrl → 归一化
            pose = build_pose(cal, action, hi_arr, args.mode)
            if hand is not None:
                hand.move(pose, speed=args.speed)
                sent += 1
            if labels[i] != last_label:
                last_label = labels[i]
                print(f"  [{time.perf_counter() - t_start:5.1f}s] {last_label:6s} "
                      f"action{np.round(action, 2).tolist()} → pos{pose}")
            # 电流监视。仿真的手势是六路**同时**打满的,真机上拇指旋转与四指
            # 闭合同时进行会互相蹭 —— gestures.py 靠分步避开了,直接喂仿真
            # action 没有这层保护,所以这里盯着电流。读一路要一次往返,全读
            # 会拖垮下发节奏,所以限频、且只读一次就够(报错会挤掉下发帧)。
            now = time.perf_counter()
            if hand is not None and now - last_check > 0.5:
                last_check = now
                cur = hand.read_currents()
                peak = [max(a, b) for a, b in zip(peak, cur)]
                hot = [(MOTORS[j], c) for j, c in enumerate(cur) if c > 500]
                if hot:
                    print(f"  [{now - t_start:5.1f}s] ⚠ 电流偏高 {hot}")
            # 按仿真时间节流,让真机动作和仿真同速
            target_t = t_start + i * dt
            slack = target_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
        if hand is not None:
            print(f"\n完成为 {sent} 次下发。各路峰值电流(mA): " +
                  " ".join(f"{MOTORS[j]}={peak[j]}" for j in range(N_MOTORS)))
    finally:
        if hand is not None:
            hand.move(cal.pose_from_norm([0.0] * N_MOTORS), speed=140)
            hand.wait_settled()
            hand.close()
    return 0


def report_gap(hand, cal, action, hi):
    """读真机实际角度,和仿真期望角度比。"""
    real_deg = hand.read_angle()
    sim_deg = np.degrees(np.asarray(action) * np.asarray(hi))
    print(f"  {'电机':11s} {'仿真弯曲°':>10s} {'真机弯曲°':>10s} {'差':>8s}")
    for sim_i, real_i in enumerate(SIM_TO_REAL):
        open_deg, closed_deg = cal.deg_span(real_i)
        sign = 1.0 if closed_deg > open_deg else -1.0
        real_bend = (real_deg[real_i] - open_deg) * sign
        print(f"  {MOTORS[real_i]:11s} {sim_deg[sim_i]:10.2f} "
              f"{real_bend:10.2f} {real_bend - sim_deg[sim_i]:8.2f}")


def cmd_compare(args, cal, model, hi):
    """对一组 action 依次下发,量化仿真与真机的角度差。"""
    lo, hi_arr = ctrl_limits(model)
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"映射模式 {args.mode};对每个 level 让**四指**同步弯曲,拇指不动\n")
    print(f"{'level':>6s} " + " ".join(f"{MOTORS[SIM_TO_REAL[i]]:>10s}"
                                       for i in range(4)))
    print(f"{'':>6s} " + " ".join(f"{'仿真/真机':>10s}" for _ in range(4)))

    if args.dry_run:
        for lv in levels:
            action = np.array([lv, lv, lv, lv, 0.0, 0.0])
            print(f"{lv:6.2f} pos={build_pose(cal, action, hi_arr, args.mode)}")
        return 0

    rows = []
    with ROHand() as hand:
        hand.move(cal.pose_from_norm([0.0] * N_MOTORS), speed=140)
        hand.wait_settled()
        for lv in levels:
            action = np.array([lv, lv, lv, lv, 0.0, 0.0])
            hand.move(build_pose(cal, action, hi_arr, args.mode), speed=args.speed)
            hand.wait_settled()
            time.sleep(0.35)
            real_deg = hand.read_angle()
            sim_deg = np.degrees(action * hi_arr)
            cells, gaps = [], []
            for sim_i in range(4):
                real_i = SIM_TO_REAL[sim_i]
                open_deg, closed_deg = cal.deg_span(real_i)
                sign = 1.0 if closed_deg > open_deg else -1.0
                real_bend = (real_deg[real_i] - open_deg) * sign
                cells.append(f"{sim_deg[sim_i]:4.0f}/{real_bend:5.1f}")
                gaps.append(real_bend - sim_deg[sim_i])
            rows.append((lv, gaps))
            print(f"{lv:6.2f} " + " ".join(f"{c:>10s}" for c in cells))
        hand.move(cal.pose_from_norm([0.0] * N_MOTORS), speed=140)
        hand.wait_settled()

    worst = max((abs(g), lv, MOTORS[SIM_TO_REAL[i]])
                for lv, gaps in rows for i, g in enumerate(gaps))
    print(f"\n最大角度差 {worst[0]:.1f}° (level={worst[1]}, {worst[2]})")
    if args.mode == "normalized":
        print("normalized 模式下这个差是**必然的**:仿真 ctrlrange 1.57rad=90°,"
              "真机实测行程 ~73°,\n满弯时差 ~17°。要角度一致就用 --mode angle,"
              "代价是仿真的大弯曲角真机到不了、会被夹到限位。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["normalized", "angle"],
                    default="normalized", help="仿真→真机的对齐方式")
    ap.add_argument("--gesture", help="下发单个仿真手势")
    ap.add_argument("--mirror", action="store_true", help="实时镜像仿真轨迹")
    ap.add_argument("--compare", action="store_true", help="量化 sim2real 差距")
    ap.add_argument("--list", action="store_true", help="列出仿真侧手势")
    ap.add_argument("--rate", type=float, default=20.0, help="镜像下发频率 Hz")
    ap.add_argument("--speed", type=int, default=200, help="真机速度 0-255")
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--blend", type=float, default=0.8)
    ap.add_argument("--left", action="store_true", help="用左手模型")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("仿真侧手势(lesson1_gestures.GESTURES):")
        for label, g in GESTURES:
            print(f"  {label:6s} {dict(g)}")
        print(f"\n仿真 DOF 顺序: {DOF}")
        print(f"真机电机顺序: {list(MOTORS)}")
        print(f"仿真→真机映射: {SIM_TO_REAL}")
        return 0

    model, _ = load(left=args.left)
    _, hi = ctrl_limits(model)
    cal = Calibration.load()

    if args.gesture:
        return cmd_gesture(args, cal, model, hi)
    if args.mirror:
        return cmd_mirror(args, cal, model, hi)
    if args.compare:
        return cmd_compare(args, cal, model, hi)
    ap.print_help()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HandError, KeyboardInterrupt) as exc:
        print(f"\n中断: {exc}")
        try:
            with ROHand() as h:
                h.stop()
                h.move(Calibration.load().pose_from_norm([0.0] * N_MOTORS),
                       speed=140)
                h.wait_settled()
        except Exception:                       # noqa: BLE001
            pass
        sys.exit(1)
