"""Optional MuJoCo runner and CLI simulation-to-hardware adapter.

MotionController and SafeSender in control_core.py contain the shared motion
and feedback protection. The GUI's manual/vision modes do not require this
module. SimSync owns model/data and renders either embedded frames or a native
viewer. push_external() provides optional target/feedback visualization.

CLI examples:
    python sim_sync.py --no-hand --demo
    python sim_sync.py --gesture fist
    python sim_sync.py --demo --mode angle

This optional adapter requires the parent simulation workspace and models.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import Calibration                               # noqa: E402
from gestures import GESTURES
from rohand import N_MOTORS, ROHand                      # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from joint_mapping import SIM_TO_REAL
from sim_preview import SimulationPreview                         # noqa: E402
from lesson1_gestures import DOF, ctrl_limits, load                # noqa: E402

__all__ = ["SimSync", "FollowSender", "DOF", "DOF_CN", "align_sim_to_hand",
           "reorder_real_to_sim", "reorder_sim_to_real"]

from control_core import (MotionController, SafeSender, DOF_CN, DEFAULT_SLEW,
                          DEFAULT_HZ, DEFAULT_SPEED, FINGER_LAG_COUNTS,
                          align_control_to_hand, SIM_IX, SIM_MD, SIM_RG, SIM_LT,
                          SIM_TR, SIM_TB, THUMB_BEND_MAX, THUMB_BEND_WHILE_MOVING,
                          THUMB_BEND_SAFE, THUMB_ROT_MOVE, SCRIPT_TOL, DEADBAND,
                          CURRENT_WARN_MA, STALL_MA, STALL_RELEASE, FINGER_GATE_TOL)
from joint_mapping import reorder_real_to_sim, reorder_sim_to_real
FollowSender = SafeSender
align_sim_to_hand = align_control_to_hand


class SimSync(threading.Thread, MotionController):
    """独占 MuJoCo 模型的线程:步进、显示、把目标限速后喂给 ctrl。

    对外暴露的都是**归一化量(0=张开 1=打满,仿真序)**,和 ctrlrange 解耦 ——
    换模型只要 ctrlrange 变了,上层代码不用动。

    模型加载放在 run() 里而不是 __init__:官方模型带 23MB mesh,加载要一秒多,
    在 GUI 主线程里做会明显卡一下。构造后先 start(),再 wait_ready()。
    """

    def __init__(self, left: bool = False, slew: float = DEFAULT_SLEW,
                 fps: int = 60, on_event=None, *, display="window"):
        super().__init__(daemon=True, name="SimSync")
        if display not in ("window", "embedded"):
            raise ValueError(f"未知仿真显示模式: {display}")
        self.left = left
        self.slew = slew
        self.fps = fps
        self.on_event = on_event or (lambda *a: None)
        # 命令行保留原生 viewer，GUI 显式选择 embedded。
        self.preview = SimulationPreview() if display == "embedded" else None

        self.model = self.data = None
        self.lo = self.hi = self.span = None

        MotionController.__init__(self, slew=slew)

        self._ready = threading.Event()
        # 不要命名为 _stop：threading.Thread.join() 内部会调用同名方法。
        self._stop_event = threading.Event()
        self._err = None

    # ---------------- 生命周期 ----------------

    def run(self):
        try:
            self.model, self.data = load(left=self.left)
            self.lo, self.hi = ctrl_limits(self.model)
            self.span = np.where(self.hi - self.lo == 0, 1.0, self.hi - self.lo)
            if self._stop_event.is_set():
                return
            if self.preview is not None:
                with self.preview.rendering(self.model) as refresh:
                    self._run_steps(lambda: refresh(self.data))
            else:
                import mujoco.viewer
                # CLI 调试入口保留原生鼠标交互和调试面板。
                with mujoco.viewer.launch_passive(
                        self.model, self.data, show_left_ui=False) as viewer:
                    self._run_steps(viewer.sync, viewer.is_running)
        except Exception as exc:                            # noqa: BLE001
            self._err = exc
            self.on_event("sim_error", f"仿真启动或运行失败: {exc}")
        finally:
            self._stop_event.set()
            self._ready.set()
            self.on_event("sim_closed")

    def _run_steps(self, refresh, is_running=lambda: True):
        import mujoco
        dt = self.model.opt.timestep
        frame_dt = 1.0 / self.fps
        # 一帧推进多少步。dt=2ms、60fps → 8 步。逐步 sleep 不划算:每次 sleep
        # 都有系统调用开销,成批推进再睡一次稳得多。
        n_steps = max(1, round(frame_dt / dt))

        if self._stop_event.is_set():
            return
        mujoco.mj_forward(self.model, self.data)
        refresh()  # 首帧成功后才通知就绪；渲染失败走统一错误 / 退出路径。
        self._ready.set()
        self.on_event("sim_ready")
        next_frame = time.perf_counter()
        while not self._stop_event.is_set() and is_running():
            for _ in range(n_steps):
                self.data.ctrl[:] = self.lo + self._advance(dt) * self.span
                mujoco.mj_step(self.model, self.data)
            refresh()
            next_frame += frame_dt
            slack = next_frame - time.perf_counter()
            if slack > 0:
                self._stop_event.wait(slack)
            else:
                next_frame = time.perf_counter()

    def wait_ready(self, timeout: float = 60.0) -> bool:
        return self._ready.wait(timeout) and self._err is None and self.alive()

    def alive(self) -> bool:
        return self.is_alive() and not self._stop_event.is_set()

    def stop(self):
        self._stop_event.set()


# ---------------- 命令行 ----------------

def _drive(sync, sender, hand, hz):
    """跑一拍:下发 + 节流。返回仿真窗口是否还开着。"""
    started = time.monotonic()
    if sender is not None and hand is not None:
        sender.tick(hand)
    time.sleep(max(0.0, 1.0 / hz - (time.monotonic() - started)))
    return sync.alive()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gesture", help="仿真里演一个手势(真机跟随)")
    ap.add_argument("--demo", action="store_true", help="依次演全部手势")
    ap.add_argument("--mode", choices=["normalized", "angle"],
                    default="normalized", help="仿真→真机的对齐方式")
    ap.add_argument("--slew", type=float, default=DEFAULT_SLEW,
                    help="限速,归一化/秒")
    ap.add_argument("--hz", type=float, default=DEFAULT_HZ, help="下发频率")
    ap.add_argument("--speed", type=int, choices=range(1, 256),
                    default=DEFAULT_SPEED, metavar="1..255", help="真机电机速度上限")
    ap.add_argument("--left", action="store_true", help="用左手模型")
    ap.add_argument("--no-hand", action="store_true", help="只跑仿真,不连真机")
    args = ap.parse_args()
    if args.hz <= 0:
        ap.error("--hz 必须大于 0")

    names = (list(GESTURES) if args.demo
             else ([args.gesture] if args.gesture else []))
    unknown = [n for n in names if n not in GESTURES]
    if unknown:
        print(f"未知手势 {unknown};可用: {list(GESTURES)}")
        return 2

    sync = SimSync(left=args.left, slew=args.slew,
                   on_event=lambda *a: print("  ·", *a))
    sync.start()
    if not sync.wait_ready():
        print("✗ 仿真没能启动")
        return 1
    print(f"仿真已启动({'左' if args.left else '右'}手),关掉 MuJoCo 窗口即结束")

    hand = None
    sender = None
    if not args.no_hand:
        cal = Calibration.load()
        hand = ROHand().open()
        print(f"已连接真机 {hand.versions()}")
        align_sim_to_hand(sync, cal, hand)      # 先对齐,避免使能瞬间跳变
        sender = FollowSender(cal, sync, mode=args.mode, speed=args.speed,
                              on_warn=lambda m: print(f"  ⚠ {m}"))

    try:
        for n in names:
            g = GESTURES[n]
            print(f"▶ {g.name_cn} ({g.name}) — {g.desc}")
            if sync.play_gesture(g) and sender is not None:
                print("    (该手势含「碰到就停」步骤,跟随模式下不生效)")
            while sync.busy() and _drive(sync, sender, hand, args.hz):
                pass
            if not sync.alive():
                break
        if sync.alive():
            print("手势播完,窗口还开着 —— 可以继续用鼠标看,关窗退出")
        while _drive(sync, sender, hand, args.hz):
            pass
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        sync.stop()
        if hand is not None:
            try:
                hand.stop()
                hand.move(Calibration.load().pose_from_norm([0.0] * N_MOTORS),
                          speed=140)
                hand.wait_settled()
            except Exception:                               # noqa: BLE001
                pass
            hand.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
