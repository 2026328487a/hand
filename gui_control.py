"""ROH-LiteS001 真机控制 GUI。

任务面板以自适应抓握实验为主:四指同时接触、拇指对掌、保持并记录接触轮廓。
界面不是官方的 OHandSetting —— 那个是厂商调试用。

线程模型:tkinter 主线程只画界面 + 用 after() 轮询结果队列,不碰 CAN。
一个 worker 线程独占连好的手(PCAN 通道一次只能一个进程用),串行处理命令。
这是必要的:一次 move() 往返虽只要 1.8ms,但手势是分步的,每一步都要等机械
到位,直接在主线程做会卡死界面。

用法::

    python gui_control.py                # 正常启动
    python gui_control.py --selftest     # 只验证界面能开,1 秒后自动关(不连手)

布局:
    [顶部] 连接面板: 连接/断开按钮 + 状态灯 + 手信息
    [左侧] 安全按钮: 张开 / 急停(手指停住,不收回)
    [中间] 自适应抓握任务流程与单次结果
    [右侧] 摄像头 / 视频实时预览 + 电机状态面板
    [底部] 日志
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from datetime import datetime
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import Calibration                     # noqa: E402
from gestures import GESTURES, play                     # noqa: E402
from rohand import MOTORS, MOTOR_CN, N_MOTORS, HandError, ROHand  # noqa: E402
from task_mode import (DEFAULT_CLOSE_NORM, DEFAULT_CONTACT_MA,
                       perform_adaptive_grasp)            # noqa: E402

# 仿真模块依赖上级目录的 MuJoCo 模型。GUI 本身仍然应该能在没有仿真工作区
# 时启动，所以这里故意延迟失败；仿真面板会显示具体原因，而不影响真机面板。
try:                                                       # noqa: E402
    from sim_sync import (DOF_CN, FollowSender, SimSync,
                          align_sim_to_hand, reorder_real_to_sim)
    SIM_IMPORT_ERROR = None
except Exception as _sim_exc:                              # noqa: E402
    DOF_CN = ("食指", "中指", "无名指", "小指", "拇指旋转", "拇指弯曲")
    FollowSender = SimSync = align_sim_to_hand = None
    reorder_real_to_sim = None
    SIM_IMPORT_ERROR = _sim_exc

# 视觉模块再额外依赖 mediapipe + opencv + 模型文件。它比仿真更容易缺,所以
# **单独**一个 try —— 没装 mediapipe 不应该连仿真面板一起拖下水。
try:                                                       # noqa: E402
    from vision_hand import VisionSource
    VISION_IMPORT_ERROR = None
except Exception as _vis_exc:                              # noqa: E402
    VisionSource = None
    VISION_IMPORT_ERROR = _vis_exc

# 命令队列:主线程放进去,worker 取出来执行。取出来后串行,不会并发。
# 事件队列:worker 把状态/进度放进来,主线程 after() 轮询刷新界面。
CODES = ("connect", "disconnect", "stop", "open")
FOLLOW_HZ = 30
FOLLOW_SPEED = 140


def step_progress_to_str(i: int, total: int, norms, pose, contact) -> str:
    n = " ".join(f"{v:.2f}" for v in norms)
    p = " ".join(str(v) for v in pose)
    tag = f"  [接触停住 {contact}]" if contact else ""
    return f"步骤{i}/{total}  norm[{n}]  →  pos [{p}]{tag}"


class HandWorker(threading.Thread):
    """独占手的后台线程。主线程通过 put(cmd) 发命令,通过 events 收反馈。

    cmd 可以是 CODES 里的字符串,或一个手势名。手势名会解析成 GESTURES[name]
    并经 play() 执行。命令串行,前一个完成了才做下一个。
    """

    def __init__(self, cal: Calibration, commands, events):
        super().__init__(daemon=True)
        self.cal = cal
        self.commands = commands      # queue.Queue
        self.events = events          # queue.Queue  (主线程 after 轮询)
        self.hand: ROHand | None = None
        self._idle = False
        self._connected = False
        self.sim = None
        self.sim_mode = "off"
        self.sender = None
        self.follow_speed = FOLLOW_SPEED
        self._reverse_t = 0.0
        self.vision = None          # 视频驱动源,见 _start_vision
        self._pose_t = 0.0
        self._last_stalled = ()     # 上次上报的「顶住被钉住」的电机
        self._task_cancel = threading.Event()
        self._active_grasp = None
        self._grasp_started_t = None

    # ----- 给主线程的便捷方法 -----
    @property
    def connected(self):
        return self._connected

    def put(self, cmd):
        if cmd in ("stop", "disconnect", "__quit__"):
            self._task_cancel.set()
        self.commands.put(cmd)

    # ----- 线程主体 -----
    def run(self):
        period = 1.0 / FOLLOW_HZ
        next_tick = time.monotonic() + period
        while True:
            try:
                # CAN 耗时包含在周期里，不能在读写完成后再固定等 30ms。
                cmd = self.commands.get(timeout=max(0.0, next_tick - time.monotonic()))
            except queue.Empty:
                self._sim_tick()
                self._maybe_report_state()
                next_tick += period
                if next_tick <= time.monotonic():
                    # CAN 已耗尽周期，只取下一拍的最新目标，不追补历史帧。
                    next_tick = time.monotonic()
                continue
            if cmd == "connect":
                self._do_connect()
            elif cmd == "disconnect":
                self._do_disconnect()
            elif cmd == "stop":
                self._do_stop()
            elif cmd == "open":
                self._do_open()
            elif isinstance(cmd, tuple) and cmd:
                if cmd[0] in ("task_grasp", "task_release"):
                    self._do_task_command(cmd)
                else:
                    self._do_sim_command(cmd)
            elif cmd == "__quit__":
                self._do_disconnect()
                self._stop_sim()
                return
            if time.monotonic() >= next_tick:
                self._sim_tick()
                self._maybe_report_state()
                next_tick += period
                if next_tick <= time.monotonic():
                    next_tick = time.monotonic()

    # ----- 仿真 / 真机同步 -----
    def _send_sim(self, *payload):
        self._send(("sim",) + payload)

    def _do_sim_command(self, cmd):
        op = cmd[0]
        if op == "sim_start":
            self._start_sim()
        elif op == "sim_stop":
            self._stop_sim()
        elif op == "sim_mode":
            self._set_sim_mode(cmd[1])
        elif op == "follow_speed":
            self.follow_speed = max(1, min(255, int(cmd[1])))
            if self.sender is not None:
                self.sender.speed = self.follow_speed
            self._send_sim("log", f"真机跟随速度: {self.follow_speed}/255")
        elif op == "vision_start":
            self._start_vision(*cmd[1:])
        elif op == "vision_stop":
            self._stop_vision()
        elif op == "vision_event":
            self._on_vision_event(*cmd[1:])
        elif op == "sim_event":
            self._handle_sim_lifecycle(*cmd[1:])

    def _start_vision(self, source, loop=True, preview=False, mirror=False, handedness="any"):
        if VisionSource is None:
            self._send_sim("vision_error", f"视觉模块不可用: {VISION_IMPORT_ERROR}")
            return
        if self.vision is not None and self.vision.is_alive():
            self._send_sim("log", "视频源仍在运行或释放资源，请停止后等待再启动")
            return
        if self.sim_mode == "reverse":
            self._send_sim("vision_error", "reverse 模式下不能启动视频，请先切换到 off")
            return
        self._pause_video_follow("切换视频源，真机跟随已暂停")
        try:
            vision = VisionSource(
                source, self.sim, loop=bool(loop), preview=bool(preview),
                mirror=mirror, handedness=handedness,
                on_event=lambda kind, *args: self.put(("vision_event", vision, kind, args)))
            self.vision = vision
            self._send_sim("vision_starting")
            vision.start()
        except Exception as exc:
            self.vision = None
            self._send_sim("vision_error", str(exc))

    def _pause_video_follow(self, reason):
        """Only the CAN owner may stop the physical hand. Never auto-resume."""
        if self.sim_mode != "follow":
            return
        self._set_sim_mode("off")
        if self.hand is not None:
            try:
                self.hand.stop()
            except Exception as exc:
                self._send_sim("error", f"停止命令失败: {exc}")
        self._send_sim("warn", reason)

    def _stop_vision(self):
        self._pause_video_follow("视频已停止，真机跟随已暂停")
        if self.vision is not None:
            self.vision.stop()
            self._send_sim("vision_stopping")
        # No join here: MediaPipe cleanup must never block CAN or the emergency stop.

    def _on_vision_event(self, vision, kind, args):
        # Events from an earlier source must not change a restarted stream.
        if vision is not self.vision:
            return
        if kind in ("vision_error", "vision_stopping", "vision_closed"):
            self._pause_video_follow("视频源结束或异常，真机跟随已暂停")
        if kind != "vision_closed":
            self._send_sim(kind, *args)
        # Final closed UI event is emitted only after is_alive() becomes false.

    def _publish_vision(self):
        if self.vision is None:
            return
        now = time.monotonic()
        if now - self._pose_t < 0.1:
            return
        self._pose_t = now
        self._send_sim("vision_state", self.vision.snapshot())
        if self.sim is not None and self.vision.alive():
            self._send_sim("pose", list(self.sim.action()))
        if not self.vision.is_alive():
            self._send_sim("vision_closed")
            self.vision = None

    def _start_sim(self):
        if self.sim is not None and self.sim.alive():
            self._send_sim("log", "MuJoCo 仿真已经在运行")
            return
        if self.sim is not None and self.sim.is_alive():
            self._send_sim("error", "仿真窗口正在关闭，请稍后再启动")
            return
        self.sim = None
        if SimSync is None:
            self._send_sim("error", f"仿真不可用: {SIM_IMPORT_ERROR}")
            return
        try:
            sim = SimSync(on_event=lambda kind, *args: self.put(("sim_event", sim, kind, args)))
            self.sim = sim
            self.sim.start()
            if not self.sim.wait_ready(30.0):
                detail = getattr(self.sim, "_err", None)
                self.sim = None
                raise RuntimeError(detail or "窗口没有正常启动")
            self.sim_mode = "off"
            if self.vision is not None:
                self.vision.set_sync(self.sim)
            self._send_sim("ready")
        except Exception as exc:
            self.sim = None
            self._send_sim("error", f"启动仿真失败: {exc}")

    def _stop_sim(self):
        self._stop_vision()          # 视频源抓着 sim 的引用,必须先停
        sim = self.sim
        if sim is not None:
            sim.stop()
            if sim is not threading.current_thread():
                sim.join(timeout=3.0)
            if sim.is_alive():
                self._send_sim("error", "仿真线程仍在退出，暂不能重新启动")
                return
            if self.sim is sim:
                self.sim = None
        self.sim_mode = "off"
        self.sender = None
        self._send_sim("closed")

    def _set_sim_mode(self, mode):
        if mode not in ("off", "follow", "reverse"):
            return
        if mode != "off" and self.sim is None:
            self._send_sim("error", "请先打开 MuJoCo 仿真窗口")
            self._send_sim("mode", "off")
            return
        if mode != "off" and not self._connected:
            self._send_sim("error", "同步模式需要先连接真机")
            self._send_sim("mode", "off")
            return
        if mode == "follow":
            if self.vision is not None and not self.vision.snapshot()["control_ready"]:
                self._pause_video_follow("手部识别尚未就绪，真机跟随已暂停")
                self._send_sim("error", "请先让控制手进入画面并识别成功，再启用 follow")
                self._send_sim("mode", self.sim_mode)
                return
            try:
                align_sim_to_hand(self.sim, self.cal, self.hand)
                # 对齐把仿真硬拉到了真机当前姿态,滑杆必须跟着走。不然滑杆还
                # 停在旧值上,下一次拖动是从旧值起跳 —— 使能瞬间不跳,第一次
                # 碰滑杆才跳,更难查。
                self._send_sim("pose", list(self.sim.action()))
                self.sender = FollowSender(
                    self.cal, self.sim, speed=self.follow_speed,
                    on_warn=lambda msg: self._send_sim("warn", msg))
                self.sender.resync()
            except Exception as exc:
                self._pause_video_follow("对齐失败，真机跟随已暂停")
                self.sim_mode = "off"
                self.sender = None
                self._send_sim("error", f"启用仿真→真机失败: {exc}")
                self._send_sim("mode", "off")
                return
        elif mode == "reverse":
            # reverse 让真机当主,视频也想当主 —— 不能并存,切过去就把视频停掉
            self._stop_vision()
            try:
                self.sim.push_external(None)
            except Exception:
                pass
            self.sender = None
        else:
            if self.sim is not None:
                self.sim.push_external(None)
            self.sender = None
        self.sim_mode = mode
        self._last_stalled = ()
        self._send_sim("stalled", [])
        self._send_sim("mode", mode)

    def _handle_sim_lifecycle(self, sim, kind, args):
        if sim is not self.sim:
            return
        self._send_sim(kind, *args)
        if kind == "sim_closed":
            self._stop_vision()
            self.sim_mode = "off"
            self.sender = None
            self.sim = None

    def _sim_tick(self):
        """只在 HandWorker 线程访问 CAN；GUI 不直接读写真机。"""
        if self.vision is not None and self.sim_mode == "follow":
            state = self.vision.snapshot()
            # 短暂丢手时保持最后一个有效目标，避免跟随模式因为遮挡/出画
            # 立刻退回 off。视觉线程本身会保持最后姿态；只有视频线程结束
            # 或发生错误才停止真机。用户仍可随时手动切回 off。
            if not self.vision.alive() or state["error"]:
                self._pause_video_follow("视频源结束或异常，真机跟随已暂停")
        self._publish_vision()
        if self.sim is None or not self.sim.alive():
            self._pause_video_follow("仿真窗口已关闭，真机跟随已暂停")
            return
        if self.sim_mode == "follow" and self.sender is not None and self.hand:
            try:
                self.sender.tick(self.hand)
                # 钉住状态变化时才上报,不然 30Hz 会把事件队列刷满
                stalled = tuple(self.sender.stalled)
                if stalled != self._last_stalled:
                    self._last_stalled = stalled
                    self._send_sim("stalled", list(stalled))
            except Exception as exc:
                self._send_sim("error", f"同步下发失败: {exc}")
                self._pause_video_follow("通信异常，真机跟随已暂停")
        elif self.sim_mode == "reverse" and self.hand:
            now = time.monotonic()
            if now - self._reverse_t < 0.06:
                return
            self._reverse_t = now
            try:
                pos = self.hand.read_pos()
                norms = [self.cal.norm_from_pos(i, p)
                         for i, p in enumerate(pos)]
                # 通道重排**只在这里做一次**,然后把仿真序的结果同时给仿真和
                # 界面。之前界面自己拿 pos 再排一遍,抄错了映射表(掰食指亮的
                # 是中指),而且 App 上根本没有 cal —— 两个 bug 都是"同一份映射
                # 抄了两份"引起的,所以现在只留一份。
                sim_norms = reorder_real_to_sim(norms)
                self.sim.push_external(sim_norms)
                self._send_sim("pose", sim_norms)
            except Exception as exc:
                self._send_sim("error", f"真机→仿真读取失败: {exc}")

    # ----- 各命令实现 -----
    def _do_connect(self):
        if self._connected:
            self._send(("busy", "已连接,请先断开"))
            return
        self._send(("busy", "正在连接真机 ..."))
        try:
            hand = ROHand().open()
        except Exception as exc:
            self._send(("error", f"连接失败: {exc}"))
            return
        try:
            info = hand.versions()
            cali = hand.read_cali_data()
        except Exception as exc:
            hand.close()
            self._send(("error", f"读手信息失败: {exc}"))
            return
        self.hand = hand
        self._connected = True
        self._send(("connected", info, cali))

    def _do_disconnect(self):
        if not self._connected:
            self._task_cancel.clear()
            self._send(("busy", "还没有连接"))
            return
        self._set_sim_mode("off")
        try:
            # 断开前回到张开姿态,避免手停在夹紧状态
            self.hand.move(self.cal.pose_from_norm([0.0] * N_MOTORS), speed=140)
            self.hand.wait_settled()
        except Exception as exc:
            self._send(("busy", f"断开前回张开失败: {exc}"))
        self.hand.close()
        self.hand = None
        self._connected = False
        self._active_grasp = None
        self._grasp_started_t = None
        self._task_cancel.clear()
        self._send(("disconnected", "已断开"))

    def _do_stop(self):
        if not self._connected:
            self._task_cancel.clear()
            self._send(("busy", "还没有连接"))
            return
        try:
            self._set_sim_mode("off")
            self.hand.stop()   # FingerStop: 停在当前位置,不收回
            self._task_cancel.clear()
            self._send(("busy", "已急停(停在当前位置)"))
        except Exception as exc:
            self._send(("error", f"急停失败: {exc}"))

    def _do_open(self):
        if not self._connected:
            self._send(("busy", "还没有连接"))
            return
        if self.sim_mode == "follow":
            self._send(("busy", "仿真→真机同步中，请先切回 off 再执行真机任务"))
            return
        play(self.hand, self.cal, GESTURES["open"], verbose=False,
             on_step=self._stop_cb)
        if self._active_grasp is not None:
            self._active_grasp = None
            self._grasp_started_t = None
            self._send(("task_aborted", "已手动张开，抓握记录未完成"))

    def _do_task_command(self, cmd):
        if cmd[0] == "task_grasp":
            self._do_adaptive_grasp(*cmd[1:])
        elif cmd[0] == "task_release":
            self._do_task_release()

    def _do_adaptive_grasp(self, contact_ma, close_norm):
        if not self._connected or self.hand is None:
            self._send(("task_error", "请先连接真机"))
            return
        if self.sim_mode != "off":
            self._send(("task_error", "请先把仿真同步模式切到 off"))
            return
        if self._active_grasp is not None:
            self._send(("task_error", "当前已有抓握任务；请先松开并记录"))
            return
        self._send(("task_started",))

        def on_progress(phase, label, touched, norms, contacts):
            state = "接触停住" if touched else "到达闭合上限"
            self._send(("task_progress", phase, 3, label, state, norms, contacts))

        try:
            result = perform_adaptive_grasp(
                self.hand, self.cal, contact_ma=contact_ma,
                close_norm=close_norm, on_progress=on_progress,
                should_cancel=self._task_cancel.is_set)
        except InterruptedError as exc:
            self._active_grasp = None
            self._grasp_started_t = None
            self._send(("task_aborted", str(exc)))
            return
        except Exception as exc:
            self._active_grasp = None
            self._grasp_started_t = None
            try:
                self.hand.stop()
            except Exception:
                pass
            self._send(("task_error", str(exc)))
            return

        self._active_grasp = result
        self._grasp_started_t = time.monotonic()
        self._send(("task_holding", result))

    def _do_task_release(self):
        if not self._connected or self.hand is None:
            self._send(("task_error", "请先连接真机"))
            return
        if self._active_grasp is None or self._grasp_started_t is None:
            self._send(("task_error", "当前没有可结束的抓握记录"))
            return
        report = dict(self._active_grasp)
        report["hold_seconds"] = max(0.0, time.monotonic() - self._grasp_started_t)
        self._send(("task_releasing",))
        feedback_error = None
        try:
            report["release_norms"] = [
                self.cal.norm_from_pos(i, p)
                for i, p in enumerate(self.hand.read_pos())]
            report["release_currents_ma"] = [
                int(x) for x in self.hand.read_currents()]
        except Exception as exc:
            feedback_error = str(exc)
        try:
            play(self.hand, self.cal, GESTURES["open"], verbose=False,
                 on_step=self._stop_cb)
        except Exception as exc:
            self._send(("task_error", f"自动松开失败: {exc}"))
            return
        self._active_grasp = None
        self._grasp_started_t = None
        if feedback_error is not None:
            report["release_feedback_error"] = feedback_error
        self._send(("task_released", report))

    def _stop_cb(self, i, total, norms, pose, contact):
        if contact is not None:
            self._send(("step", i, total, step_progress_to_str(
                i, total, norms, pose, contact)))
        else:
            self._send(("step", i, total, step_progress_to_str(
                i, total, norms, pose, None)))

    def _maybe_report_state(self):
        """工作线程空闲时,每 2 秒主动上报一次电机状态,让界面保持活动。

        不放在手势执行期间 —— 那会挤占下发节奏。只在命令队列空的时候做。
        """
        if not self._connected:
            return
        now = time.monotonic()
        if getattr(self, "_last_state_t", 0) and now - self._last_state_t < 2.0:
            return
        self._last_state_t = now
        try:
            if self.sim_mode == "follow" and self.sender is not None:
                # 跟随保护已经读取过反馈；界面回显复用它，不再占一次 CAN。
                pos = list(self.sender.positions)
                cur = list(self.sender.currents)
            else:
                pos = self.hand.read_pos()
                cur = self.hand.read_currents()
            # 状态帧不标注轻量,让主线程去判断有没有人在看
            self._send(("state", pos, cur))
        except Exception as exc:
            self._send(("error", f"读状态失败: {exc}"))

    def _send(self, *payload):
        # 注意:*payload 会把每个位置参数收进元组。调用处是 _send(("busy","..."))
        # 即单个位置参数(一个元组),这里直接 put 会把事件变成
        # (("busy","..."),) —— 嵌套结构,_handle_event 里 payload[0] 取到的是
        # 内层元组而不是 kind 字符串,导致 "if kind=='connected'" 永远不成立。
        # 所以这里显式 flat 化:若 payload 只有一个元素,则 put 那个元素本身。
        if len(payload) == 1 and isinstance(payload[0], tuple):
            self.events.put(payload[0])
        else:
            self.events.put(payload)


class App(tk.Tk):
    def __init__(self, cal, selftest=False):
        super().__init__()
        self.title("ROH-LiteS001 真机控制台")
        self.geometry(f"{min(1240, self.winfo_screenwidth() - 60)}x{min(880, self.winfo_screenheight() - 90)}")
        self.minsize(1080, 720)
        self.selftest = selftest
        self._closing = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        icon = Path(__file__).with_name("app.ico")
        if sys.platform == "win32" and icon.is_file():
            self.iconbitmap(str(icon))

        self.commands = queue.Queue()
        self.events = queue.Queue()
        # 状态量,worker 上报后由主线程更新
        self.motor_pos = [0] * N_MOTORS
        self.motor_cur = [0] * N_MOTORS
        self.device = None
        self.sim_mode_var = tk.StringVar(value="off")
        self.follow_speed_var = tk.StringVar(value=str(FOLLOW_SPEED))
        self.sim_status_label = None
        self.sim_mode_combo = None
        self.sim_sliders = []
        self.sim_slider_vars = []
        self.sim_value_vars = []
        self._slider_guard = False      # 见 _set_sliders:防回显触发控制
        self.vision_path_var = tk.StringVar(value=str(Path(__file__).with_name("1.mp4")))
        self.vision_source_var = tk.StringVar(value="摄像头")
        self.camera_var = tk.StringVar()
        self.camera_devices = {}
        self._camera_busy = False
        self.vision_mirror_var = tk.BooleanVar(value=True)
        self.vision_hand_var = tk.StringVar(value="自动")
        self._preview_source = None
        self._preview_seq = 0
        self._preview_image = None
        self._vision_error = False
        self._vision_ui_state = None
        self.vision_loop_var = tk.BooleanVar(value=True)
        self.vision_status_label = None
        self.task_contact_var = tk.StringVar(value=str(DEFAULT_CONTACT_MA))
        self.task_close_var = tk.StringVar(value=f"{DEFAULT_CLOSE_NORM:.2f}")
        self.task_status_label = None
        self.task_result = None
        self.task_start_btn = None
        self.task_release_btn = None
        self.task_stable_btn = None
        self.task_failed_btn = None
        self._task_state = "idle"
        self._task_outcome = None

        # 日志文件:venv 会拉子进程,子进程 stdout 不回显到终端,
        # 界面日志又可能被 tkinter 吞错 —— 文件是唯一可靠记录。
        self.log_path = Path(__file__).with_name("gui.log")
        try:
            self.log_path.write_text("", encoding="utf-8")   # 每次启动清空
        except Exception:                                   # noqa: BLE001
            pass

        self.worker = HandWorker(cal, self.commands, self.events)
        self.worker.start()

        self._build_ui()
        if not selftest:
            self._refresh_cameras()

        self._ensure_poll()
        if selftest:
            self.after(1000, self._on_close)

    # ---------- UI ----------
    def _build_ui(self):
        pad = {"padx": 6, "pady": 3}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)
        ttk.Button(top, text="连接", command=lambda: self.worker.put("connect"),
                   width=8).pack(side="left", padx=2)
        ttk.Button(top, text="断开", command=lambda: self.worker.put(
            "disconnect"), width=8).pack(side="left", padx=2)
        self.status_label = ttk.Label(top, text="未连接", foreground="red")
        self.status_label.pack(side="left", padx=8)
        self.info_label = ttk.Label(top, text="", foreground="#555")
        self.info_label.pack(side="left", padx=8)

        self._build_sim_panel(pad)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, **pad)

        # 左侧: 安全
        left = ttk.LabelFrame(body, text="安全")
        left.pack(side="left", fill="y", padx=4, pady=4)
        ttk.Button(left, text="张开（安全姿态）",
                   command=lambda: self.worker.put("open"),
                   width=18).pack(padx=12, pady=8, fill="x")
        ttk.Button(left, text="急停（停在当前位置）",
                   command=lambda: self.worker.put("stop"),
                   width=18).pack(padx=12, pady=8, fill="x")
        ttk.Button(left, text="退出并回张开",
                   command=self._on_close,
                   width=18).pack(padx=12, pady=8, fill="x")

        # 中间: 以物体接触和结果记录为目标的任务流程
        mid = ttk.LabelFrame(body, text="任务：自适应抓握")
        mid.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        ttk.Label(
            mid,
            text=("把物体放在手掌前并保持手部固定。四指会同时缓慢闭合；接触的手指\n"
                  "会单独停住，其余继续，然后再尝试拇指对掌。任务记录接触，不判断物体是否夹稳。"),
            justify="left", wraplength=440).pack(anchor="w", padx=10, pady=(10, 8))

        settings = ttk.Frame(mid)
        settings.pack(fill="x", padx=10, pady=4)
        ttk.Label(settings, text="接触阈值").pack(side="left")
        ttk.Spinbox(settings, from_=200, to=600, increment=25, width=6,
                    textvariable=self.task_contact_var).pack(side="left", padx=(4, 12))
        ttk.Label(settings, text="四指闭合上限").pack(side="left")
        ttk.Spinbox(settings, from_=0.25, to=0.90, increment=0.05, width=6,
                    textvariable=self.task_close_var, format="%.2f").pack(side="left", padx=4)
        ttk.Label(settings, text="归一化").pack(side="left", padx=(0, 8))

        actions = ttk.Frame(mid)
        actions.pack(fill="x", padx=10, pady=6)
        self.task_start_btn = ttk.Button(
            actions, text="开始抓取并保持", command=self._start_grasp_task)
        self.task_start_btn.pack(side="left", padx=(0, 5))
        self.task_stable_btn = ttk.Button(
            actions, text="标记稳定", command=lambda: self._mark_task_outcome("稳定"))
        self.task_stable_btn.pack(side="left", padx=3)
        self.task_failed_btn = ttk.Button(
            actions, text="标记失败 / 滑落", command=lambda: self._mark_task_outcome("失败/滑落"))
        self.task_failed_btn.pack(side="left", padx=3)
        self.task_release_btn = ttk.Button(
            actions, text="松开并保存记录", command=self._release_grasp_task)
        self.task_release_btn.pack(side="left", padx=(8, 0))

        self.task_status_label = ttk.Label(
            mid, text="未连接。先连接真机，再张开手并放好物体。", foreground="#555")
        self.task_status_label.pack(anchor="w", padx=10, pady=(4, 3))
        self.task_result = tk.Text(mid, height=12, wrap="word")
        self.task_result.pack(fill="both", expand=True, padx=10, pady=(3, 10))
        self.task_result.insert(
            "1.0", "单次结果会显示各指接触位置、电流和人工稳定性判定。\n"
                   "任务记录追加保存到 real/task_logs/grasp_trials.jsonl。")
        self.task_result.configure(state="disabled")
        self._refresh_task_controls()

        # 右侧: 实时预览 + 电机状态
        right = ttk.Frame(body)
        right.pack(side="left", fill="y", padx=4)

        video = ttk.LabelFrame(right, text="实时视频 / 手部关键点")
        video.pack(fill="x", padx=4, pady=4)
        self.video_canvas = tk.Canvas(video, width=360, height=270,
                                      bg="#17212b", highlightthickness=0)
        self.video_canvas.pack(fill="both", expand=True)
        self._video_item = self.video_canvas.create_image(180, 135, anchor="center")
        self._video_hint = self.video_canvas.create_text(
            180, 135, text="选择摄像头并点击启动", fill="#dde6ed", font=("Microsoft YaHei UI", 11))

        motors = ttk.LabelFrame(right, text="电机状态")
        motors.pack(fill="both", expand=True, padx=4, pady=4)
        self.pos_labels = []
        self.cur_labels = []
        for i in range(N_MOTORS):
            row = ttk.Frame(motors)
            row.pack(fill="x", padx=4, pady=1)
            ttk.Label(row, width=7, text=MOTOR_CN[i]).pack(side="left")
            pl = ttk.Label(row, width=9, text="--", anchor="e")
            pl.pack(side="left")
            ttk.Label(row, text="pos").pack(side="left", padx=(6, 2))
            cl = ttk.Label(row, width=7, text="--", anchor="e")
            cl.pack(side="left")
            ttk.Label(row, text="mA").pack(side="left", padx=(6, 2))
            self.pos_labels.append(pl)
            self.cur_labels.append(cl)

        # 底部: 日志
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="both", expand=True, **pad)
        ttk.Label(bottom, text="日志").pack(anchor="w")
        self.log_text = tk.Text(bottom, height=7)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

        self.log_text.tag_configure("warn", foreground="#b00")
        self.log_text.tag_configure("ok", foreground="#070")

    def _build_sim_panel(self, pad):
        """交互式仿真控制区；MuJoCo 画面本身由 passive viewer 独立显示。"""
        box = ttk.LabelFrame(self, text="MuJoCo 仿真同步（仿真窗口独立显示）")
        box.pack(fill="x", **pad)

        controls = ttk.Frame(box)
        controls.pack(fill="x", padx=6, pady=(4, 0))
        ttk.Button(controls, text="打开仿真窗口", width=16,
                   command=lambda: self.worker.put(("sim_start",))).pack(
                       side="left", padx=2)
        ttk.Button(controls, text="关闭仿真窗口", width=16,
                   command=lambda: self.worker.put(("sim_stop",))).pack(
                       side="left", padx=2)
        ttk.Label(controls, text="模式").pack(side="left", padx=(14, 3))
        self.sim_mode_combo = ttk.Combobox(
            controls, textvariable=self.sim_mode_var, state="readonly",
            values=("off", "follow", "reverse"), width=10)
        self.sim_mode_combo.pack(side="left")
        self.sim_mode_combo.bind("<<ComboboxSelected>>", self._sim_mode_changed)
        self.sim_status_label = ttk.Label(controls, text="未启动", foreground="#555")
        self.sim_status_label.pack(side="left", padx=10)
        ttk.Label(controls, text="off=只仿真  follow=仿真→真机  reverse=真机→仿真",
                  foreground="#555").pack(side="left", padx=4)

        speed_row = ttk.Frame(box)
        speed_row.pack(fill="x", padx=6, pady=3)
        ttk.Label(speed_row, text="真机跟随速度（1–255）").pack(side="left")
        ttk.Spinbox(speed_row, from_=1, to=255, width=5,
                    textvariable=self.follow_speed_var).pack(side="left", padx=4)
        ttk.Button(speed_row, text="应用速度",
                   command=self._apply_follow_speed).pack(side="left")

        sliders = ttk.Frame(box)
        sliders.pack(fill="x", padx=6, pady=2)
        for i, name in enumerate(DOF_CN):
            col = ttk.Frame(sliders)
            col.grid(row=0, column=i, sticky="ew", padx=2)
            sliders.columnconfigure(i, weight=1)
            var = tk.DoubleVar(value=0.0)
            self.sim_slider_vars.append(var)
            shown = tk.StringVar(value="0.00")
            self.sim_value_vars.append(shown)
            ttk.Label(col, text=name, anchor="center").pack(fill="x")
            scale = ttk.Scale(
                col, from_=0.0, to=1.0, orient="horizontal", variable=var,
                command=lambda value, j=i: self._sim_slider_changed(j, value))
            scale.pack(fill="x")
            self.sim_sliders.append(scale)
            # 显示用单独的 StringVar:DoubleVar 直接绑标签会打印全精度浮点
            # (0.4523809523809524),width=5 的格子里只能看到一截。
            ttk.Label(col, textvariable=shown, width=5,
                      anchor="center").pack()

        self._build_vision_row(box)

    def _build_vision_row(self, box):
        row = ttk.Frame(box)
        row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text="输入源:").pack(side="left")
        self.vision_source_combo = ttk.Combobox(
            row, textvariable=self.vision_source_var, values=("摄像头", "视频文件"),
            state="readonly", width=9)
        self.vision_source_combo.pack(side="left", padx=4)
        self.vision_source_combo.bind("<<ComboboxSelected>>", self._source_changed)
        self.camera_row = ttk.Frame(row)
        self.camera_row.pack(side="left")
        self.camera_combo = ttk.Combobox(self.camera_row, textvariable=self.camera_var,
                                         state="readonly", width=28)
        self.camera_combo.pack(side="left")
        self.camera_refresh_btn = ttk.Button(self.camera_row, text="刷新", width=6,
                                              command=self._refresh_cameras)
        self.camera_refresh_btn.pack(side="left", padx=3)
        self.file_row = ttk.Frame(row)
        self.video_path_entry = ttk.Entry(self.file_row, textvariable=self.vision_path_var, width=36)
        self.video_path_entry.pack(side="left")
        self.video_pick_btn = ttk.Button(self.file_row, text="选择…", width=6, command=self._pick_video)
        self.video_pick_btn.pack(side="left", padx=3)
        self.video_loop_check = ttk.Checkbutton(self.file_row, text="循环", variable=self.vision_loop_var)
        self.video_loop_check.pack(side="left")
        self.vision_start_btn = ttk.Button(row, text="启动", width=7, command=self._start_vision)
        self.vision_start_btn.pack(side="left", padx=(8, 2))
        self.vision_stop_btn = ttk.Button(row, text="停止", width=7,
                                          command=lambda: self.worker.put(("vision_stop",)))
        self.vision_stop_btn.pack(side="left", padx=2)
        self.vision_stop_btn.state(["disabled"])
        ttk.Label(row, text="控制手:").pack(side="left", padx=(8, 0))
        self.vision_hand_combo = ttk.Combobox(row, textvariable=self.vision_hand_var,
                                             values=("自动", "左手", "右手"), state="readonly", width=6)
        self.vision_hand_combo.pack(side="left", padx=3)
        self.vision_mirror_check = ttk.Checkbutton(row, text="镜像预览", variable=self.vision_mirror_var)
        self.vision_mirror_check.pack(side="left", padx=4)
        status = ttk.Frame(box)
        status.pack(fill="x", padx=6, pady=(0, 5))
        self.vision_status_label = ttk.Label(status, text="未启动", foreground="#555")
        self.vision_status_label.pack(side="left", padx=(0, 12))
        ttk.Button(status, text="启用真机跟随", command=lambda: self.worker.put(("sim_mode", "follow"))).pack(side="right", padx=3)
        ttk.Label(status, text="先预览识别，再打开仿真、连接真机并启用跟随", foreground="#555").pack(side="right", padx=8)
        if VisionSource is None:
            self.vision_status_label.config(text=f"不可用: {VISION_IMPORT_ERROR}", foreground="#b00")

    def _refresh_task_controls(self):
        if self.task_start_btn is None:
            return
        connected = bool(self.worker.connected)
        idle = self._task_state == "idle"
        can_start = (connected and idle and self.sim_mode_var.get() == "off"
                     and not self._closing)
        self.task_start_btn.state(["!disabled" if can_start else "disabled"])
        holding = self._task_state == "holding"
        for button in (self.task_stable_btn, self.task_failed_btn,
                       self.task_release_btn):
            button.state(["!disabled" if holding else "disabled"])

    def _start_grasp_task(self):
        try:
            contact_ma = int(self.task_contact_var.get())
            close_norm = float(self.task_close_var.get())
            if not 200 <= contact_ma <= 600:
                raise ValueError
            if not 0.25 <= close_norm <= 0.9:
                raise ValueError
        except ValueError:
            self._log_append("阈值需为 200–600 mA，闭合上限需为 0.25–0.90", "warn")
            return
        if not self.worker.connected:
            self._log_append("请先连接真机", "warn")
            return
        self._task_state = "running"
        self._task_outcome = None
        self._set_task_result("任务开始：四指同时闭合，接触后分别停住，随后拇指对掌。")
        self.task_status_label.config(text="抓取进行中… 急停会停在当前位置。", foreground="#b60")
        self._refresh_task_controls()
        self.worker.put(("task_grasp", contact_ma, close_norm))

    def _mark_task_outcome(self, outcome):
        if self._task_state != "holding":
            return
        self._task_outcome = outcome
        self.task_status_label.config(
            text=f"正在保持 · 人工判定：{outcome}",
            foreground="green" if outcome == "稳定" else "#b00")
        self._log_append(f"本次抓取人工判定：{outcome}")

    def _release_grasp_task(self):
        if self._task_state != "holding":
            return
        self._task_state = "releasing"
        self.task_status_label.config(text="正在松开并记录…", foreground="#555")
        self._refresh_task_controls()
        self.worker.put(("task_release",))

    def _set_task_result(self, text):
        self.task_result.configure(state="normal")
        self.task_result.delete("1.0", "end")
        self.task_result.insert("1.0", text)
        self.task_result.configure(state="disabled")

    @staticmethod
    def _format_grasp_report(report, outcome, saved=False):
        lines = [
            f"保持时间：{report.get('hold_seconds', 0.0):.1f} 秒",
            f"人工判定：{outcome or '未标记'}",
            f"接触阈值：{report.get('contact_ma', '?')} mA；"
            f"四指闭合上限：{report.get('close_norm', '?')}",
            "",
            "部位        深度    接触电流",
        ]
        contacts = report.get("contacts", {})
        norms = report.get("norms", [0.0] * N_MOTORS)
        for motor in (1, 2, 3, 4, 5, 0):
            current = contacts.get(motor, contacts.get(str(motor)))
            current_text = f"{current} mA" if current is not None else "未触发"
            lines.append(f"{MOTOR_CN[motor]:8s} {norms[motor]:.2f}    {current_text}")
        held_currents = report.get("currents_at_hold_ma")
        if held_currents:
            lines.extend(("", "保持时各路电流 (mA)：" + " / ".join(map(str, held_currents))))
        if saved:
            lines.extend(("", "记录已追加到 task_logs/grasp_trials.jsonl。"))
        else:
            lines.extend(("", "松开后保存本次记录。"))
        return "\n".join(lines)

    def _save_grasp_report(self, report):
        record = dict(report)
        record["task"] = "adaptive_grasp"
        record["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")
        record["outcome"] = self._task_outcome or "未标记"
        path = Path(__file__).with_name("task_logs") / "grasp_trials.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def _source_changed(self, _event=None):
        self.camera_row.pack_forget()
        self.file_row.pack_forget()
        shown = self.camera_row if self.vision_source_var.get() == "摄像头" else self.file_row
        shown.pack(side="left", before=self.vision_start_btn)

    def _refresh_cameras(self):
        if self._camera_busy:
            return
        self._camera_busy = True
        self.camera_refresh_btn.state(["disabled"])
        def discover():
            try:
                from camera_capture import list_cameras
                devices, error = list_cameras(), None
            except Exception as exc:
                devices, error = [], str(exc)
            self.events.put(("cameras", devices, error))
        threading.Thread(target=discover, name="CameraDiscovery", daemon=True).start()

    def _pick_video(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="选择人手控制视频",
                    filetypes=[("视频", "*.mp4 *.avi *.mov *.mkv"), ("全部", "*.*")])
        if path:
            self.vision_path_var.set(path)

    def _start_vision(self):
        if self.vision_source_var.get() == "摄像头":
            source = self.camera_devices.get(self.camera_var.get())
            if source is None:
                self._log_append("✗ 请刷新并选择一个可用摄像头", "warn")
                return
        else:
            source = self.vision_path_var.get().strip()
            if not Path(source).is_file():
                self._log_append("✗ 请选择存在的视频文件", "warn")
                return
        self._vision_error = False
        self._set_vision_controls(True)
        self.worker.put(("vision_start", source, self.vision_loop_var.get(), False,
                         self.vision_mirror_var.get(),
                         {"自动": "any", "左手": "Left", "右手": "Right"}[self.vision_hand_var.get()]))

    def _set_vision_controls(self, running):
        if self._closing:
            return
        for widget in (self.vision_source_combo, self.camera_combo, self.vision_hand_combo):
            widget.configure(state="disabled" if running else "readonly")
        for widget in (self.camera_refresh_btn, self.video_path_entry, self.video_pick_btn,
                       self.video_loop_check, self.vision_mirror_check, self.vision_start_btn):
            widget.state(["disabled" if running else "!disabled"])
        self.vision_stop_btn.state(["!disabled" if running else "disabled"])
        self._set_sliders_enabled(not running and self.sim_mode_var.get() != "reverse")

    def _refresh_preview(self):
        vision = self.worker.vision
        if vision is not self._preview_source:
            self._preview_source, self._preview_seq = vision, 0
        if vision is None or not vision.alive():
            self.video_canvas.itemconfigure(self._video_hint, state="normal", text="视频已停止" if self._preview_image else "选择摄像头并点击启动")
            self.video_canvas.itemconfigure(self._video_item, image="")
            return
        packet = vision.preview_frame(self._preview_seq)
        if packet is None:
            return
        from PIL import Image, ImageTk
        self._preview_seq, rgb = packet
        picture = Image.fromarray(rgb)
        picture.thumbnail((self.video_canvas.winfo_width(), self.video_canvas.winfo_height()))
        self._preview_image = ImageTk.PhotoImage(picture, master=self)
        self.video_canvas.coords(self._video_item, self.video_canvas.winfo_width() / 2,
                                 self.video_canvas.winfo_height() / 2)
        self.video_canvas.itemconfigure(self._video_item, image=self._preview_image)
        self.video_canvas.itemconfigure(self._video_hint, state="hidden")

    def _sim_mode_changed(self, _event=None):
        self._refresh_task_controls()
        self.worker.put(("sim_mode", self.sim_mode_var.get()))

    def _apply_follow_speed(self):
        try:
            speed = int(self.follow_speed_var.get())
            if not 1 <= speed <= 255:
                raise ValueError
        except ValueError:
            self._log_append("跟随速度请输入 1–255 的整数", "warn")
            return
        self.worker.put(("follow_speed", speed))

    def _set_sliders_enabled(self, on: bool):
        state = "!disabled" if on else "disabled"
        for s in self.sim_sliders:
            s.state([state])

    def _set_sliders(self, norms_sim):
        """把滑杆摆到给定姿态(归一化,仿真序)。reverse 回显走这里。

        必须带 `_slider_guard`:给 DoubleVar 赋值会触发 ttk.Scale 的 command,
        也就是回调进 _sim_slider_changed 再去 set_one() —— 真机每报一帧就把
        仿真目标覆盖一次,顺带清掉正在播的手势脚本。只是显示,不该产生控制。
        """
        self._slider_guard = True
        try:
            for i, v in enumerate(norms_sim):
                self.sim_slider_vars[i].set(v)
                self.sim_value_vars[i].set(f"{v:.2f}")
        finally:
            self._slider_guard = False

    def _sim_slider_changed(self, i, value):
        # SimSync.set_one 自带锁,主线程直接调是安全的;走命令队列反而会被
        # worker 的 CAN 节拍拖慢,拖动手感变差。
        self.sim_value_vars[i].set(f"{float(value):.2f}")
        if self._slider_guard:
            return
        if self.worker.sim is not None:
            self.worker.sim.set_one(i, float(value))

    def _log_append(self, text, tag=None):
        """写界面日志。tag 可选:'ok' 绿色、'warn' 红色,用于区分事件性质。

        只在主线程被调用(通过 _handle_event)。worker 线程的日志已改为发事件,
        不再直接碰 tkinter 控件。
        """
        try:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", text + "\n", tag)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        except Exception:                           # noqa: BLE001
            pass
        try:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
        except Exception:                           # noqa: BLE001
            pass

    # ---------- 事件轮询 ----------
    def _poll(self):
        """每 33ms 检查 worker 发来的事件,刷新界面。

        关键:这个回调必须持有**强引用**,否则 tkinter 的 after 回调会被垃圾
        回收,执行时报 "invalid command name ..._poll",after 链断裂,界面
        从此不再刷新 —— 事件照常到队列,但界面永远停在旧状态(表现为"点了
        连接没反应")。所以下面用 self._poll_ref 常驻引用。
        """
        try:
            while True:
                payload = self.events.get_nowait()
                self._handle_event(payload)
        except queue.Empty:
            pass
        except Exception as exc:
            # _handle_event 的异常不能让 after 链断掉,也不能被 tkinter 吞掉。
            # 必须落盘,否则"点了没反应"永远查不到。
            import traceback as _tb
            with open(self.log_path, "a", encoding="utf-8") as _f:
                _f.write(f"  !!! _handle_event 异常: {type(exc).__name__}: {exc}\n")
                _f.write("      " + "\n      ".join(
                    _tb.format_exc().splitlines()) + "\n")
        try:
            self._refresh_preview()
        except Exception as exc:
            self._log_append(f"预览显示失败: {exc}", "warn")
        self._after_id = self.after(33, self._poll_ref)

    def _ensure_poll(self):
        """首次启动后持有强引用并调度。"""
        # 用 bound method 自身作引用,不被 GC
        self._poll_ref = self._poll
        self._after_id = self.after(60, self._poll_ref)

    def _handle_event(self, payload):
        kind = payload[0]
        if kind == "cameras":
            _, devices, error = payload
            self._camera_busy = False
            self.camera_devices = {f"{index} · {name}": index for index, name in devices}
            self.camera_combo.configure(values=tuple(self.camera_devices))
            if self.camera_var.get() not in self.camera_devices:
                self.camera_var.set(next(iter(self.camera_devices), ""))
            if not self._closing and (self.worker.vision is None or not self.worker.vision.is_alive()):
                self.camera_refresh_btn.state(["!disabled"])
            if error or not devices:
                self._log_append("✗ 摄像头枚举: " + (error or "未找到可用摄像头"), "warn")
        elif kind == "connected":
            _, info, cali = payload
            self.device = info
            self.status_label.config(text="已连接", foreground="green")
            self.info_label.config(
                text=f"固件 {info.get('firmware','?')}  "
                     f"UID {info.get('uid','?')}")
            self._task_state = "idle"
            self.task_status_label.config(
                text="已连接。先张开手，把物体放到手掌前，再开始抓取。",
                foreground="green")
            self._refresh_task_controls()
            self._log_append("✓ 已连接，可以执行自适应抓握任务", "ok")
        elif kind == "disconnected":
            self.device = None
            self.status_label.config(text="未连接", foreground="red")
            self.info_label.config(text="")
            self._task_state = "idle"
            self._task_outcome = None
            self.task_status_label.config(text="未连接。", foreground="#555")
            self._refresh_task_controls()
            self._log_append(payload[1])
        elif kind == "task_started":
            self._task_state = "running"
            self.task_status_label.config(text="抓取进行中…", foreground="#b60")
            self._refresh_task_controls()
            self._log_append("▶ 自适应抓取开始")
        elif kind == "task_progress":
            _, phase, total, label, state, _norms, contacts = payload
            contacted = [MOTOR_CN[i] for i in sorted(contacts)]
            suffix = f"；接触：{'、'.join(contacted)}" if contacted else ""
            self.task_status_label.config(
                text=f"步骤 {phase}/{total}：{label} · {state}{suffix}",
                foreground="#b60")
            self._log_append(f"步骤 {phase}/{total} {label}：{state}{suffix}")
        elif kind == "task_holding":
            _, report = payload
            self._task_state = "holding"
            self._task_outcome = None
            self.task_status_label.config(
                text="抓取流程已完成，正在保持。请观察物体并人工标记结果。",
                foreground="green")
            self._set_task_result(self._format_grasp_report(report, None))
            self._refresh_task_controls()
            self._log_append("✓ 抓取姿态已建立，进入保持阶段", "ok")
        elif kind == "task_releasing":
            self._task_state = "releasing"
            self.task_status_label.config(text="正在松开并保存记录…", foreground="#555")
            self._refresh_task_controls()
        elif kind == "task_released":
            _, report = payload
            self._task_state = "idle"
            try:
                path = self._save_grasp_report(report)
                self._set_task_result(self._format_grasp_report(
                    report, self._task_outcome, saved=True))
                self.task_status_label.config(
                    text=f"本次记录已保存 · 人工判定：{self._task_outcome or '未标记'}",
                    foreground="green")
                self._log_append(f"✓ 抓取记录已追加到 {path}", "ok")
            except Exception as exc:
                self._set_task_result(self._format_grasp_report(
                    report, self._task_outcome, saved=False))
                self.task_status_label.config(text="松开完成，但记录文件写入失败。", foreground="#b00")
                self._log_append(f"✗ 保存抓取记录失败: {exc}", "warn")
            self._task_outcome = None
            self._refresh_task_controls()
        elif kind == "task_aborted":
            self._task_state = "idle"
            self._task_outcome = None
            self.task_status_label.config(
                text=f"任务已中止：{payload[1]}。手部停在当前位置，请检查后再张开。",
                foreground="#b00")
            self._set_task_result("任务中止，未保存为完整试验记录。")
            self._refresh_task_controls()
            self._log_append("⚠ " + payload[1], "warn")
        elif kind == "task_error":
            self._task_state = "idle"
            self._task_outcome = None
            self.task_status_label.config(text="任务未完成，请检查提示并确认手部姿态。", foreground="#b00")
            self._set_task_result(payload[1])
            self._refresh_task_controls()
            self._log_append("✗ 抓取任务: " + payload[1], "warn")
        elif kind in ("busy",):
            self._log_append(payload[1])
        elif kind == "error":
            self._log_append("✗ " + payload[1], "warn")
        elif kind == "step":
            _, i, total, text = payload
            self._log_append(f"   {text}")
            self._set_progress(i, total)
        elif kind == "state":
            _, pos, cur = payload
            self.motor_pos = pos
            self.motor_cur = cur
            self._refresh_motors(pos, cur)
        elif kind == "sim":
            self._handle_sim_event(payload[1:])

    def _handle_sim_event(self, payload):
        kind = payload[0]
        if kind == "ready":
            self.sim_status_label.config(text="仿真运行中", foreground="green")
            self._log_append("✓ MuJoCo 仿真窗口已打开", "ok")
        elif kind == "closed":
            self.sim_status_label.config(text="未启动", foreground="#555")
            self.sim_mode_var.set("off")
            self._log_append("MuJoCo 仿真窗口已关闭")
        elif kind == "mode":
            mode = payload[1]
            self.sim_mode_var.set(mode)
            label = {"off": "只仿真", "follow": "仿真→真机",
                     "reverse": "真机→仿真"}.get(mode, mode)
            self.sim_status_label.config(text=label, foreground="green")
            self._refresh_task_controls()
            self._log_append(f"同步模式: {label}", "ok")
        elif kind == "pose":
            # reverse 模式下把真机实际姿态反映到滑杆,避免滑杆停在旧目标上。
            # 收到的已经是**仿真序的归一化量**(worker 那边排好的),这里直接用,
            # 不再自己换算 —— 映射表只应该存在一份。
            self._set_sliders(payload[1])
        elif kind == "stalled":
            # 真机侧「顶住了」的那几路。标红是为了让人**当场看见**是哪一路顶住
            # 了 —— 只靠日志滚动很容易漏掉,而顶住通常意味着机械干涉或抓到了
            # 东西,是需要立刻知道的事。
            idx = set(payload[1])
            for i in range(N_MOTORS):
                self.cur_labels[i].config(
                    foreground="#b00" if i in idx else "")
            if idx:
                self._log_append(
                    "⚠ 顶住并已钉住: "
                    + "、".join(MOTOR_CN[i] for i in sorted(idx))
                    + "（该路停止加力，其余照常跟随；目标退回即自动解除）",
                    "warn")
        elif kind == "log":
            self._log_append(payload[1])
        elif kind == "vision_starting":
            self._set_vision_controls(True)
            self.vision_status_label.config(text="正在打开视频和识别模型…", foreground="#b60")
        elif kind == "vision_ready":
            self._set_vision_controls(True)
            self._log_append(f"✓ 视频已启动: {payload[1]}", "ok")
        elif kind == "vision_stopping":
            self.vision_status_label.config(text="正在释放视频资源…", foreground="#555")
            self.vision_stop_btn.state(["disabled"])
        elif kind == "vision_closed":
            self._set_vision_controls(False)
            self.vision_status_label.config(text="出错，请查看日志" if self._vision_error else "已停止",
                                            foreground="#b00" if self._vision_error else "#555")
            if self._vision_ui_state != "closed":
                self._log_append("视频资源已释放，可以重新启动")
            self._vision_ui_state = "closed"
        elif kind == "vision_error":
            self._vision_error = True
            self.vision_status_label.config(text="出错，请查看日志", foreground="#b00")
            if self.worker.vision is None:
                self._set_vision_controls(False)
            self._log_append("✗ 视频驱动: " + payload[1], "warn")
        elif kind == "vision_state":
            state = payload[1]
            self._vision_ui_state = state["state"]
            labels = {"starting": "启动中", "tracking": "识别中", "missing": "未检测到控制手",
                      "stopping": "正在释放资源", "stopped": "已停止", "error": "出错"}
            label = labels[state["state"]]
            if state["state"] in ("tracking", "missing"):
                if state["frame_age"] > VisionSource.LOST_TIMEOUT:
                    label = "画面中断"
                label += f" · {state['fps']:.1f} fps · 处理 {state['latency_ms']:.0f} ms"
            self.vision_status_label.config(text=label,
                foreground="green" if state["control_ready"] else "#b60")
        elif kind == "warn":
            self._log_append("⚠ " + payload[1], "warn")
        elif kind == "sim_ready":
            self.sim_status_label.config(text="仿真运行中", foreground="green")
        elif kind == "sim_closed":
            self.sim_status_label.config(text="未启动", foreground="#555")
            self.sim_mode_var.set("off")
            self._set_sliders_enabled(True)
            self._refresh_task_controls()
            self._log_append("MuJoCo viewer 已关闭")
        elif kind in ("error", "sim_error"):
            # sim_error 是 SimSync 线程自己抛的(模型加载失败、渲染线程异常)。
            # 漏掉这个分支的话,仿真起不来时界面完全静默,只能去翻 gui.log。
            self._log_append("✗ " + payload[1], "warn")

    def _set_progress(self, i, total):
        # 简单进度条,放在状态区
        pct = int(i / total * 100) if total else 0
        self.info_label.config(text=f"{self.info_label.cget('text')} · {pct}%")

    def _refresh_motors(self, pos, cur):
        for i in range(N_MOTORS):
            self.pos_labels[i].config(text=str(pos[i]))
            self.cur_labels[i].config(text=str(cur[i]))

    def _on_close(self):
        """先异步清理设备/仿真，再关 tkinter，避免窗口先退导致后台线程遗留。"""
        if self._closing:
            return
        self._closing = True
        self._set_controls_enabled(False)
        self.worker.put("__quit__")
        self._shutdown_deadline = time.monotonic() + 8.0
        self.after(50, self._finish_close)

    def _set_controls_enabled(self, enabled):
        state = "!disabled" if enabled else "disabled"
        for child in self.winfo_children():
            self._set_widget_tree_state(child, state)

    @staticmethod
    def _set_widget_tree_state(widget, state):
        try:
            widget.state([state])
        except (AttributeError, tk.TclError):
            pass
        for child in widget.winfo_children():
            App._set_widget_tree_state(child, state)

    def _finish_close(self):
        vision = self.worker.vision
        sim = self.worker.sim
        pending = (self.worker.is_alive() or (vision is not None and vision.is_alive())
                   or (sim is not None and sim.is_alive()))
        if pending:
            if time.monotonic() >= self._shutdown_deadline:
                self._log_append("正在等待摄像头 / 模型释放资源…")
                self._shutdown_deadline = time.monotonic() + 8.0
            self.after(100, self._finish_close)
            return
        self.after_cancel(self._after_id)
        self.destroy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="只验证界面,1 秒后自动关闭,不连手")
    args = ap.parse_args()

    cal = Calibration.load()
    app = App(cal, selftest=args.selftest)
    app.mainloop()
    return 0


if __name__ == "__main__":
    from gui_startup import run_gui
    sys.exit(run_gui(main))
