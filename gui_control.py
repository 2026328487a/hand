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
    [上方] 摄像头预览 / 内嵌三维手模型，连接与安全按钮始终可见
    [中间] 仿真、视觉输入与关节控制
    [下方] 自适应抓握任务 + 电机状态；日志可折叠
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
from gui_startup import configure_window_scaling         # noqa: E402
from gestures import GESTURES, play                     # noqa: E402
from rohand import MOTORS, MOTOR_CN, N_MOTORS, HandError, ROHand  # noqa: E402
from task_mode import (DEFAULT_CLOSE_NORM, DEFAULT_CONTACT_MA,
                       perform_adaptive_grasp)            # noqa: E402

# 仿真模块依赖上级目录的 MuJoCo 模型。GUI 本身仍然应该能在没有仿真工作区
# 时启动，所以这里故意延迟失败；仿真面板会显示具体原因，而不影响真机面板。
from control_core import (DOF_CN, MotionController, SafeSender,
                          align_control_to_hand)
from joint_mapping import reorder_real_to_sim
try:                                                       # noqa: E402
    from sim_sync import SimSync
    SIM_IMPORT_ERROR = None
except Exception as _sim_exc:
    SimSync = None
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
CONTROL_SOURCES = {"手动滑杆": "manual", "视觉输入": "vision", "仿真调试": "follow"}
CONTROL_LABELS = {"off": "已暂停", "manual": "手动滑杆控制中",
                  "vision": "视觉控制中", "follow": "仿真驱动真机"}


def step_progress_to_str(i: int, total: int, norms, pose, contact) -> str:
    n = " ".join(f"{v:.2f}" for v in norms)
    p = " ".join(str(v) for v in pose)
    tag = f"  [接触停住 {contact}]" if contact else ""
    return f"步骤{i}/{total}  norm[{n}]  →  pos [{p}]{tag}"


class HandWorker(threading.Thread):
    """独占手的后台线程。主线程通过 put(cmd) 发命令,通过 events 收反馈。

    连接、任务和模式命令串行执行；关节输入合并为每路最新目标。
    急停撤销此前排队的请求，连续控制和抓握任务独占控制权。
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
        self.controller = MotionController()
        self.control_source = "manual"
        self.preview_mode = "target"
        self._input_lock = threading.Lock()
        self._input_epoch = 0
        self._generation = 0
        self._manual_pending = {}
        self._control_t = time.monotonic()
        self._vision_seq = -1
        self._vision_after = 0.0
        self._feedback_pose = None
        self._task_running = False
        self.control_mode = "off"
        self.sender = None
        self.follow_speed = FOLLOW_SPEED
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
        with self._input_lock:
            if cmd in ("stop", "disconnect", "__quit__"):
                self._generation += 1
                self._input_epoch += 1
                self._manual_pending.clear()
                self._task_cancel.set()
                self.commands.put(cmd)
                return
            if isinstance(cmd, tuple) and cmd and cmd[0] == "manual_target":
                _, epoch, index, value = cmd
                if (epoch == self._input_epoch and not self._task_cancel.is_set()
                        and not self._task_running and self._active_grasp is None):
                    self._manual_pending[index] = float(value)
                return
            if (self._task_running or self._active_grasp is not None) and (
                    isinstance(cmd, tuple) and cmd and cmd[0] in ("control_mode", "select_source")):
                self._send_sim("warn", "抓握任务占用控制权，请先松开或急停")
                self._mode_event()
                return
            self.commands.put(("request", self._generation, cmd))

    def _dispatch(self, cmd):
        if isinstance(cmd, tuple) and cmd and cmd[0] == "request":
            _, generation, cmd = cmd
            if generation != self._generation:
                return
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

    def run(self):
        period = 1.0 / FOLLOW_HZ
        next_tick = time.monotonic() + period
        while True:
            try:
                cmd = self.commands.get(timeout=max(0.0, next_tick - time.monotonic()))
            except queue.Empty:
                cmd = None
            try:
                if cmd == "__quit__":
                    self._do_disconnect()
                    self._stop_vision()
                    self._stop_sim()
                    return
                if cmd is not None:
                    self._dispatch(cmd)
                if time.monotonic() >= next_tick:
                    self._control_tick()
                    self._maybe_report_state()
                    next_tick += period
                    if next_tick <= time.monotonic():
                        next_tick = time.monotonic()
            except Exception as exc:
                self._pause_control(f"控制异常: {exc}")

    # ----- Continuous control; simulation is an optional consumer/source -----
    def _send_sim(self, *payload):
        self._send(("sim",) + payload)

    def _mode_event(self):
        self._send_sim("mode", self.control_mode, self._input_epoch, self.control_source)

    def _invalidate_input(self):
        with self._input_lock:
            self._input_epoch += 1
            self._manual_pending.clear()

    def _pause_control(self, reason=None):
        was_active = self.control_mode != "off"
        self.control_mode = "off"
        self.sender = None
        self._invalidate_input()
        self.controller.set_state(self.controller.action())
        self._last_stalled = ()
        self._send_sim("stalled", [])
        ok = True
        if was_active and self.hand is not None:
            try:
                self.hand.stop()
            except Exception as exc:
                ok = False
                self._send_sim("warn", f"停止命令失败: {exc}")
        self._mode_event()
        if reason:
            self._send_sim("warn", reason)
        return ok

    def _do_sim_command(self, cmd):
        op = cmd[0]
        if op == "sim_start":
            self._start_sim()
        elif op == "sim_stop":
            self._stop_sim()
        elif op == "control_mode":
            self._set_control_mode(cmd[1])
        elif op == "select_source" and cmd[1] in CONTROL_SOURCES.values():
            if self._active_grasp is not None or self._task_running:
                self._send_sim("warn", "抓握任务占用控制权，请先松开或急停")
                self._mode_event()
                return
            self._pause_control()
            self.control_source = cmd[1]
            self._vision_seq = -1
            self._vision_after = time.perf_counter()
            if self.sim is not None:
                self.sim.push_external(None)
                self.sim.set_state(self.controller.action())
            self._mode_event()
        elif op == "preview_mode" and cmd[1] in ("target", "feedback"):
            self.preview_mode = cmd[1]
        elif op == "follow_speed":
            self.follow_speed = max(1, min(255, int(cmd[1])))
            if self.sender is not None:
                self.sender.speed = self.follow_speed
            self._send_sim("log", f"真机速度: {self.follow_speed}/255")
        elif op == "vision_start":
            self._start_vision(*cmd[1:])
        elif op == "vision_stop":
            self._stop_vision()
        elif op == "vision_event":
            self._on_vision_event(*cmd[1:])
        elif op == "sim_event":
            self._handle_sim_lifecycle(*cmd[1:])

    def _set_control_mode(self, mode):
        if mode not in ("off", "manual", "vision", "follow"):
            return
        if self._task_running or self._active_grasp is not None:
            self._send_sim("warn", "抓握任务占用控制权，请先松开或急停")
            self._mode_event()
            return
        if not self._pause_control():
            return
        if mode == "off":
            return
        if self._task_cancel.is_set():
            return
        if not self._connected or self.hand is None:
            self._send_sim("warn", "请先连接真机")
            return
        if mode == "follow" and (self.sim is None or not self.sim.wait_ready(0)):
            self._send_sim("warn", "仿真调试需要先启动三维仿真并等待就绪")
            return
        if mode == "vision" and (self.vision is None or not self.vision.snapshot()["control_ready"]):
            self._send_sim("warn", "请先启动识别，并让控制手进入画面")
            return
        try:
            source = self.sim if mode == "follow" else self.controller
            source.push_external(None)
            align_control_to_hand(source, self.cal, self.hand)
            self.controller.set_state(source.action())
            self._feedback_pose = list(source.action())
            self._vision_after = time.perf_counter()
            self._vision_seq = -1
            self._control_t = time.monotonic()
            if self._task_cancel.is_set():
                return
            self.sender = SafeSender(self.cal, source, speed=self.follow_speed,
                                     on_warn=lambda msg: self._send_sim("warn", msg))
            self.control_source = mode
            self.control_mode = mode
            self._send_sim("pose", list(source.action()))
            self._mode_event()
        except Exception as exc:
            self._pause_control(f"启用控制失败: {exc}")

    def _start_vision(self, source, loop=True, preview=False, mirror=False, handedness="any"):
        if VisionSource is None:
            self._send_sim("vision_error", f"视觉模块不可用: {VISION_IMPORT_ERROR}")
            return
        if self.vision is not None and self.vision.is_alive():
            self._send_sim("log", "视频源仍在运行或释放资源，请停止后等待再启动")
            return
        if self.control_mode == "vision":
            self._pause_control("切换视频源，视觉控制已暂停")
        try:
            # GUI consumes timestamped snapshots. Inference never writes control targets.
            vision = VisionSource(source, loop=bool(loop), preview=bool(preview),
                                  mirror=mirror, handedness=handedness,
                                  on_event=lambda kind, *args: self.put(("vision_event", vision, kind, args)))
            self.vision = vision
            self._vision_seq = -1
            self._vision_after = time.perf_counter()
            self._send_sim("vision_starting")
            vision.start()
        except Exception as exc:
            self.vision = None
            self._send_sim("vision_error", str(exc))

    def _stop_vision(self):
        if self.control_mode == "vision":
            self._pause_control("视频已停止，视觉控制已暂停")
        if self.vision is not None:
            self.vision.stop()
            self._send_sim("vision_stopping")

    def _on_vision_event(self, vision, kind, args):
        if vision is not self.vision:
            return
        if kind in ("vision_error", "vision_stopping", "vision_closed") and self.control_mode == "vision":
            self._pause_control("视频源结束或异常，视觉控制已暂停")
        if kind != "vision_closed":
            self._send_sim(kind, *args)

    def _publish_vision(self):
        if self.vision is None:
            return
        now = time.monotonic()
        if now - self._pose_t < 0.1:
            return
        self._pose_t = now
        self._send_sim("vision_state", self.vision.snapshot())
        if self.control_source == "vision":
            self._send_sim("pose", list(self.controller.action()))
        if not self.vision.is_alive():
            self._send_sim("vision_closed")
            self.vision = None

    def _start_sim(self):
        if self.sim is not None and self.sim.is_alive():
            self._send_sim("log", "三维仿真仍在运行或退出，请稍后再试")
            return
        if SimSync is None:
            self._send_sim("error", f"仿真不可用: {SIM_IMPORT_ERROR}")
            return
        try:
            sim = SimSync(display="embedded",
                          on_event=lambda kind, *args: self.put(("sim_event", sim, kind, args)))
            sim.set_state(self.controller.action())
            self.sim = sim
            self._send_sim("starting")
            sim.start()
        except Exception as exc:
            self.sim = None
            self._send_sim("error", f"启动仿真失败: {exc}")

    def _stop_sim(self):
        if self.control_mode == "follow":
            self._pause_control("仿真调试已停止")
        if self.sim is not None:
            self._send_sim("stopping")
            self.sim.stop()  # Never join the renderer on the CAN owner thread.
        else:
            self._send_sim("closed")

    def _handle_sim_lifecycle(self, sim, kind, args):
        if sim is not self.sim:
            return
        if kind == "sim_ready":
            if sim.alive():
                self._send_sim("ready")
            return
        self._send_sim(kind, *args)
        if kind == "sim_closed":
            if self.control_mode == "follow":
                self._pause_control("仿真已退出，仿真调试控制已暂停")
            # Keep reference until thread exits; GUI shutdown also waits on it.

    def _control_tick(self):
        now = time.monotonic()
        dt = min(max(now - self._control_t, 0.0), 0.1)
        self._control_t = now
        if self._task_cancel.is_set():
            if self.control_mode != "off":
                self._pause_control()
            return
        if self.sim is not None and not self.sim.is_alive():
            if self.control_mode == "follow":
                self._pause_control("仿真已退出，仿真调试控制已暂停")
            self.sim = None
            self._send_sim("closed")
        with self._input_lock:
            pending, self._manual_pending = self._manual_pending, {}
        if not self._task_running and self._active_grasp is None:
            if self.control_source in ("manual", "follow"):
                source = self.sim if self.control_source == "follow" else self.controller
                if source is not None:
                    for index, value in pending.items():
                        source.set_one(index, value)
            elif self.vision is not None:
                state = self.vision.snapshot()
                if not self.vision.alive() or state["error"]:
                    if self.control_mode == "vision":
                        self._pause_control("视频源结束或异常，视觉控制已暂停")
                    self.controller.set_target(self.controller.action())
                elif not state["control_ready"]:
                    self.controller.set_target(self.controller.action())
                elif (state["sequence"] != self._vision_seq
                      and state["captured_at"] is not None
                      and state["captured_at"] >= self._vision_after):
                    self.controller.set_target(state["norms"])
                    self._vision_seq = state["sequence"]
            self.controller._advance(dt)
        if self.control_mode == "follow" and (self.sim is None or not self.sim.alive()):
            self._pause_control("仿真已停止，仿真调试控制已暂停")
        if self.control_mode != "off" and self.sender is not None and self.hand is not None:
            try:
                self.sender.tick(self.hand, should_cancel=self._task_cancel.is_set)
                self._feedback_pose = reorder_real_to_sim([
                    self.cal.norm_from_pos(i, p) for i, p in enumerate(self.sender.positions)])
                stalled = tuple(self.sender.stalled)
                if stalled != self._last_stalled:
                    self._last_stalled = stalled
                    self._send_sim("stalled", list(stalled))
            except Exception as exc:
                self._pause_control(f"通信或控制异常，已暂停: {exc}")
        self._publish_vision()
        if self.sim is not None and self.sim.alive() and self.control_source != "follow":
            pose = (self._feedback_pose if self.preview_mode == "feedback"
                    and self._connected else self.controller.action())
            if pose is not None:
                self.sim.push_external(pose)

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
        self._pause_control()
        if not self._connected:
            self._task_cancel.clear()
            return
        self._task_cancel.clear()
        try:
            self._open_hand()
        except Exception as exc:
            self._send(("busy", f"断开前回张开失败: {exc}"))
        finally:
            self.hand.close()
            self.hand = None
            self._connected = False
            self._feedback_pose = None
            self._active_grasp = None
            self._grasp_started_t = None
            self._task_cancel.clear()
            self._send(("disconnected", "已断开"))

    def _do_stop(self):
        was_active = self.control_mode != "off"
        stopped = self._pause_control()
        try:
            if self.hand is not None and not was_active:
                self.hand.stop()
            if self._active_grasp is not None:
                self._active_grasp = None
                self._grasp_started_t = None
                self._send(("task_aborted", "保持任务已急停"))
            if stopped:
                self._send(("busy", "已急停（停在当前位置）"))
        except Exception as exc:
            self._send(("error", f"急停失败: {exc}"))
        finally:
            self._task_cancel.clear()

    def _open_hand(self):
        actual = [self.cal.norm_from_pos(i, p) for i, p in enumerate(self.hand.read_pos())]
        play(self.hand, self.cal, GESTURES["open"], prev_norms=actual,
             verbose=False, on_step=self._stop_cb,
             should_cancel=self._task_cancel.is_set)
        align_control_to_hand(self.controller, self.cal, self.hand)
        self._feedback_pose = list(self.controller.action())
        self._send_sim("pose", self._feedback_pose)

    def _do_open(self):
        if not self._connected:
            self._send(("busy", "请先连接真机"))
            return
        if not self._pause_control():
            return
        self._task_running = True
        try:
            self._open_hand()
        except Exception as exc:
            self.hand.stop()
            self._send(("error", f"张开未完成: {exc}"))
        finally:
            self._task_running = False
            if self._active_grasp is not None:
                self._active_grasp = None
                self._grasp_started_t = None
                self._send(("task_aborted", "已手动张开或中止张开，抓握记录未完成"))

    def _do_task_command(self, cmd):
        self._invalidate_input()
        self._task_running = True
        try:
            if cmd[0] == "task_grasp":
                self._do_adaptive_grasp(*cmd[1:])
            elif cmd[0] == "task_release":
                self._do_task_release()
        finally:
            self._task_running = False
            self._mode_event()

    def _do_adaptive_grasp(self, contact_ma, close_norm):
        if not self._connected or self.hand is None:
            self._send(("task_error", "请先连接真机"))
            return
        if self.control_mode != "off":
            self._send(("task_error", "请先暂停连续控制，再开始抓握任务"))
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

        self.controller.set_state(reorder_real_to_sim(result["norms"]))
        self._feedback_pose = list(self.controller.action())
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
            self._open_hand()
        except Exception as exc:
            self.hand.stop()
            self._send(("task_release_error", f"自动松开失败: {exc}"))
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
        """连续控制复用反馈；空闲时以 5 Hz 读取，任务中不额外占用 CAN。"""
        if not self._connected:
            return
        now = time.monotonic()
        if getattr(self, "_last_state_t", 0) and now - self._last_state_t < (0.1 if self.sender is not None else 0.2):
            return
        self._last_state_t = now
        try:
            if self.control_mode != "off" and self.sender is not None:
                # 跟随保护已经读取过反馈；界面回显复用它，不再占一次 CAN。
                pos = list(self.sender.positions)
                cur = list(self.sender.currents)
            else:
                pos = self.hand.read_pos()
                cur = self.hand.read_currents()
            self._feedback_pose = reorder_real_to_sim([
                self.cal.norm_from_pos(i, p) for i, p in enumerate(pos)])
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
        # 也覆盖直接 import / 构造 App 的入口；必须早于 Tk 创建窗口。
        configure_window_scaling()
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
        self.control_mode_var = tk.StringVar(value="off")
        self.control_source_var = tk.StringVar(value="手动滑杆")
        self.preview_mode_var = tk.StringVar(value="目标姿态")
        self._input_epoch = 0
        self._control_pending = False
        self.actual_value_vars = []
        self.follow_speed_var = tk.StringVar(value=str(FOLLOW_SPEED))
        self.sim_status_label = None
        self.control_mode_combo = None
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
        self._sim_preview_source = None
        self._sim_preview_seq = 0
        self._sim_preview_rgb = None
        self._sim_preview_image = None
        self._sim_preview_size = None
        self._sim_drag = None
        self._sim_hint_text = "三维预览可选；手动和视觉控制无需启动仿真"
        self._sim_failed = False
        self.sim_preview_var = tk.BooleanVar(value=True)
        self.log_visible_var = tk.BooleanVar(value=False)
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
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=3, minsize=150)
        self.rowconfigure(3, weight=1, minsize=190)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", **pad)
        ttk.Button(top, text="连接", command=lambda: self.worker.put("connect"),
                   width=8).pack(side="left", padx=2)
        ttk.Button(top, text="断开", command=lambda: self.worker.put(
            "disconnect"), width=8).pack(side="left", padx=2)
        self.status_label = ttk.Label(top, text="未连接", foreground="red")
        self.status_label.pack(side="left", padx=8)
        self.info_label = ttk.Label(top, text="", foreground="#555")
        self.info_label.pack(side="left", padx=8)
        ttk.Button(top, text="退出并回张开", command=self._on_close).pack(side="right", padx=2)
        ttk.Button(top, text="急停（停在当前位置）",
                   command=lambda: self.worker.put("stop")).pack(side="right", padx=4)
        ttk.Button(top, text="张开（安全姿态）",
                   command=lambda: self.worker.put("open")).pack(side="right", padx=4)

        self._build_preview_panels(pad)
        self._build_sim_panel(pad)

        body = ttk.Frame(self)
        body.grid(row=3, column=0, sticky="nsew", **pad)

        # 中间: 以物体接触和结果记录为目标的任务流程
        mid = ttk.LabelFrame(body, text="任务：自适应抓握")
        mid.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        ttk.Label(
            mid,
            text="固定手部并放好物体。四指接触后分别停住，再尝试拇指对掌；是否夹稳需人工判定。",
            justify="left", wraplength=760).pack(anchor="w", padx=8, pady=(3, 2))

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
        self.task_result = tk.Text(mid, height=3, width=40, wrap="word")
        self.task_result.pack(fill="both", expand=True, padx=10, pady=(3, 10))
        self.task_result.insert(
            "1.0", "单次结果会显示各指接触位置、电流和人工稳定性判定。\n"
                   "任务记录追加保存到 real/task_logs/grasp_trials.jsonl。")
        self.task_result.configure(state="disabled")
        self._refresh_task_controls()

        # 任务旁始终显示电机反馈。
        right = ttk.Frame(body)
        right.pack(side="left", fill="y", padx=4)

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
        bottom.grid(row=4, column=0, sticky="ew", **pad)
        bottom.columnconfigure(0, weight=1)
        ttk.Checkbutton(bottom, text="展开日志", variable=self.log_visible_var,
                        command=self._toggle_log).grid(row=0, column=0, sticky="w")
        self.log_text = tk.Text(bottom, height=3)
        self.log_text.grid(row=1, column=0, sticky="ew")
        self.log_text.grid_remove()
        self.log_text.configure(state="disabled")

        self.log_text.tag_configure("warn", foreground="#b00")
        self.log_text.tag_configure("ok", foreground="#070")

    def _toggle_log(self):
        if self.log_visible_var.get():
            self.log_text.grid()
        else:
            self.log_text.grid_remove()

    def _build_preview_panels(self, pad):
        views = ttk.Frame(self)
        views.grid(row=1, column=0, sticky="nsew", **pad)
        views.columnconfigure((0, 1), weight=1, uniform="preview")
        views.rowconfigure(0, weight=1)
        video = ttk.LabelFrame(views, text="摄像头 / 手部关键点")
        video.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        sim = ttk.LabelFrame(views, text="三维手模型")
        sim.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        for panel in (video, sim):
            panel.columnconfigure(0, weight=1)
            panel.rowconfigure(1, weight=1)
        ttk.Label(video, text="实时视频与识别结果", foreground="#555").grid(
            row=0, column=0, sticky="w", padx=6, pady=5)
        tools = ttk.Frame(sim)
        tools.grid(row=0, column=0, sticky="ew", padx=6, pady=2)
        row = ttk.Frame(tools)
        row.pack(fill="x")
        self.sim_start_btn = ttk.Button(row, text="启动三维", width=9,
                                        command=lambda: self.worker.put(("sim_start",)))
        self.sim_start_btn.pack(side="left", padx=2)
        self.sim_stop_btn = ttk.Button(row, text="停止三维", width=9,
                                       command=lambda: self.worker.put(("sim_stop",)))
        self.sim_stop_btn.pack(side="left", padx=2)
        self.sim_stop_btn.state(["disabled"])
        self.sim_status_label = ttk.Label(row, text="未启动", foreground="#555")
        self.sim_status_label.pack(side="left", padx=4)
        ttk.Button(row, text="复位视角", width=9, command=self._reset_sim_camera).pack(side="right")
        row = ttk.Frame(tools)
        row.pack(fill="x", pady=2)
        self.preview_mode_combo = ttk.Combobox(row, textvariable=self.preview_mode_var,
                                               values=("目标姿态", "实际反馈"),
                                               state="readonly", width=10)
        self.preview_mode_combo.pack(side="left", padx=2)
        self.preview_mode_combo.bind("<<ComboboxSelected>>", self._preview_mode_changed)
        ttk.Label(row, text="拖动旋转 · 滚轮缩放", foreground="#555").pack(side="left", padx=4)
        ttk.Checkbutton(row, text="显示画面", variable=self.sim_preview_var,
                        command=self._refresh_sim_preview).pack(side="right")

        self.video_canvas = tk.Canvas(video, width=480, height=270,
                                      bg="#17212b", highlightthickness=0)
        self.video_canvas.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self._video_item = self.video_canvas.create_image(240, 135, anchor="center")
        self._video_hint = self.video_canvas.create_text(
            240, 135, text="选择摄像头并点击启动", fill="#dde6ed", font=("Microsoft YaHei UI", 11))
        self.sim_canvas = tk.Canvas(sim, width=480, height=270,
                                    bg="#101820", highlightthickness=0)
        self.sim_canvas.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self._sim_image_item = self.sim_canvas.create_image(240, 135, anchor="center")
        self._sim_hint = self.sim_canvas.create_text(
            240, 135, text=self._sim_hint_text, fill="#dde6ed", font=("Microsoft YaHei UI", 11))
        self.sim_canvas.bind("<ButtonPress-1>", self._sim_mouse_down)
        self.sim_canvas.bind("<B1-Motion>", self._sim_mouse_drag)
        self.sim_canvas.bind("<ButtonRelease-1>", self._sim_mouse_up)
        self.sim_canvas.bind("<MouseWheel>", self._sim_mouse_wheel)
        self.sim_canvas.bind("<Button-4>", lambda _e: self._zoom_sim_camera(1))
        self.sim_canvas.bind("<Button-5>", lambda _e: self._zoom_sim_camera(-1))

    def _build_sim_panel(self, pad):
        """Physical control and input selection are independent of 3D rendering."""
        box = ttk.LabelFrame(self, text="真机控制 · 三维预览可单独启停")
        box.grid(row=2, column=0, sticky="ew", **pad)
        controls = ttk.Frame(box)
        controls.pack(fill="x", padx=6, pady=(4, 0))
        ttk.Label(controls, text="控制方式").pack(side="left", padx=2)
        self.control_mode_combo = ttk.Combobox(
            controls, textvariable=self.control_source_var, state="readonly",
            values=tuple(CONTROL_SOURCES), width=12)
        self.control_mode_combo.pack(side="left", padx=4)
        self.control_mode_combo.bind("<<ComboboxSelected>>", self._control_mode_changed)
        self.control_enable_btn = ttk.Button(controls, text="启用真机控制", command=self._enable_control)
        self.control_enable_btn.pack(side="left", padx=4)
        self.control_pause_btn = ttk.Button(controls, text="暂停控制",
                                             command=lambda: self.worker.put(("control_mode", "off")))
        self.control_pause_btn.pack(side="left", padx=4)
        self.control_status_label = ttk.Label(controls, text="已暂停", foreground="#555")
        self.control_status_label.pack(side="left", padx=8)

        speed_row = ttk.Frame(box)
        speed_row.pack(fill="x", padx=6, pady=3)
        ttk.Label(speed_row, text="真机速度（1–255）").pack(side="left")
        ttk.Spinbox(speed_row, from_=1, to=255, width=5,
                    textvariable=self.follow_speed_var).pack(side="left", padx=4)
        ttk.Button(speed_row, text="应用速度", command=self._apply_follow_speed).pack(side="left")
        ttk.Label(speed_row, text="滑杆 / 数值输入设定目标；Enter 应用数值。暂停时仅预览。",
                  foreground="#555").pack(side="left", padx=12)

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
            entry = ttk.Entry(col, textvariable=shown, width=6, justify="center")
            entry.pack()
            entry.bind("<Return>", lambda event, j=i: self._joint_value_entered(j))
            self.sim_sliders.append(entry)
            actual = tk.StringVar(value="实际 --")
            self.actual_value_vars.append(actual)
            ttk.Label(col, textvariable=actual, foreground="#555", anchor="center").pack()

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
        self.vision_enable_btn = ttk.Button(status, text="启用视觉控制",
                                            command=lambda: self.worker.put(("control_mode", "vision")))
        self.vision_enable_btn.pack(side="right", padx=3)
        ttk.Label(status, text="识别就绪后连接并启用；无需启动仿真", foreground="#555").pack(side="right", padx=8)
        if VisionSource is None:
            self.vision_status_label.config(text=f"不可用: {VISION_IMPORT_ERROR}", foreground="#b00")

    def _refresh_task_controls(self):
        if self.task_start_btn is None:
            return
        connected = bool(self.worker.connected)
        idle = self._task_state == "idle"
        can_start = (connected and idle and self.control_mode_var.get() == "off"
                     and not self._closing)
        self.task_start_btn.state(["!disabled" if can_start else "disabled"])
        self._refresh_control_widgets()
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
        self._refresh_control_widgets()

    def _refresh_preview(self):
        vision = self.worker.vision
        self.video_canvas.coords(self._video_hint, self.video_canvas.winfo_width() / 2,
                                 self.video_canvas.winfo_height() / 2)
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

    def _refresh_sim_preview(self):
        sim = self.worker.sim
        canvas = self.sim_canvas
        size = (max(1, canvas.winfo_width()), max(1, canvas.winfo_height()))
        canvas.coords(self._sim_hint, size[0] / 2, size[1] / 2)
        if sim is not self._sim_preview_source:
            self._sim_preview_source = sim
            self._sim_preview_seq = 0
            self._sim_preview_rgb = self._sim_preview_image = None
            self._sim_preview_size = None
            self._sim_drag = None
            canvas.itemconfigure(self._sim_image_item, image="")
        preview = sim.preview if sim is not None else None
        if preview is None or not sim.alive():
            canvas.itemconfigure(self._sim_image_item, image="")
            canvas.itemconfigure(self._sim_hint, state="normal", text=self._sim_hint_text)
            return
        enabled = self.sim_preview_var.get()
        preview.set_enabled(enabled and canvas.winfo_ismapped() and not self._closing)
        if not enabled:
            canvas.itemconfigure(self._sim_image_item, image="")
            canvas.itemconfigure(self._sim_hint, state="normal", text="画面已暂停，仿真继续运行")
            self._sim_preview_size = None
            return
        packet = preview.frame(self._sim_preview_seq)
        if packet is not None:
            self._sim_preview_seq, self._sim_preview_rgb = packet
        if self._sim_preview_rgb is None:
            canvas.itemconfigure(self._sim_hint, state="normal", text=self._sim_hint_text)
            return
        if packet is None and size == self._sim_preview_size:
            return
        from PIL import Image, ImageTk, ImageOps
        picture = ImageOps.contain(Image.fromarray(self._sim_preview_rgb), size)
        self._sim_preview_image = ImageTk.PhotoImage(picture, master=self)
        self._sim_preview_size = size
        canvas.coords(self._sim_image_item, size[0] / 2, size[1] / 2)
        canvas.itemconfigure(self._sim_image_item, image=self._sim_preview_image)
        canvas.itemconfigure(self._sim_hint, state="hidden")

    def _camera_preview(self):
        sim = self.worker.sim
        return sim.preview if sim is not None and sim.alive() and not self._closing else None

    def _sim_mouse_down(self, event):
        self._sim_drag = (event.x, event.y)

    def _sim_mouse_up(self, _event):
        self._sim_drag = None

    def _sim_mouse_drag(self, event):
        preview = self._camera_preview()
        if preview is not None and self._sim_drag is not None:
            x, y = self._sim_drag
            preview.orbit((event.x - x) / max(1, self.sim_canvas.winfo_width()),
                          (event.y - y) / max(1, self.sim_canvas.winfo_height()))
        self._sim_drag = (event.x, event.y)

    def _sim_mouse_wheel(self, event):
        self._zoom_sim_camera(event.delta if sys.platform == "darwin" else event.delta / 120)
        return "break"

    def _zoom_sim_camera(self, steps):
        preview = self._camera_preview()
        if preview is not None:
            preview.zoom(steps)

    def _reset_sim_camera(self):
        preview = self._camera_preview()
        if preview is not None:
            preview.reset_camera()

    def _control_mode_changed(self, _event=None):
        self._control_pending = True
        self._refresh_control_widgets()
        self.worker.put(("select_source", CONTROL_SOURCES[self.control_source_var.get()]))

    def _enable_control(self):
        self.worker.put(("control_mode", CONTROL_SOURCES[self.control_source_var.get()]))

    def _preview_mode_changed(self, _event=None):
        self.worker.put(("preview_mode", "feedback" if self.preview_mode_var.get() == "实际反馈" else "target"))

    def _refresh_control_widgets(self):
        if self._closing:
            return
        idle = self._task_state == "idle"
        ready = idle and not self._control_pending
        active = self.control_mode_var.get() != "off"
        connected = self.worker.connected
        self.control_mode_combo.configure(state="readonly" if idle else "disabled")
        self.control_enable_btn.state(["!disabled" if ready and connected and not active else "disabled"])
        self.control_pause_btn.state(["!disabled" if idle and active else "disabled"])
        self.vision_enable_btn.state(["!disabled" if ready and connected else "disabled"])
        manual = self.control_source_var.get() != "视觉输入"
        self._set_sliders_enabled(ready and manual)
        self.preview_mode_combo.configure(state="disabled" if self.control_source_var.get() == "仿真调试" else "readonly")
        if not idle:
            self.control_status_label.config(text="抓握任务占用控制权", foreground="#b60")

    def _joint_value_entered(self, index):
        try:
            value = float(self.sim_value_vars[index].get())
            if not 0 <= value <= 1:
                raise ValueError
        except ValueError:
            self.sim_value_vars[index].set(f"{self.sim_slider_vars[index].get():.2f}")
            self.control_status_label.config(text="关节目标请输入 0–1 的数值", foreground="#b00")
            return
        self.sim_slider_vars[index].set(value)
        self._sim_slider_changed(index, value)

    def _apply_follow_speed(self):
        try:
            speed = int(self.follow_speed_var.get())
            if not 1 <= speed <= 255:
                raise ValueError
        except ValueError:
            self._log_append("真机速度请输入 1–255 的整数", "warn")
            return
        self.worker.put(("follow_speed", speed))

    def _set_sliders_enabled(self, on: bool):
        state = "!disabled" if on else "disabled"
        for s in self.sim_sliders:
            s.state([state])

    def _set_sliders(self, norms_sim):
        """回显接管时的姿态或视觉目标；实际位置另行显示。

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
        self.sim_value_vars[i].set(f"{float(value):.2f}")
        if self._slider_guard or self._control_pending:
            return
        self.worker.put(("manual_target", self._input_epoch, i, float(value)))

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
            self._refresh_sim_preview()
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
            for variable in self.actual_value_vars:
                variable.set("实际 --")
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
        elif kind == "task_release_error":
            self._task_state = "holding"
            self.task_status_label.config(text=payload[1], foreground="#b00")
            self._refresh_task_controls()
            self._log_append(payload[1], "warn")
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
        if kind == "starting":
            self._sim_failed = False
            self._sim_hint_text = "正在加载三维手模型…"
            self.sim_status_label.config(text="正在启动…", foreground="#b60")
            self.sim_start_btn.state(["disabled"])
            if not self._closing:
                self.sim_stop_btn.state(["!disabled"])
        elif kind == "stopping":
            self._sim_hint_text = "正在停止仿真…"
            self.sim_status_label.config(text="正在停止…", foreground="#555")
            self.sim_start_btn.state(["disabled"])
            self.sim_stop_btn.state(["disabled"])
        elif kind in ("ready", "sim_ready"):
            self._sim_hint_text = "正在生成三维画面…"
            self.sim_status_label.config(text="仿真运行中", foreground="green")
            self.sim_start_btn.state(["disabled"])
            if not self._closing:
                self.sim_stop_btn.state(["!disabled"])
            self._log_append("✓ 仿真已启动，三维画面显示在主窗口中", "ok")
        elif kind in ("closed", "sim_closed"):
            self._sim_hint_text = "仿真出错，请展开日志" if self._sim_failed else "三维预览已停止；手动和视觉控制可继续使用"
            self.sim_status_label.config(text="仿真出错" if self._sim_failed else "未启动",
                                         foreground="#b00" if self._sim_failed else "#555")
            self.sim_stop_btn.state(["disabled"])
            if not self._closing:
                self.sim_start_btn.state(["!disabled"])
            self._refresh_task_controls()
            self._log_append("仿真已停止")
        elif kind == "mode":
            _, mode, self._input_epoch, source = payload
            self._control_pending = False
            self.control_mode_var.set(mode)
            self.control_source_var.set(next(label for label, value in CONTROL_SOURCES.items() if value == source))
            self.control_status_label.config(text=CONTROL_LABELS[mode],
                                             foreground="#555" if mode == "off" else "green")
            self._refresh_task_controls()
            self._log_append(CONTROL_LABELS[mode])
        elif kind == "pose":
            # 接管 / 视觉目标已按统一关节顺序排列，回显不生成输入命令。
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
            if self.control_mode_var.get() == "off":
                self.control_status_label.config(text=payload[1], foreground="#b00")
            self._log_append("⚠ " + payload[1], "warn")
        elif kind in ("error", "sim_error"):
            # sim_error 是 SimSync 线程自己抛的(模型加载失败、渲染线程异常)。
            # 漏掉这个分支的话,仿真起不来时界面完全静默,只能去翻 gui.log。
            self._log_append("✗ " + payload[1], "warn")
            if kind == "sim_error" or self.worker.sim is None:
                if not self._closing:
                    self.sim_start_btn.state(["!disabled"])
                self.sim_stop_btn.state(["disabled"])
            if kind == "sim_error":
                self._sim_failed = True
                self._sim_hint_text = "仿真出错，请展开日志"
                self.sim_status_label.config(text="仿真出错", foreground="#b00")

    def _set_progress(self, i, total):
        # 简单进度条,放在状态区
        pct = int(i / total * 100) if total else 0
        self.info_label.config(text=f"{self.info_label.cget('text')} · {pct}%")

    def _refresh_motors(self, pos, cur):
        actual = reorder_real_to_sim([self.worker.cal.norm_from_pos(i, p) for i, p in enumerate(pos)])
        for variable, value in zip(self.actual_value_vars, actual):
            variable.set(f"实际 {value:.2f}")
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
