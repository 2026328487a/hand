"""从视频里的人手 → 6 路归一化姿态 → 喂给 SimSync(仿真跟着动,真机再跟仿真)。

支持摄像头实时输入和视频文件回放。摄像头采集只保留最新帧，避免识别速度
低于相机帧率时积压；GUI 内嵌预览，CLI 可用 --preview 打开独立预览窗口。

数据流(接在已有架构的最上游,下游原封不动):

    视频帧 → MediaPipe 21 关键点 → 6 路归一化 → SimSync.set_target()
                                                      ↓ slew 限速
                                                  MuJoCo 显示
                                                      ↓ follow 模式
                                                  FollowSender → 真机

也就是说视频只是**第三种输入源**,和滑杆、手势按钮平级。限速、拇指让位互锁、
电流监视这些保护全都在下游,视频这一路白捡。

关于 MediaPipe 版本(**坑**):官方 demo(roh_demos/gesture_ctrled_rohand)钉的是
mediapipe 0.10.13,用 `mp.solutions.hands`。装上的 1.0.1 **把整个 mp.solutions
删掉了**,那份代码在新版上直接 AttributeError。这里用的是 Tasks API,而且它
不自带模型,要单独下 hand_landmarker.task(见 --model 默认路径)。

为什么用 world landmarks 而不是图像坐标:world landmarks 是以手掌几何中心为
原点的**米制 3D 坐标**,算关节角时天然对画幅、距离、手的朝向免疫。图像坐标的
z 量纲很怪,x/y 还带宽高比畸变,官方 demo 拿 y 坐标比大小判断"手指竖没竖起来",
手一转就失效 —— 而且那样只能得到 0/1 二值,拿不到连续开合量。

用法::

    python vision_hand.py --calibrate --video hand.mp4   # 先量区间
    python vision_hand.py --video hand.mp4 --no-hand     # 只驱动仿真
    python vision_hand.py --video hand.mp4               # 仿真 + 真机跟随
    python vision_hand.py --video hand.mp4 --preview     # 另开窗口看关键点
    python vision_hand.py --camera 0 --no-hand --preview  # 摄像头离线预览
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "models" / "hand_landmarker.task"

# ---------------- 关键点编号(MediaPipe 手部 21 点) ----------------
WRIST = 0
T_CMC, T_MCP, T_IP, T_TIP = 1, 2, 3, 4
I_MCP, I_PIP, I_DIP, I_TIP = 5, 6, 7, 8
M_MCP, M_PIP, M_DIP, M_TIP = 9, 10, 11, 12
R_MCP, R_PIP, R_DIP, R_TIP = 13, 14, 15, 16
P_MCP, P_PIP, P_DIP, P_TIP = 17, 18, 19, 20

FOUR = (("index", I_MCP, I_PIP, I_DIP, I_TIP),
        ("middle", M_MCP, M_PIP, M_DIP, M_TIP),
        ("ring", R_MCP, R_PIP, R_DIP, R_TIP),
        ("little", P_MCP, P_PIP, P_DIP, P_TIP))

# 画预览用的骨架连线。mediapipe 1.x 没了 solutions,连 HAND_CONNECTIONS 也一起
# 没了,所以这里直接写死 —— 21 点的拓扑是固定的,不会变。
CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4),
               (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0


def _turn(a, b, c, d) -> float:
    """两段骨之间的转折角(度)。伸直 = 0,对折 = 180。"""
    # acos(dot) 在两段接近共线时数值条件很差：world landmark 只有极小
    # 的帧间噪声，dot 接近 1，经过 acos 后就会变成明显的角度跳动。用
    # atan2(|cross|, dot) 同时利用正弦和余弦，保持几何意义不变但稳定得多。
    v1, v2 = _unit(b - a), _unit(d - c)
    return math.degrees(math.atan2(
        float(np.linalg.norm(np.cross(v1, v2))),
        float(np.clip(np.dot(v1, v2), -1.0, 1.0))))


def palm_frame(p, left: bool):
    """手掌坐标系。左手把横轴翻过来,左右手算出来同号。"""
    along = _unit(p[M_MCP] - p[WRIST])         # 掌根 → 中指根
    across = _unit(p[I_MCP] - p[P_MCP])        # 小指根 → 食指根(拇指那一侧)
    if left:
        across = -across
    return along, across, _unit(np.cross(across, along))


def features(p, left: bool) -> dict:
    """21 个 world landmarks(米) → 特征字典。probe_vision.py 量的就是这些。"""
    f = {}
    palm_len = max(float(np.linalg.norm(p[M_MCP] - p[WRIST])), 1e-9)
    palm_w = max(float(np.linalg.norm(p[I_MCP] - p[P_MCP])), 1e-9)

    for name, mcp, pip, dip, tip in FOUR:
        # 三段折角相加:掌骨→近节(把 MCP 屈曲算进来)、近节→中节、中节→远节。
        # ROH 的手一根手指只有一个主动自由度,远端是连杆耦合的,所以总弯曲量
        # 才是对的信号,不该只取某一个关节。
        f[f"{name}_curl"] = (_turn(p[WRIST], p[mcp], p[mcp], p[pip])
                             + _turn(p[mcp], p[pip], p[pip], p[dip])
                             + _turn(p[pip], p[dip], p[dip], p[tip]))
        f[f"{name}_reach"] = float(np.linalg.norm(p[tip] - p[WRIST])) / palm_len

    f["thumb_bend"] = (_turn(p[T_CMC], p[T_MCP], p[T_MCP], p[T_IP])
                       + _turn(p[T_MCP], p[T_IP], p[T_IP], p[T_TIP]))

    along, across, normal = palm_frame(p, left)
    t = p[T_MCP] - p[WRIST]
    t_in = t - np.dot(t, normal) * normal
    f["rot_inplane"] = math.degrees(math.atan2(float(np.dot(t_in, along)),
                                               float(np.dot(t_in, across))))
    f["rot_outplane"] = math.degrees(math.asin(float(np.clip(
        np.dot(_unit(t), normal), -1.0, 1.0))))
    # 拇指尖到小指根 / 掌宽。两个都是刚体内部距离,比值对旋转和尺度都免疫。
    # 实测这是几个候选里唯一能把"拇指张开"和"拇指扫向掌心"分开的。
    f["rot_tip2pinky"] = float(np.linalg.norm(p[T_TIP] - p[P_MCP])) / palm_w
    f["rot_tip2index"] = float(np.linalg.norm(p[T_TIP] - p[I_MCP])) / palm_w
    return f


# ---------------- 特征 → 归一化 ----------------

@dataclass
class DofMap:
    """一路自由度的映射。lo → 0(张开),hi → 1(闭合);invert 表示特征反向。"""
    feature: str
    lo: float
    hi: float
    invert: bool = False

    def norm(self, f: dict) -> float:
        v = f[self.feature]
        span = self.hi - self.lo
        x = 0.0 if abs(span) < 1e-9 else (v - self.lo) / span
        if self.invert:
            x = 1.0 - x
        return float(min(1.0, max(0.0, x)))


# 顺序 = 仿真序(lesson1_gestures.DOF):食指 中指 无名指 小指 拇指旋转 拇指弯曲。
# 区间由 probe_vision.py 在 5 张已知手势样图上量出来(见 README)。四指很稳,
# **拇指旋转那一路必须按自己的视频重标** —— 跑 --calibrate。
# DEFAULT_MAP = (
#     DofMap("index_curl", 45.0, 240.0),
#     DofMap("middle_curl", 45.0, 240.0),
#     DofMap("ring_curl", 45.0, 240.0),
#     DofMap("little_curl", 45.0, 240.0),
#     DofMap("rot_tip2pinky", 0.80, 1.80, invert=True),   # thumb_root = 旋转
#     DofMap("thumb_bend", 25.0, 115.0),                  # thumb = 弯曲
# )
DEFAULT_MAP = (
    DofMap("index_curl", 57.22, 225.57),
    DofMap("middle_curl", 37.03, 259.93),
    DofMap("ring_curl", 26.91, 245.43),
    DofMap("little_curl", 48.60, 225.06),
    DofMap("rot_tip2pinky", 0.54, 1.94, invert=True),
    DofMap("thumb_bend", 22.84, 80.45),
)


class LandmarkMapper:
    """特征 → 6 路归一化,带指数平滑和丢帧保持。

    为什么要平滑:关键点本身有抖动,直接喂下去六个电机会持续小幅抽搐 —— 电机
    不像仿真,每次微动都是真实的机械磨损和电流。EMA 很便宜,效果立竿见影。

    为什么丢帧要**保持**而不是归零:手临时被遮挡/出画时检测会断几帧,归零等于
    命令手突然张开,真机上就是一次猛动。保持上一帧是安全的做法。
    """

    def __init__(self, maps=DEFAULT_MAP, ema: float = 0.15,
                 hold_frames: int = 15, median_window: int = 3,
                 deadband: float = 0.003):
        self.maps = tuple(maps)
        self.ema = float(min(1.0, max(0.0, ema)))
        self.hold_frames = hold_frames
        self.median_window = max(1, int(median_window))
        self.deadband = max(0.0, float(deadband))
        self._smooth = None
        self._raw_history = deque(maxlen=self.median_window)
        self._missing = 0

    def reset(self):
        self._smooth = None
        self._raw_history.clear()
        self._missing = 0

    def update(self, p, left: bool):
        """有检测结果时调。返回平滑后的 6 路归一化(仿真序)。"""
        measured = features(p, left)
        raw = np.array([m.norm(measured) for m in self.maps])
        # MediaPipe 的单帧误差通常是孤立尖峰；中值滤波先去掉尖峰，EMA
        # 再负责连续变化的平滑，避免单个坏帧把姿态条和真机拉动一下。
        self._raw_history.append(raw)
        filtered = np.median(np.stack(tuple(self._raw_history)), axis=0)
        if self._smooth is None:
            self._smooth = filtered
        else:
            candidate = self.ema * filtered + (1.0 - self.ema) * self._smooth
            # 小于约 0.3% 全行程的变化视为检测噪声，保持上一次输出。
            delta = candidate - self._smooth
            self._smooth = np.where(np.abs(delta) < self.deadband,
                                    self._smooth, candidate)
        self._missing = 0
        return self._smooth.copy()

    def miss(self):
        """这一帧没检测到手。返回该保持的姿态,或 None(丢太久了)。"""
        self._missing += 1
        if self._smooth is None or self._missing > self.hold_frames:
            return None
        return self._smooth.copy()


# ---------------- 检测器 ----------------

class HandTracker:
    """MediaPipe Tasks 的薄封装。VIDEO 模式要求时间戳**严格递增**。"""

    def __init__(self, model_path=DEFAULT_MODEL, num_hands: int = 1,
                 min_conf: float = 0.5, handedness: str = "any"):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (HandLandmarker,
                                                   HandLandmarkerOptions,
                                                   RunningMode)
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到模型 {model_path}\n"
                "下载: curl -L -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task")
        self._mp = mp
        self.handedness = handedness
        self.det = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=max(num_hands, 2 if handedness != "any" else 1),
            min_hand_detection_confidence=min_conf,
            min_tracking_confidence=min_conf))
        self._last_ts = -1

    def detect(self, frame_bgr, ts_ms: int):
        """返回 (world_landmarks, 是否左手, 图像坐标 landmarks) 或 None。"""
        import cv2
        mp = self._mp
        # 时间戳必须递增,相等都不行 —— 视频帧率算出来的 ms 会撞,这里兜一下
        ts_ms = max(int(ts_ms), self._last_ts + 1)
        self._last_ts = ts_ms
        res = self.det.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), ts_ms)
        if not res.hand_world_landmarks:
            return None
        candidates = [i for i, side in enumerate(res.handedness)
                      if self.handedness == "any" or side[0].category_name == self.handedness]
        if not candidates:
            return None
        index = candidates[0]
        left = res.handedness[index][0].category_name == "Left"
        world = np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[index]])
        image = np.array([[l.x, l.y] for l in res.hand_landmarks[index]])
        return world, left, image

    def close(self):
        try:
            self.det.close()
        except Exception:                                   # noqa: BLE001
            pass


def draw_overlay(frame, image_lm, norms, names):
    """在帧上画骨架 + 6 路条形图。只用于 --preview。"""
    import cv2
    h, w = frame.shape[:2]
    if image_lm is not None:
        pts = [(int(x * w), int(y * h)) for x, y in image_lm]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
    if norms is None:
        cv2.putText(frame, "no hand", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        return frame
    for i, (v, name) in enumerate(zip(norms, names)):
        y = 24 + i * 22
        cv2.rectangle(frame, (110, y - 11), (110 + 160, y + 3), (60, 60, 60), 1)
        cv2.rectangle(frame, (110, y - 11), (110 + int(160 * v), y + 3),
                      (0, 180, 255), -1)
        cv2.putText(frame, f"{name} {v:.2f}", (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
    return frame


# ---------------- 视频 → SimSync ----------------

class VisionSource(threading.Thread):
    """Asynchronous camera/file inference with a bounded preview mailbox.

    No Tk or CAN calls run here. The CAN owner checks snapshot() before sending.
    """

    LOST_TIMEOUT = 0.6

    def __init__(self, source, sync=None, mapper=None, model_path=DEFAULT_MODEL,
                 loop=True, preview=False, realtime=True, on_event=None,
                 mirror=False, handedness="any"):
        super().__init__(daemon=True, name="VisionSource")
        self.source, self.sync = source, sync
        self.mapper = mapper or LandmarkMapper()
        self.model_path = model_path
        self.loop, self.preview, self.realtime = loop, preview, realtime
        self.mirror, self.handedness = mirror, handedness
        self.on_event = on_event or (lambda *a: None)
        self.norms = None
        self.fps_real = 0.0
        self.detected = False
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self._ready_ok = False
        self._err = None
        self._lock = threading.Lock()
        self._preview_frame = None
        self._preview_seq = 0
        self._state = "starting"
        self._last_detection = None
        self._last_frame = None
        self._latency_ms = 0.0
        self._camera = None

    def set_sync(self, sync):
        with self._lock:
            self.sync = sync

    def _freeze_target(self):
        if self.sync is not None:
            self.sync.set_target(self.sync.action())

    def stop(self):
        # Serialize with set_target so no inference result can write after stop.
        with self._lock:
            self._stop_event.set()
            self.detected = False
            self._state = "stopping"
            self._freeze_target()
        if self._camera is not None:
            self._camera.stop()

    def alive(self):
        return self.is_alive() and not self._stop_event.is_set()

    def wait_ready(self, timeout=30.0):
        return self._ready.wait(timeout) and self._ready_ok and self.alive()

    def snapshot(self):
        now = time.perf_counter()
        with self._lock:
            age = float("inf") if self._last_detection is None else now - self._last_detection
            frame_age = float("inf") if self._last_frame is None else now - self._last_frame
            return dict(state=self._state, detected=self.detected,
                        fps=self.fps_real, latency_ms=self._latency_ms,
                        frame_age=frame_age, detection_age=age,
                        control_ready=(self._ready_ok and not self._stop_event.is_set()
                                       and self.detected and age < self.LOST_TIMEOUT
                                       and frame_age < self.LOST_TIMEOUT),
                        tracking_lost=(age >= self.LOST_TIMEOUT or frame_age >= self.LOST_TIMEOUT),
                        error=None if self._err is None else str(self._err))

    def preview_frame(self, after=0):
        with self._lock:
            if self._preview_seq <= after:
                return None
            return self._preview_seq, self._preview_frame

    def run(self):
        cap = tracker = None
        try:
            import cv2
            is_camera = isinstance(self.source, int)
            if is_camera:
                from camera_capture import LatestCamera
                self._camera = LatestCamera(self.source)
                self._camera.start()
                packet = self._camera.read_latest(timeout=10.0)
                if packet is None:
                    raise RuntimeError(str(self._camera.error or "摄像头 10 秒内没有返回画面"))
                fps = 30.0
            else:
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    raise RuntimeError(f"打不开视频源 {self.source!r}")
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                if not (1.0 < fps < 240.0):
                    fps = 30.0
            if self._stop_event.is_set():
                return
            tracker = HandTracker(self.model_path, handedness=self.handedness)
            if self._stop_event.is_set():
                return
            sequence = frame_i = 0
            started = next_frame = time.perf_counter()
            while not self._stop_event.is_set():
                if is_camera:
                    packet = self._camera.read_latest(sequence)
                    if packet is None:
                        if self._camera.error is not None:
                            raise self._camera.error
                        if time.perf_counter() - (self._last_frame or started) > 2.0:
                            raise RuntimeError("摄像头画面已中断")
                        continue
                    sequence, captured, frame = packet
                    timestamp = int(captured * 1000)
                else:
                    ok, frame = cap.read()
                    if not ok and self.loop and frame_i:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        self.mapper.reset()
                    if not ok:
                        if not frame_i:
                            raise RuntimeError("视频没有可解码的画面")
                        break
                    captured = time.perf_counter()
                    timestamp = int(frame_i * 1000.0 / fps)
                res = tracker.detect(frame, timestamp)
                if self._stop_event.is_set():
                    break
                if res is None:
                    norms = self.mapper.miss()
                    image_lm = None
                else:
                    world, left, image_lm = res
                    norms = self.mapper.update(world, left)
                processed = time.perf_counter()
                frame_i += 1
                with self._lock:
                    if self._stop_event.is_set():
                        break
                    was_detected = self.detected
                    self.detected = res is not None
                    self.norms = None if norms is None else norms.copy()
                    self._last_frame = captured
                    self._latency_ms = (processed - captured) * 1000.0
                    self.fps_real = frame_i / max(processed - started, 0.001)
                    self._state = "tracking" if self.detected else "missing"
                    if self.detected:
                        self._last_detection = captured
                        if self.sync is not None and processed - captured < self.LOST_TIMEOUT:
                            self.sync.set_target(norms)
                    elif was_detected:
                        # Do not continue slewing toward a stale gesture when the hand vanishes.
                        self._freeze_target()
                # Resize before drawing; mirror display only, never the control coordinates.
                scale = min(480 / frame.shape[1], 360 / frame.shape[0], 1.0)
                shown = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
                display_lm = image_lm
                if self.mirror:
                    shown = cv2.flip(shown, 1)
                    display_lm = None if image_lm is None else image_lm.copy()
                    if display_lm is not None:
                        display_lm[:, 0] = 1.0 - display_lm[:, 0]
                shown = draw_overlay(shown, display_lm, norms,
                                     ("index", "middle", "ring", "little", "thumb rot", "thumb bend"))
                rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._preview_seq += 1
                    self._preview_frame = rgb
                if not self._ready_ok:
                    self._ready_ok = True
                    self._ready.set()
                    self.on_event("vision_ready", f"{self.source} · {frame.shape[1]}×{frame.shape[0]}")
                if self.preview:
                    cv2.imshow("vision_hand", shown)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                if self.realtime and not is_camera:
                    next_frame += 1.0 / fps
                    self._stop_event.wait(max(0, next_frame - time.perf_counter()))
        except Exception as exc:
            self._err = exc
            self.on_event("vision_error", str(exc))
        finally:
            self.stop()
            self._ready.set()
            if cap is not None:
                cap.release()
            self.on_event("vision_stopping")
            if self._camera is not None:
                self._camera.stop()
                # Keep this source alive until the camera owner releases the device.
                while self._camera.is_alive():
                    self._camera.join(timeout=0.2)
            if self.preview:
                try:
                    import cv2
                    cv2.destroyWindow("vision_hand")
                except Exception:
                    pass
            if tracker is not None:
                tracker.close()
            with self._lock:
                self._state = "error" if self._err is not None else "stopped"
            self.on_event("vision_closed")


# ---------------- 标定 ----------------

def calibrate(source, model_path=DEFAULT_MODEL, lo_pct=5.0, hi_pct=95.0,
              max_frames=6000):
    """把视频过一遍,统计各特征的分位数,给出建议区间。

    用分位数而不是 min/max:关键点偶尔会跳出一两个离谱值,取极值会把整个区间
    撑坏,结果就是正常动作只用到刻度的一小段。
    """
    import cv2
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频源 {source!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 < fps < 240.0):
        fps = 30.0
    tracker = HandTracker(model_path)
    acc: dict = {}
    n_frame = n_hit = 0
    try:
        while n_frame < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            res = tracker.detect(frame, int(n_frame * 1000.0 / fps))
            n_frame += 1
            if res is None:
                continue
            n_hit += 1
            world, left, _ = res
            for k, v in features(world, left).items():
                acc.setdefault(k, []).append(v)
    finally:
        tracker.close()
        cap.release()

    if not n_hit:
        raise RuntimeError(f"{n_frame} 帧里一只手都没检出 —— 换个视频?")

    print(f"共 {n_frame} 帧,检出 {n_hit} 帧 ({n_hit / n_frame * 100:.0f}%)\n")
    print(f"{'特征':16s} {f'p{lo_pct:g}':>10s} {f'p{hi_pct:g}':>10s} "
          f"{'最小':>10s} {'最大':>10s}")
    stats = {}
    for k, vals in acc.items():
        a = np.asarray(vals)
        lo, hi = np.percentile(a, [lo_pct, hi_pct])
        stats[k] = (float(lo), float(hi))
        print(f"{k:16s} {lo:10.2f} {hi:10.2f} {a.min():10.2f} {a.max():10.2f}")

    print("\n把下面这段替换 vision_hand.py 里的 DEFAULT_MAP:\n")
    print("DEFAULT_MAP = (")
    for m in DEFAULT_MAP:
        lo, hi = stats.get(m.feature, (m.lo, m.hi))
        if m.invert:
            print(f'    DofMap("{m.feature}", {lo:.2f}, {hi:.2f}, invert=True),')
        else:
            print(f'    DofMap("{m.feature}", {lo:.2f}, {hi:.2f}),')
    print(")")
    return stats


# ---------------- 命令行 ----------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="视频文件路径")
    src.add_argument("--camera", type=int, help="摄像头序号")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="landmarker 模型")
    ap.add_argument("--calibrate", action="store_true", help="只统计区间,不驱动")
    ap.add_argument("--preview", action="store_true", help="另开窗口看关键点")
    ap.add_argument("--no-hand", action="store_true", help="只驱动仿真,不连真机")
    ap.add_argument("--no-loop", action="store_true", help="放完就停,不循环")
    ap.add_argument("--mode", choices=["normalized", "angle"],
                    default="normalized")
    ap.add_argument("--ema", type=float, default=0.15, help="平滑系数,越小越稳")
    ap.add_argument("--left", action="store_true", help="用左手仿真模型")
    ap.add_argument("--mirror", action="store_true", help="镜像预览，不改变控制坐标")
    ap.add_argument("--handedness", choices=("any", "Left", "Right"), default="any")
    args = ap.parse_args()

    source = args.camera if args.camera is not None else args.video
    if args.calibrate:
        calibrate(source, args.model)
        return 0

    from sim_sync import (DOF_CN, FollowSender, SimSync, align_sim_to_hand,
                          DEFAULT_HZ)
    from calibration import Calibration
    from rohand import ROHand

    sync = SimSync(left=args.left, on_event=lambda *a: print("  ·", *a))
    hand = sender = vision = None
    armed = False
    exit_code = 0
    try:
        sync.start()
        if not sync.wait_ready():
            raise RuntimeError("仿真没能启动")
        vision = VisionSource(source, sync, mapper=LandmarkMapper(ema=args.ema),
                              model_path=args.model, loop=not args.no_loop,
                              preview=args.preview, mirror=args.mirror,
                              handedness=args.handedness,
                              on_event=lambda *a: print("  ·", *a))
        vision.start()
        if not vision.wait_ready():
            raise RuntimeError(f"视频源打不开: {vision._err}")
        print("运行中 —— 关掉 MuJoCo 窗口或 Ctrl-C 结束；丢手后停止，需重新运行")
        last = 0.0
        while sync.alive() and vision.alive():
            state = vision.snapshot()
            if not args.no_hand and not armed and state["control_ready"]:
                cal = Calibration.load()
                hand = ROHand().open()
                print(f"已连接真机 {hand.versions()}")
                align_sim_to_hand(sync, cal, hand)
                sender = FollowSender(cal, sync, mode=args.mode,
                                      on_warn=lambda m: print(f"  ⚠ {m}"))
                armed = True
            if armed:
                if state["tracking_lost"] or state["error"]:
                    print("手部识别 / 画面中断，停止真机跟随")
                    break
                sender.tick(hand)
            now = time.perf_counter()
            if now - last > 1.0:
                last = now
                print(f"  {state['state']} · {state['fps']:.1f} fps · 处理 {state['latency_ms']:.0f} ms")
            time.sleep(1.0 / DEFAULT_HZ)
        if vision._err is not None:
            raise RuntimeError(str(vision._err))
    except KeyboardInterrupt:
        print("中断")
    except Exception as exc:
        print(f"✗ {exc}")
        exit_code = 1
    finally:
        if vision is not None:
            vision.stop()
        # Stop CAN before waiting for slow native model cleanup.
        if hand is not None:
            try:
                hand.stop()
            except Exception as exc:
                print(f"✗ 停止命令失败: {exc}")
                exit_code = 1
            finally:
                hand.close()
        sync.stop()
        if sync.ident is not None:
            sync.join(timeout=5)
        if vision is not None and vision.ident is not None:
            vision.join(timeout=15)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
