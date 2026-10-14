"""MuJoCo 离屏预览：渲染只在仿真线程，GUI 只读最新帧和提交视角参数。"""

from contextlib import contextmanager
import math
import threading
import time


class SimulationPreview:
    def __init__(self, width=640, height=480, fps=30):
        self.width, self.height, self.fps = width, height, fps
        self._lock = threading.Lock()
        self._enabled = True
        self._frame = None
        self._seq = 0
        self._azimuth, self._elevation, self._zoom = 60.0, -12.0, 0.85

    def set_enabled(self, enabled):
        """仅暂停出图，仿真步进、限速和真机跟随继续运行。"""
        with self._lock:
            self._enabled = bool(enabled)

    def orbit(self, dx, dy):
        """鼠标位移按预览区域宽高归一化。"""
        with self._lock:
            self._azimuth = (self._azimuth - float(dx) * 180) % 360
            self._elevation = max(-89.0, min(89.0, self._elevation + float(dy) * 180))

    def zoom(self, steps):
        with self._lock:
            self._zoom = max(0.15, min(4.0, self._zoom * math.exp(-0.12 * max(-20, min(20, steps)))))

    def reset_camera(self):
        with self._lock:
            self._azimuth, self._elevation, self._zoom = 60.0, -12.0, 0.85

    def frame(self, after_seq=0):
        """有新帧时返回 (序号, RGB)；像素只读，邮箱永远只保留最新一帧。"""
        with self._lock:
            if self._frame is None or self._seq == after_seq:
                return None
            return self._seq, self._frame

    @contextmanager
    def rendering(self, model):
        """创建、使用、释放 OpenGL 上下文必须在同一仿真线程完成。"""
        import mujoco

        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        base_distance = camera.distance
        next_frame = 0.0
        try:
            with mujoco.Renderer(model, height=self.height, width=self.width) as renderer:
                def refresh(data):
                    nonlocal next_frame
                    now = time.perf_counter()
                    with self._lock:
                        if not self._enabled or now < next_frame:
                            return
                        azimuth, elevation, zoom = self._azimuth, self._elevation, self._zoom
                    camera.azimuth, camera.elevation = azimuth, elevation
                    camera.distance = base_distance * zoom
                    renderer.update_scene(data, camera=camera)
                    pixels = renderer.render()
                    pixels.flags.writeable = False
                    with self._lock:
                        if self._enabled:
                            self._seq += 1
                            self._frame = pixels
                    # 不补历史帧，避免渲染积压阻塞仿真。
                    next_frame = now + 1.0 / self.fps

                yield refresh
        finally:
            with self._lock:
                self._frame = None
