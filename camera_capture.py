"""Camera discovery and a single-frame mailbox for low-latency capture."""

import sys
import threading
import time


def list_cameras():
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    return [(camera.index, camera.name) for camera in enumerate_cameras(backend)]


class LatestCamera(threading.Thread):
    """The capture thread owns VideoCapture; inference only consumes its newest frame.

    A slow inference never builds an ever-growing queue of old camera frames.
    Stop does not release the native capture from a competing thread.
    """

    def __init__(self, index, width=640, height=480, fps=30):
        super().__init__(daemon=True, name=f"Camera-{index}")
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._latest = None
        self.error = None

    def stop(self):
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

    def read_latest(self, after=0, timeout=0.2):
        with self._condition:
            self._condition.wait_for(
                lambda: (self._latest is not None and self._latest[0] > after)
                or self.error is not None or self._stop_event.is_set(), timeout)
            if self._latest is not None and self._latest[0] > after:
                return self._latest
            return None

    def run(self):
        import cv2

        cap = None
        try:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {self.index}，请检查占用或系统相机权限")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            sequence = 0
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("摄像头取帧失败，设备可能已断开")
                sequence += 1
                with self._condition:
                    self._latest = (sequence, time.perf_counter(), frame)
                    self._condition.notify_all()
        except Exception as exc:
            self.error = exc
        finally:
            if cap is not None:
                cap.release()
            self.stop()
