"""内嵌渲染回归：真实 OpenGL 隐藏窗口、控制与预览解耦、错误清理。"""

import threading
import time
import unittest
from unittest.mock import patch

import glfw
import mujoco.viewer
import numpy as np

from gui_startup import configure_window_scaling
from sim_sync import SimSync


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("等待仿真状态超时")


class EmbeddedSimulationTests(unittest.TestCase):
    def test_frames_camera_pause_and_restart_without_visible_viewer(self):
        configure_window_scaling()
        create_window = glfw.create_window
        visibility = []

        def hidden_window(*args, **kwargs):
            window = create_window(*args, **kwargs)
            if window:
                visibility.append(glfw.get_window_attrib(window, glfw.VISIBLE))
            return window

        with patch.object(glfw, "create_window", side_effect=hidden_window), \
                patch.object(mujoco.viewer, "launch_passive",
                             side_effect=AssertionError("GUI must not open a viewer")):
            for _ in range(2):
                sim = SimSync(display="embedded")
                sim.start()
                try:
                    self.assertTrue(sim.wait_ready(10), sim._err)
                    seq, pixels = wait_for(sim.preview.frame)
                    self.assertEqual(pixels.shape, (480, 640, 3))
                    self.assertGreater(pixels.std(), 1)
                    self.assertFalse(pixels.flags.writeable)

                    sim.preview.orbit(0.25, 0.1)
                    moved_seq, moved = wait_for(lambda: sim.preview.frame(seq))
                    self.assertGreater(np.abs(moved.astype(float) - pixels).mean(), 1)
                    sim.preview.zoom(3)
                    zoom_seq, zoomed = wait_for(lambda: sim.preview.frame(moved_seq))
                    self.assertGreater(np.abs(zoomed.astype(float) - moved).mean(), 1)
                    sim.preview.reset_camera()
                    wait_for(lambda: sim.preview.frame(zoom_seq))

                    sim.preview.set_enabled(False)
                    frozen_seq = sim.preview.frame()[0]
                    sim.set_target([0.4, 0, 0, 0, 0, 0])
                    wait_for(lambda: sim.action()[0] >= 0.39)
                    self.assertEqual(sim.preview.frame()[0], frozen_seq)
                    self.assertTrue(sim.alive())
                    sim.preview.set_enabled(True)
                    wait_for(lambda: sim.preview.frame(frozen_seq))
                finally:
                    sim.stop()
                    sim.join(5)
                self.assertFalse(sim.is_alive())
                self.assertIsNone(sim.preview.frame())
                self.assertIsNone(sim._err)
        self.assertEqual(visibility, [glfw.FALSE, glfw.FALSE])

    def test_render_failure_releases_context_and_reports_closed(self):
        configure_window_scaling()
        events = []
        renderer_factory = mujoco.Renderer
        renderers = []

        def broken_renderer(*args, **kwargs):
            renderer = renderer_factory(*args, **kwargs)
            renderers.append(renderer)
            renderer.render = lambda: (_ for _ in ()).throw(RuntimeError("render failed"))
            return renderer

        sim = SimSync(display="embedded", on_event=lambda *event: events.append(event))
        with patch.object(mujoco, "Renderer", side_effect=broken_renderer):
            sim.start()
            sim.join(10)
        self.assertFalse(sim.is_alive())
        self.assertFalse(sim.wait_ready(0))
        self.assertIn("render failed", str(sim._err))
        self.assertEqual([e[0] for e in events], ["sim_error", "sim_closed"])
        self.assertIsNone(sim.preview.frame())
        self.assertTrue(renderers)
        self.assertIsNone(renderers[0]._gl_context)

    def test_stop_during_load_does_not_open_renderer(self):
        entered, release = threading.Event(), threading.Event()
        from sim_sync import load

        def slow_load(**kwargs):
            entered.set()
            release.wait(5)
            return load(**kwargs)

        events = []
        sim = SimSync(display="embedded", on_event=lambda *event: events.append(event))
        with patch("sim_sync.load", side_effect=slow_load), \
                patch.object(sim.preview, "rendering") as rendering:
            sim.start()
            try:
                self.assertTrue(entered.wait(5))
            finally:
                sim.stop()
                release.set()
                sim.join(10)
        self.assertFalse(sim.is_alive())
        self.assertFalse(sim.wait_ready(0))
        rendering.assert_not_called()
        self.assertEqual(events, [("sim_closed",)])

    def test_model_failure_unblocks_waiter_and_reports_closed(self):
        events = []
        sim = SimSync(display="embedded", on_event=lambda *event: events.append(event))
        with patch("sim_sync.load", side_effect=RuntimeError("model missing")):
            sim.start()
            sim.join(5)
        self.assertFalse(sim.is_alive())
        self.assertFalse(sim.wait_ready(0))
        self.assertEqual([e[0] for e in events], ["sim_error", "sim_closed"])


if __name__ == "__main__":
    unittest.main()
