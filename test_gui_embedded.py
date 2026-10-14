"""单窗口集成检查：实际 Tk + MuJoCo，禁止访问真机或摄像头。"""

import ctypes
import os
from pathlib import Path
import tempfile
import time
from tkinter import font
import unittest
from unittest.mock import patch

import glfw
import mujoco.viewer

import gui_control as gui


class EmbeddedGuiTests(unittest.TestCase):
    def test_preview_controls_resize_and_restart(self):
        errors = []
        create_window = glfw.create_window
        visible_windows = []

        def check_window(*args, **kwargs):
            window = create_window(*args, **kwargs)
            if window:
                visible_windows.append(glfw.get_window_attrib(window, glfw.VISIBLE))
            return window

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(gui, "__file__", str(Path(folder) / "gui_control.py")), \
                patch.object(gui.App, "_refresh_cameras"), \
                patch.object(gui.ROHand, "open", side_effect=AssertionError("hardware access")), \
                patch.object(glfw, "create_window", side_effect=check_window), \
                patch.object(mujoco.viewer, "launch_passive",
                             side_effect=AssertionError("unexpected native viewer")):
            app = gui.App(gui.Calibration.load())
            app.report_callback_exception = lambda *exc: errors.append(exc)
            app.geometry("1240x880+20+20")

            def wait_for(predicate, timeout=10):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    app.update()
                    self.assertFalse(errors, errors)
                    if predicate():
                        return
                    time.sleep(0.02)
                self.fail("GUI 未进入预期状态；" + app.log_text.get("1.0", "end"))

            def capture(name):
                destination = os.environ.get("DEXHAND_GUI_CAPTURE_DIR")
                if destination and os.name == "nt":
                    from PIL import ImageGrab
                    user = ctypes.WinDLL("user32")
                    user.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
                    user.GetAncestor.restype = ctypes.c_void_p
                    hwnd = user.GetAncestor(app.winfo_id(), 2)
                    path = Path(destination) / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    ImageGrab.grab(window=hwnd).save(path)

            try:
                app.update()
                baseline = (app.tk.call("tk", "scaling"),
                            font.nametofont("TkDefaultFont").metrics())
                for cycle in range(2):
                    app.sim_start_btn.invoke()
                    wait_for(lambda: app._sim_preview_image is not None)
                    sim = app.worker.sim
                    self.assertIsNotNone(sim.preview)
                    self.assertEqual(app.sim_canvas.itemcget(app._sim_hint, "state"), "hidden")
                    self.assertTrue(app.sim_start_btn.instate(["disabled"]))

                    seq = app._sim_preview_seq
                    app.sim_canvas.event_generate("<ButtonPress-1>", x=100, y=100)
                    app.sim_canvas.event_generate("<B1-Motion>", x=210, y=130)
                    app.sim_canvas.event_generate("<ButtonRelease-1>", x=210, y=130)
                    app.sim_canvas.event_generate("<MouseWheel>", delta=120)
                    wait_for(lambda: app._sim_preview_seq > seq)
                    app._reset_sim_camera()

                    app.sim_preview_var.set(False)
                    app._refresh_sim_preview()
                    frozen = sim.preview.frame()[0]
                    app.sim_slider_vars[0].set(0.35)
                    app._sim_slider_changed(0, "0.35")
                    wait_for(lambda: sim.action()[0] >= 0.34)
                    self.assertEqual(sim.preview.frame()[0], frozen)
                    self.assertEqual(app.sim_canvas.itemcget(app._sim_image_item, "image"), "")
                    app.sim_preview_var.set(True)
                    app._refresh_sim_preview()
                    wait_for(lambda: app._sim_preview_seq > frozen)

                    if cycle == 0:
                        capture("embedded_gui.png")
                        app.geometry("1080x720+20+20")
                        app.log_visible_var.set(True)
                        app._toggle_log()
                        wait_for(lambda: app._sim_preview_size == (
                            app.sim_canvas.winfo_width(), app.sim_canvas.winfo_height()))
                        # 小窗口 / 展开日志后，主要操作按钮仍在可见区域。
                        for widget in (app.sim_start_btn, app.task_release_btn,
                                       app.vision_start_btn, app.cur_labels[-1], app.log_text):
                            self.assertTrue(widget.winfo_ismapped(), str(widget))
                            self.assertLess(widget.winfo_rooty() + widget.winfo_height(),
                                            app.winfo_rooty() + app.winfo_height())
                            self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(),
                                                 app.winfo_rootx() + app.winfo_width())
                        capture("embedded_gui_small.png")
                        app.log_visible_var.set(False)
                        app._toggle_log()
                        app.geometry("1240x880+20+20")

                    app.sim_stop_btn.invoke()
                    wait_for(lambda: app.worker.sim is None and
                             app.sim_start_btn.instate(["!disabled"]))
                    self.assertFalse(sim.is_alive())
                    self.assertIsNone(sim.preview.frame())
                    self.assertEqual(baseline, (app.tk.call("tk", "scaling"),
                                               font.nametofont("TkDefaultFont").metrics()))
                self.assertEqual(visible_windows, [glfw.FALSE, glfw.FALSE])
                self.assertNotIn("预览显示失败", app.log_text.get("1.0", "end"))
                # 窗口关闭时仿真仍运行，也必须释放渲染线程。
                app.sim_start_btn.invoke()
                wait_for(lambda: app._sim_preview_image is not None)
                last_sim = app.worker.sim
            finally:
                app._on_close()
                app.mainloop()
            self.assertFalse(app.worker.is_alive())
            self.assertFalse(last_sim.is_alive())
            self.assertFalse(errors, errors)


if __name__ == "__main__":
    unittest.main()
