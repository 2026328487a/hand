"""Real Tk interaction with fake hardware; no camera or simulation dependency."""

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import gui_control as gui
from test_direct_control import FakeHand


class DirectGuiTests(unittest.TestCase):
    def test_manual_vision_switch_and_pause_without_simulation(self):
        cal = gui.Calibration.load()
        hand = FakeHand(cal)
        hand.open = lambda: hand
        hand.close = lambda: None
        hand.versions = lambda: {"firmware": "test", "uid": "fake"}
        hand.read_cali_data = lambda: None
        hand.wait_settled = lambda **kwargs: True
        errors = []
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(gui, "__file__", str(Path(folder) / "gui_control.py")), \
                patch.object(gui.App, "_refresh_cameras"), \
                patch.object(gui, "ROHand", return_value=hand), \
                patch.object(gui, "SimSync", None):
            app = gui.App(cal)
            app.report_callback_exception = lambda *exc: errors.append(exc)

            def wait_for(predicate, timeout=4):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    app.update()
                    self.assertFalse(errors, errors)
                    if predicate():
                        return
                    time.sleep(0.01)
                self.fail(app.log_text.get("1.0", "end"))

            try:
                app.worker.put("connect")
                wait_for(lambda: app.control_enable_btn.instate(["!disabled"]))
                app.control_enable_btn.invoke()
                wait_for(lambda: app.control_mode_var.get() == "manual")
                app.sim_value_vars[0].set("0.35")
                app._joint_value_entered(0)
                wait_for(lambda: app.worker.controller.action()[0] >= 0.34)
                wait_for(lambda: app.actual_value_vars[0].get() == "实际 0.35")
                self.assertEqual(app.sim_value_vars[0].get(), "0.35")
                self.assertIsNone(app.worker.sim)

                app.control_source_var.set("视觉输入")
                app._control_mode_changed()
                wait_for(lambda: not app._control_pending and app.control_mode_var.get() == "off")
                self.assertGreater(hand.stops, 0)
                self.assertTrue(app.sim_sliders[0].instate(["disabled"]))
                state = dict(state="tracking", error=None, control_ready=True, sequence=1,
                             captured_at=time.perf_counter(), norms=[0.6, 0, 0, 0, 0, 0],
                             frame_age=0, fps=30, latency_ms=20)
                vision = Mock()
                vision.snapshot.side_effect = lambda: dict(state)
                vision.alive.return_value = vision.is_alive.return_value = True
                vision.preview_frame.return_value = None

                def stop_vision():
                    vision.alive.return_value = vision.is_alive.return_value = False

                vision.stop.side_effect = stop_vision
                app.worker.vision = vision
                app.vision_enable_btn.invoke()
                wait_for(lambda: app.control_mode_var.get() == "vision")
                state.update(sequence=2, captured_at=time.perf_counter())
                wait_for(lambda: app.worker.controller.action()[0] >= 0.59)
                app.control_pause_btn.invoke()
                wait_for(lambda: app.control_mode_var.get() == "off")
                moves = len(hand.moves)
                state.update(sequence=3, captured_at=time.perf_counter(), norms=[0.2] * 6)
                wait_for(lambda: app.worker.controller.action()[0] < 0.5)
                self.assertEqual(len(hand.moves), moves)
                self.assertIsNone(app.worker.sim)
            finally:
                app._on_close()
                app.mainloop()
            self.assertFalse(app.worker.is_alive())
            self.assertFalse(errors, errors)


if __name__ == "__main__":
    unittest.main()
