"""Real Tk interaction with fake hardware; no camera or simulation dependency."""

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import gui_control as gui
from test_direct_control import FakeHand


class DirectGuiTests(unittest.TestCase):
    def test_fingertip_mode_selector_and_fallback_status_without_devices(self):
        errors = []
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(gui, '__file__', str(Path(folder) / 'gui_control.py')), \
                patch.object(gui.App, '_refresh_cameras'), \
                patch.object(gui.ROHand, 'open', side_effect=AssertionError('hardware access')):
            app = gui.App(gui.Calibration.load())
            app.withdraw()
            app.report_callback_exception = lambda *exc: errors.append(exc)
            try:
                self.assertEqual(app.vision_retarget_var.get(), '原样全速')
                self.assertFalse(app.rate_limit_var.get())
                with patch.object(app.worker, 'put') as send:
                    app.rate_limit_check.invoke()
                self.assertEqual(send.call_args.args[0], ('rate_limit', True))
                with patch.object(app.worker, 'put') as send:
                    app.collision_avoidance_check.invoke()
                self.assertEqual(send.call_args.args[0], ('collision_avoidance', False))
                app.follow_hz_var.set('45')
                with patch.object(app.worker, 'put') as send:
                    app.follow_hz_apply_btn.invoke()
                self.assertEqual(send.call_args.args[0], ('follow_hz', 45))
                app._set_vision_controls(True)
                self.assertEqual(str(app.vision_retarget_combo.cget('state')), 'disabled')
                app._set_vision_controls(False)
                self.assertEqual(str(app.vision_retarget_combo.cget('state')), 'readonly')
                state = dict(state='tracking', frame_age=0, fps=30, latency_ms=40,
                             control_ready=True, retargeting={'state': 'fallback'})
                app._handle_sim_event(('vision_state', state))
                self.assertIn('原有映射回退', app.vision_status_label.cget('text'))
                app._handle_sim_event(('retarget_fallback', '模型不可用，使用原有映射'))
                self.assertIn('模型不可用', app.log_text.get('1.0', 'end'))
                app.vision_source_var.set('视频文件')
                video = Path(folder) / 'test.mp4'
                video.touch()
                app.vision_path_var.set(str(video))
                with patch.object(app.worker, 'put') as send:
                    app._start_vision()
                self.assertEqual(send.call_args.args[0][-1], 'direct')
                app._handle_sim_event(('motion_settings', dict(direct=True, speed=255, hz=60,
                                                              rate_limit=False, collision_avoidance=False)))
                self.assertEqual(app.follow_speed_var.get(), '255')
                self.assertEqual(app.follow_hz_var.get(), '60')
                self.assertFalse(app.collision_avoidance_var.get())
                self.assertIn('disabled', app.collision_avoidance_check.state())
                self.assertIn('disabled', app.rate_limit_check.state())
                self.assertIn('disabled', app.follow_speed_spin.state())
                app._handle_sim_event(('motion_settings', dict(direct=False, speed=140, hz=30,
                                                              rate_limit=False, collision_avoidance=True)))
                self.assertNotIn('disabled', app.collision_avoidance_check.state())
                self.assertEqual(app.follow_speed_var.get(), '140')
                app.vision_retarget_var.set('原有映射')
                with patch.object(app.worker, 'put') as send:
                    app._start_vision()
                self.assertEqual(send.call_args.args[0][-1], 'features')
            finally:
                app._on_close()
                app.mainloop()
            self.assertFalse(errors, errors)

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
