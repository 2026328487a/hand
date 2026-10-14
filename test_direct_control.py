"""No hardware: direct control, source arbitration, and cancellation regression."""

import queue
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from calibration import Calibration
from control_core import MotionController, SafeSender, THUMB_BEND_SAFE
from gui_control import HandWorker


class FakeHand:
    def __init__(self, cal):
        self.pos = cal.pose_from_norm([0.0] * 6)
        self.currents = [0] * 6
        self.moves = []
        self.stops = 0

    def read_pos(self):
        return list(self.pos)

    def read_currents(self):
        return list(self.currents)

    def move(self, pose, speed=255):
        self.pos = list(pose)
        self.moves.append((list(pose), speed))

    def stop(self):
        self.stops += 1


class DirectControlTests(unittest.TestCase):
    def setUp(self):
        self.worker = HandWorker(Calibration.load(), queue.Queue(), queue.Queue())
        self.hand = self.worker.hand = FakeHand(self.worker.cal)
        self.worker._connected = True

    def tick(self, dt=1 / 30):
        self.worker._control_t = time.monotonic() - dt
        self.worker._control_tick()

    def target(self, index, value, epoch=None):
        self.worker.put(("manual_target", self.worker._input_epoch if epoch is None else epoch,
                         index, value))

    def vision(self, value=0.7):
        state = dict(control_ready=True, error=None, state="tracking",
                     sequence=1, captured_at=time.perf_counter(), norms=[value, 0, 0, 0, 0, 0])
        source = Mock()
        source.snapshot.side_effect = lambda: dict(state)
        source.alive.return_value = source.is_alive.return_value = True
        self.worker.vision = source
        return source, state

    def test_manual_control_without_simulation_aligns_then_slews(self):
        self.hand.pos = self.worker.cal.pose_from_norm([0.0, 0.4, 0.0, 0.0, 0.0, 0.0])
        self.target(0, 1.0)  # A paused preview must not become the armed target.
        self.worker._set_control_mode("manual")
        self.assertIsNone(self.worker.sim)
        np.testing.assert_allclose(self.worker.controller.action(), [0.4, 0, 0, 0, 0, 0], atol=1e-4)
        self.tick()
        self.assertEqual(self.hand.moves[-1][0], self.hand.pos)
        self.target(0, 0.8)
        self.tick()
        self.assertGreater(self.worker.controller.action()[0], 0.4)
        self.assertLess(self.worker.controller.action()[0], 0.46)

    def test_paused_slider_is_preview_only(self):
        self.target(0, 0.8)
        self.tick()
        self.assertGreater(self.worker.controller.action()[0], 0)
        self.assertEqual(self.hand.moves, [])

    def test_latest_manual_target_replaces_backlog(self):
        self.worker._set_control_mode("manual")
        for value in np.linspace(0, 1, 1000):
            self.target(0, value)
        self.target(0, 0.2)
        self.assertTrue(self.worker.commands.empty())
        self.assertEqual(len(self.worker._manual_pending), 1)
        self.tick()
        self.assertEqual(self.worker.controller.status()["target"][0], 0.2)

    def test_pause_stops_hand_and_discards_old_epoch(self):
        self.worker._set_control_mode("manual")
        epoch = self.worker._input_epoch
        self.worker._set_control_mode("off")
        self.target(0, 1, epoch)
        self.tick()
        self.assertEqual(self.hand.stops, 1)
        self.assertEqual(self.hand.moves, [])
        self.assertEqual(self.worker.controller.action()[0], 0)

    def test_stop_invalidates_queued_enable_and_slider_commands(self):
        self.worker.put(("control_mode", "manual"))
        self.target(0, 1)
        self.worker.put("stop")
        while not self.worker.commands.empty():
            self.worker._dispatch(self.worker.commands.get_nowait())
        self.tick()
        self.assertEqual(self.worker.control_mode, "off")
        self.assertEqual(self.hand.moves, [])

    def test_stop_arriving_during_feedback_prevents_next_motion(self):
        self.worker._set_control_mode("manual")
        self.target(0, 0.8)

        def read_currents():
            self.worker.put("stop")
            return [0] * 6

        self.hand.read_currents = read_currents
        self.tick()
        self.assertEqual(self.hand.moves, [])
        self.worker._dispatch(self.worker.commands.get_nowait())
        self.assertEqual(self.worker.control_mode, "off")

    def test_vision_direct_control_ignores_pre_enable_frame(self):
        source, state = self.vision()
        self.worker._set_control_mode("vision")
        self.tick()
        self.assertEqual(self.worker.controller.action()[0], 0)
        state.update(sequence=2, captured_at=time.perf_counter())
        self.tick()
        self.assertGreater(self.worker.controller.action()[0], 0)
        self.assertIsNone(self.worker.sim)
        source.set_sync.assert_not_called()

    def test_tracking_loss_freezes_target_without_disarming(self):
        _, state = self.vision()
        self.worker._set_control_mode("vision")
        state.update(sequence=2, captured_at=time.perf_counter())
        self.tick()
        previous = self.worker.controller.action()
        state["control_ready"] = False
        self.tick()
        np.testing.assert_array_equal(self.worker.controller.action(), previous)
        self.assertEqual(self.worker.control_mode, "vision")

    def test_manual_control_ignores_running_camera_and_camera_stop(self):
        self.vision()
        self.worker._set_control_mode("manual")
        self.target(0, 0.2)
        self.tick()
        self.assertEqual(self.worker.controller.status()["target"][0], 0.2)
        self.worker._stop_vision()
        self.assertEqual(self.worker.control_mode, "manual")
        self.assertEqual(self.hand.stops, 0)

    def test_simulation_stop_or_failure_does_not_disarm_direct_control(self):
        for mode in ("manual", "vision"):
            with self.subTest(mode=mode):
                self.vision()
                self.worker._set_control_mode(mode)
                sim = self.worker.sim = Mock()
                self.worker._stop_sim()
                sim.join.assert_not_called()
                self.assertEqual(self.worker.control_mode, mode)
                self.worker._handle_sim_lifecycle(sim, "sim_closed", ())
                sim.is_alive.return_value = False
                self.tick()
                self.assertIsNone(self.worker.sim)
                self.assertEqual(self.worker.control_mode, mode)

    def test_simulation_control_requires_sim_and_stops_when_it_exits(self):
        self.worker._set_control_mode("follow")
        self.assertEqual(self.worker.control_mode, "off")
        self.worker.sim = Mock()
        self.worker.sim.action.return_value = np.zeros(6)
        self.worker._set_control_mode("follow")
        self.assertEqual(self.worker.control_mode, "follow")
        self.worker._stop_sim()
        self.assertEqual(self.worker.control_mode, "off")
        self.assertEqual(self.hand.stops, 1)

    def test_holding_task_rejects_all_continuous_sources(self):
        self.worker._active_grasp = {"norms": [0.4] * 6}
        for mode in ("manual", "vision", "follow"):
            self.worker._set_control_mode(mode)
            self.worker.put(("select_source", mode))
            self.target(0, 1)
            self.tick()
            self.assertEqual(self.worker.control_mode, "off")
            self.assertEqual(self.hand.moves, [])
        self.assertTrue(self.worker.commands.empty())

    def test_failed_feedback_disarms_without_sending(self):
        self.worker._set_control_mode("manual")
        self.hand.read_currents = lambda: None
        self.tick()
        self.assertEqual(self.worker.control_mode, "off")
        self.assertEqual(self.hand.moves, [])
        self.assertEqual(self.hand.stops, 1)

    def test_stop_of_holding_task_releases_control_ownership(self):
        self.worker._active_grasp = {"norms": [0.4] * 6}
        self.worker.put("stop")
        self.worker._dispatch(self.worker.commands.get_nowait())
        self.assertIsNone(self.worker._active_grasp)
        self.worker._set_control_mode("manual")
        self.assertEqual(self.worker.control_mode, "manual")

    def test_open_can_be_cancelled_before_second_gesture_step(self):
        self.hand.pos = self.worker.cal.pose_from_norm([0.7] * 6)

        def wait_settled(**kwargs):
            self.worker.put("stop")
            return True

        self.hand.wait_settled = wait_settled
        self.worker._do_open()
        self.assertEqual(len(self.hand.moves), 1)
        # Opening the thumb may proceed; old finger-opening steps must not run.
        self.assertAlmostEqual(self.worker.cal.norm_from_pos(1, self.hand.pos[1]), 0.7, places=4)
        self.assertGreater(self.hand.stops, 0)

    def test_switch_source_stops_active_motion_and_requires_enable(self):
        self.worker._set_control_mode("manual")
        epoch = self.worker._input_epoch
        self.worker._do_sim_command(("select_source", "vision"))
        self.target(0, 1, epoch)
        self.tick()
        self.assertEqual(self.worker.control_mode, "off")
        self.assertEqual(self.worker.control_source, "vision")
        self.assertEqual(self.hand.stops, 1)
        self.assertEqual(self.hand.moves, [])


class CoreTests(unittest.TestCase):
    def test_thumb_must_clear_before_rotation_and_caps_bending(self):
        control = MotionController(slew=1)
        control.set_state([0, 0, 0, 0, 0, 0.8])
        control.set_target([0, 0, 0, 0, 1, 1])
        while control.action()[5] > THUMB_BEND_SAFE:
            previous = control.action()
            action = control._advance(0.05)
            self.assertEqual(action[4], previous[4])
            self.assertLess(action[5], previous[5])
        for _ in range(100):
            action = control._advance(0.05)
        self.assertAlmostEqual(action[4], 1)
        self.assertLessEqual(action[5], 0.9)

    def test_invalid_targets_and_feedback_fail_closed(self):
        control = MotionController()
        for target in ([0] * 5, [float("nan")] * 6, [float("inf")] * 6):
            with self.assertRaises(ValueError):
                control.set_target(target)
        cal = Calibration.load()
        hand = FakeHand(cal)
        hand.pos = [0] * 5
        with self.assertRaises(RuntimeError):
            SafeSender(cal, control).tick(hand)
        self.assertEqual(hand.moves, [])

    def test_import_and_control_with_simulator_imports_blocked(self):
        code = '''
import sys, importlib.abc, queue
class NoSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mujoco', 'glfw', 'lesson1_gestures', 'sim_sync', 'sim_bridge', 'sim_preview'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, NoSimulator())
from gui_control import HandWorker, SimSync
from calibration import Calibration
from test_direct_control import FakeHand
assert SimSync is None
w = HandWorker(Calibration.load(), queue.Queue(), queue.Queue())
w.hand = FakeHand(w.cal)
w._connected = True
w._set_control_mode('manual')
w.put(('manual_target', w._input_epoch, 0, 0.5))
w._control_tick()
assert w.control_mode == 'manual' and w.hand.moves
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
