"""Exact visual command tracking, optional avoidance and global overcurrent stop.

All hardware and camera access is replaced with fakes.
"""
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from calibration import Calibration
from control_core import MotionController, SafeSender
from joint_mapping import reorder_sim_to_real
import test_direct_control as direct_fixture
from vision_hand import DEFAULT_MAP, LandmarkMapper, features


class RawMappingTests(unittest.TestCase):
    def test_each_frame_is_exact_without_filter_history_or_deadband(self):
        mapper = LandmarkMapper(retargeting="direct")
        for raw in ([0] * 6, [1] * 6, [.9999, .5, .1, .3, .8, .9999], [0] * 6):
            np.testing.assert_array_equal(mapper.filter_action(raw), raw)
        np.testing.assert_array_equal(mapper.miss(), [0] * 6)

    def test_raw_mapping_bypasses_optimizer_and_its_collision_temporal_terms(self):
        points = np.zeros((21, 3))
        points[1:5] = ((.035, .025, 0), (.05, .035, 0), (.06, .048, 0), (.065, .061, 0))
        for base, x in ((5, .025), (9, 0), (13, -.022), (17, -.042)):
            for j in range(4):
                points[base + j] = (x, .06 + j * .026, 0)
        factory = Mock(side_effect=AssertionError("raw follow must not load a model"))
        mapper = LandmarkMapper(retargeting="direct", kinematics_factory=factory)
        for direction in (1, -1):
            p = points * [direction, 1, 1]
            left = direction == -1
            expected = [m.norm(features(p, left)) for m in DEFAULT_MAP]
            np.testing.assert_array_equal(mapper.update(p, left), expected)
        factory.assert_not_called()


class ExactSenderTests(unittest.TestCase):
    def setUp(self):
        self.cal = Calibration.load()
        self.hand = direct_fixture.FakeHand(self.cal)
        self.control = MotionController(rate_limit=False, collision_avoidance=False)
        self.sender = SafeSender(self.cal, self.control, speed=255, deadband=0,
                                 collision_avoidance=False, stop_on_stall=True,
                                 monitor_period=0, stagger_currents=True)

    def test_simultaneous_full_travel_rotation_and_bend_ignore_physical_lag(self):
        for target in ([1] * 6, [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, .9999]):
            self.control.set_target(target)
            np.testing.assert_array_equal(self.control._advance(.001), target)
            # Keep the physical feedback at a lagging, bent-thumb pose.
            self.hand.pos = self.cal.pose_from_norm([.8, .1, .1, .1, .1, .0])
            self.assertTrue(self.sender.tick(self.hand))
            expected = self.cal.pose_from_norm(reorder_sim_to_real(target))
            self.assertEqual(self.hand.moves[-1], (expected, 255))
            self.assertFalse(self.sender.waiting_thumb)
            self.assertFalse(self.control.status()["interlock"])

    def test_sub_deadband_change_is_sent_but_identical_position_is_not(self):
        self.control.set_state([.4] * 6)
        self.assertTrue(self.sender.tick(self.hand))
        first = self.hand.moves[-1][0]
        self.control.set_state([.4001] + [.4] * 5)
        self.assertTrue(self.sender.tick(self.hand))
        delta = max(abs(a - b) for a, b in zip(first, self.hand.moves[-1][0]))
        self.assertGreater(delta, 0)
        self.assertLess(delta, 200)
        self.assertFalse(self.sender.tick(self.hand))

    def test_avoidance_switch_clears_latched_rotation_and_can_restore_protection(self):
        self.control.set_collision_avoidance(True)
        self.control.set_state([0, 0, 0, 0, 0, .8])
        self.control.set_target([1] * 6)
        self.control._advance(.001)
        self.assertTrue(self.control.status()["interlock"])
        self.control.set_collision_avoidance(False)
        np.testing.assert_array_equal(self.control._advance(.001), [1] * 6)
        self.assertFalse(self.control.status()["interlock"])
        self.sender.set_collision_avoidance(True)
        self.hand.pos = self.cal.pose_from_norm([.8, 0, 0, 0, 0, 0])
        self.sender.tick(self.hand)
        self.assertEqual(self.hand.moves[-1][0][5], self.cal.pos_from_norm(5, 0))
        self.sender.set_collision_avoidance(False)
        self.sender.tick(self.hand)
        self.assertEqual(self.hand.moves[-1][0], self.sender.pose())

    def test_overcurrent_in_each_poll_group_prevents_next_command(self):
        for motor in (1, 4):
            with self.subTest(motor=motor):
                self.setUp()
                self.sender.tick(self.hand)  # Initial full feedback.
                if motor == 4:
                    self.sender.tick(self.hand)  # Advance to second group.
                self.control.set_state([1] * 6)
                self.hand.currents[motor] = 700
                count = len(self.hand.moves)
                with self.assertRaisesRegex(RuntimeError, "停止全部六路"):
                    self.sender.tick(self.hand)
                self.assertEqual(len(self.hand.moves), count)


class FastWorkerTests(unittest.TestCase):
    tick = direct_fixture.DirectControlTests.tick
    vision = direct_fixture.DirectControlTests.vision

    def setUp(self):
        direct_fixture.DirectControlTests.setUp(self)
        self.cal = self.worker.cal

    def test_raw_visual_frames_track_exactly_and_restore_manual_settings(self):
        _, state = self.vision()
        self.worker._do_sim_command(("rate_limit", True))
        self.worker._do_sim_command(("follow_speed", 120))
        self.worker._set_control_mode("vision")
        self.assertFalse(self.worker.controller.rate_limit)
        self.assertFalse(self.worker.controller.collision_avoidance)
        self.assertEqual(self.worker._control_period(), 1 / 60)
        for seq, target in enumerate(([1] * 6, [0] * 6, [1, .1, .5, .9, 1, 1]), 2):
            state.update(sequence=seq, captured_at=time.perf_counter(), norms=target)
            self.tick()
            np.testing.assert_array_equal(self.worker.controller.action(), target)
            self.assertEqual(self.hand.moves[-1],
                             (self.cal.pose_from_norm(reorder_sim_to_real(target)), 255))
        self.worker._set_control_mode("manual")
        self.assertTrue(self.worker.controller.rate_limit)
        self.assertTrue(self.worker.controller.collision_avoidance)
        self.assertEqual(self.worker.sender.speed, 120)
        self.assertEqual(self.worker._control_period(), 1 / 30)

    def test_overcurrent_pauses_all_motion_and_requires_explicit_enable(self):
        _, state = self.vision()
        self.worker._set_control_mode("vision")
        state.update(sequence=2, captured_at=time.perf_counter(), norms=[1] * 6)
        self.hand.currents[1] = 700
        count = len(self.hand.moves)
        self.tick()
        self.assertEqual(self.worker.control_mode, "off")
        self.assertIsNone(self.worker.sender)
        self.assertGreater(self.hand.stops, 0)
        self.assertEqual(len(self.hand.moves), count)
        self.hand.currents[1] = 0
        state.update(sequence=3, captured_at=time.perf_counter(), norms=[0] * 6)
        self.tick()
        self.assertEqual(len(self.hand.moves), count)

    def test_visual_mode_is_forwarded_without_changing_manual_control(self):
        self.worker._set_control_mode("manual")
        with patch("gui_control.VisionSource") as factory:
            self.worker._start_vision("fake.mp4", retargeting="direct")
        self.assertEqual(factory.call_args.kwargs["retargeting"], "direct")
        self.assertTrue(self.worker.controller.collision_avoidance)
        self.assertEqual(self.worker.sender.speed, 140)


if __name__ == "__main__":
    unittest.main()
