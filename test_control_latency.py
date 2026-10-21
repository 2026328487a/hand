"""Mock CAN timing and protection checks for direct targets and staggered reads."""

import unittest
from unittest.mock import patch

import numpy as np

from calibration import Calibration
from control_core import MotionController, SafeSender, THUMB_BEND_SAFE
from test_direct_control import FakeHand


class LatencyTests(unittest.TestCase):
    def setUp(self):
        self.cal = Calibration.load()
        self.control = MotionController(rate_limit=False)
        self.hand = FakeHand(self.cal)

    def test_direct_target_preserves_thumb_cap_and_clearance_interlock(self):
        self.control.set_target([1., 0, 0, 0, 0, 1.])
        first = self.control._advance(.001)
        self.assertEqual(first[0], 1.)
        self.assertLessEqual(first[5], .3)
        second = self.control._advance(.001)
        self.assertAlmostEqual(second[5], .9)
        self.control.set_target([1., 0, 0, 0, 1., .8])
        clearing = self.control._advance(.001)
        self.assertEqual(clearing[4], 0.)
        self.assertLessEqual(clearing[5], THUMB_BEND_SAFE)
        turned = self.control._advance(.001)
        self.assertEqual(turned[4], 1.)

    def test_staggered_current_poll_reduces_per_tick_work_and_preserves_coverage(self):
        calls, clock = [], [10.]
        read_pos = self.hand.read_pos

        def currents():
            calls.append('all')
            clock[0] += .0186
            return list(self.hand.currents)

        def current(motor):
            calls.append(motor)
            clock[0] += .0031
            return self.hand.currents[motor]

        def position():
            clock[0] += .011
            return read_pos()

        self.hand.read_currents = currents
        self.hand.read_current = current
        self.hand.read_pos = position
        sender = SafeSender(self.cal, self.control, monitor_period=0, stagger_currents=True)
        with patch('control_core.time.monotonic', side_effect=lambda: clock[0]):
            start = clock[0]
            sender.tick(self.hand)
            self.assertAlmostEqual(clock[0] - start, .0296)
            start = clock[0]
            sender.tick(self.hand)
            self.assertAlmostEqual(clock[0] - start, .0203)
            start = clock[0]
            sender.tick(self.hand)
            self.assertAlmostEqual(clock[0] - start, .0203)
        self.assertEqual(calls, ['all', 0, 1, 2, 3, 4, 5])
        self.assertTrue(sender.feedback_ready)

    def test_overcurrent_in_either_group_still_clamps_motor(self):
        sender = SafeSender(self.cal, self.control, monitor_period=0, stagger_currents=True)
        self.hand.pos = self.cal.pose_from_norm([.5] * 6)
        self.control.set_state([.8] * 6)
        sender.tick(self.hand)  # full initial sample
        self.hand.currents[1] = 700
        sender.tick(self.hand)
        self.assertIn(1, sender.stalled)
        self.hand.currents[4] = 800
        sender.tick(self.hand)
        self.assertIn(4, sender.stalled)

    def test_partial_read_failure_prevents_command(self):
        sender = SafeSender(self.cal, self.control, monitor_period=0, stagger_currents=True)
        sender.tick(self.hand)
        count = len(self.hand.moves)
        self.control.set_state([.4] * 6)
        self.hand.read_current = lambda _: float('nan')
        with self.assertRaises(RuntimeError):
            sender.tick(self.hand)
        self.assertEqual(len(self.hand.moves), count)

    def test_missing_or_nonfinite_partial_current_prevents_command(self):
        sender = SafeSender(self.cal, self.control, monitor_period=0, stagger_currents=True)
        sender.tick(self.hand)
        self.hand.read_current = lambda _: None
        with self.assertRaises((RuntimeError, TypeError)):
            sender.tick(self.hand)
        self.assertEqual(len(self.hand.moves), 1)

    def test_resync_requires_full_current_sample(self):
        sender = SafeSender(self.cal, self.control, monitor_period=0, stagger_currents=True)
        sender.tick(self.hand)
        sender.resync()
        with patch.object(self.hand, 'read_currents', wraps=self.hand.read_currents) as full:
            sender.tick(self.hand)
        full.assert_called_once()

    def test_rotation_waits_for_measured_thumb_clearance_with_direct_commands(self):
        self.hand.pos = self.cal.pose_from_norm([.8, 0, 0, 0, 0, 0])
        self.control.set_state([0, 0, 0, 0, 1, .8])
        sender = SafeSender(self.cal, self.control, monitor_period=0, stagger_currents=True)
        out = sender._guard(self.hand, None)
        self.assertEqual(out[5], self.hand.pos[5])
        self.assertEqual(out[0], self.cal.pos_from_norm(0, 0))
        self.hand.pos[0] = self.cal.pos_from_norm(0, .1)
        out = sender._guard(self.hand, None)
        self.assertEqual(out[5], self.cal.pos_from_norm(5, 1))
        self.assertEqual(out[0], self.cal.pos_from_norm(0, 0))


if __name__ == '__main__':
    unittest.main()
