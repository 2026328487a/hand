"""无硬件回归测试:仿真映射、限速和真机跟随保护。"""

from __future__ import annotations

import unittest

import numpy as np

from calibration import Calibration
from gestures import GESTURES
from sim_sync import (FINGER_LAG_COUNTS, SimSync, FollowSender,
                      reorder_real_to_sim, reorder_sim_to_real)


class FakeHand:
    def __init__(self, cal):
        self.pos = cal.pose_from_norm([0.0] * 6)
        self.currents = [0] * 6
        self.moves = []

    def read_currents(self):
        return list(self.currents)

    def read_pos(self):
        return list(self.pos)

    def move(self, pose, speed=255):
        self.moves.append((list(pose), speed))
        self.pos = list(pose)


class SimSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = Calibration.load()

    def test_channel_mapping_is_inverse(self):
        values = [0, 1, 2, 3, 4, 5]
        self.assertEqual(reorder_sim_to_real(reorder_real_to_sim(values)), values)

    def test_slew_limits_each_step(self):
        sync = SimSync(slew=1.0)
        sync.lo = np.zeros(6)
        sync.hi = np.ones(6)
        sync.span = np.ones(6)
        sync.set_target([1.0] * 6)
        action = sync._advance(0.1)
        np.testing.assert_allclose(action, [0.1] * 6)

    def test_thumb_is_capped_while_fingers_lag(self):
        sync = SimSync()
        sync.hi = np.ones(6) * 1.57
        hand = FakeHand(self.cal)
        sender = FollowSender(self.cal, sync, monitor_period=0)
        target = [0.0, 1.0, 0.0, 0.0, 0.0, 0.8]
        hand.pos = self.cal.pose_from_norm([0.0] * 6)
        out = sender._guard(
            hand, self.cal.pose_from_norm(reorder_sim_to_real(target)))
        self.assertTrue(sender.waiting_thumb)
        self.assertLessEqual(out[0], self.cal.pos_from_norm(0, 0.30))

    def test_stalled_motor_is_pinned_then_released(self):
        sync = SimSync()
        sync.hi = np.ones(6) * 1.57
        hand = FakeHand(self.cal)
        hand.pos = self.cal.pose_from_norm([0.5] * 6)
        hand.currents[1] = 600
        sender = FollowSender(self.cal, sync, monitor_period=0)
        target = self.cal.pose_from_norm(
            reorder_sim_to_real([1.0, 1.0, 1.0, 1.0, 1.0, 0.8]))
        out = sender._guard(hand, target)
        self.assertIn(1, sender.stalled)
        self.assertEqual(out[1], hand.pos[1])

        hand.currents = [0] * 6
        sender._guard(hand, self.cal.pose_from_norm([0.0] * 6))
        self.assertNotIn(1, sender.stalled)

    def test_contact_gesture_marks_unsupported_contact_stop(self):
        sync = SimSync()
        self.assertTrue(sync.play_gesture(GESTURES["pinch"]))
        self.assertTrue(sync.busy())

    def test_target_is_sampled_after_blocking_feedback(self):
        sync = SimSync()
        sync.hi = np.ones(6)
        hand = FakeHand(self.cal)
        sender = FollowSender(self.cal, sync, monitor_period=0)

        def read_currents():
            # 模拟读取 CAN 时仿真线程仍在推进。
            sync.set_state([0.3, 0, 0, 0, 0, 0])
            return [0] * 6

        hand.read_currents = read_currents
        sender.tick(hand)
        self.assertEqual(hand.moves[-1][0], sender.pose())

    def test_speed_change_is_sent_even_if_pose_does_not_change(self):
        sync = SimSync()
        sync.hi = np.ones(6)
        hand = FakeHand(self.cal)
        sender = FollowSender(self.cal, sync, monitor_period=0, speed=140)
        self.assertTrue(sender.tick(hand))
        self.assertFalse(sender.tick(hand))
        sender.speed = 120
        self.assertTrue(sender.tick(hand))
        self.assertEqual(hand.moves[-1][1], 120)

    def test_normalized_mapping_has_no_large_steps(self):
        sync = SimSync()
        sync.hi = np.ones(6)
        sender = FollowSender(self.cal, sync)
        poses = []
        for value in np.linspace(0, 1, 101):
            sync.set_state([value] * 6)
            poses.append(sender.pose())
        increments = np.diff(poses, axis=0)
        self.assertTrue(np.all(increments >= 0))
        # 1% 输入增量只产生约 655 counts；整数舍入误差至多 1 count。
        self.assertLessEqual(np.max(np.ptp(increments, axis=0)), 1)


if __name__ == "__main__":
    unittest.main()
