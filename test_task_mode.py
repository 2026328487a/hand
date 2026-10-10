"""Hardware-free checks for the task-oriented grasp sequence."""

import unittest

from task_mode import (DEFAULT_CLOSE_NORM, TaskPreconditionError,
                       perform_adaptive_grasp)


class IdentityCalibration:
    @staticmethod
    def norm_from_pos(_motor, pos):
        return float(pos)

    @staticmethod
    def pose_from_norm(norms):
        return list(norms)


class FakeHand:
    def __init__(self, contacts=None, start=None, response_step=None):
        self.pos = list(start or [0.0] * 6)
        self.contacts = dict(contacts or {})
        self.response_step = response_step
        self.move_targets = []
        self.stop_calls = 0

    def read_pos(self):
        return list(self.pos)

    def read_currents(self):
        currents = [40] * 6
        for motor, (depth, current_ma) in self.contacts.items():
            if self.pos[motor] >= depth:
                currents[motor] = current_ma
        return currents

    def move(self, target, speed=255):
        self.move_targets.append(list(target))
        if self.response_step is None:
            self.pos = list(target)
            return
        for motor, desired in enumerate(target):
            delta = float(desired) - self.pos[motor]
            step = max(-self.response_step, min(self.response_step, delta))
            self.pos[motor] += step

    def stop(self):
        self.stop_calls += 1


class AdaptiveGraspTests(unittest.TestCase):
    def run_fast(self, hand, **kwargs):
        return perform_adaptive_grasp(
            hand, IdentityCalibration(), slew=10.0, hz=100.0, **kwargs)

    def test_closes_four_fingers_together_then_opposes_thumb(self):
        hand = FakeHand()
        progress = []

        result = self.run_fast(
            hand, on_progress=lambda *args: progress.append(args))

        first = next(
            target for target in hand.move_targets
            if any(target[i] > 0 for i in (1, 2, 3, 4)))
        self.assertTrue(all(first[i] > 0 for i in (1, 2, 3, 4)))
        self.assertEqual(first[0], 0.0)
        self.assertEqual(first[5], 0.0)
        self.assertAlmostEqual(result["norms"][1], DEFAULT_CLOSE_NORM)
        self.assertAlmostEqual(result["norms"][4], DEFAULT_CLOSE_NORM)
        self.assertAlmostEqual(result["norms"][5], 0.55)
        self.assertAlmostEqual(result["norms"][0], 0.65)
        self.assertEqual(result["contacts"], {})
        self.assertEqual([step[0] for step in progress], [1, 2, 3])

    def test_each_finger_pins_at_its_own_contact_depth(self):
        hand = FakeHand(contacts={1: (0.42, 320), 3: (0.61, 330)})

        result = self.run_fast(hand)

        self.assertEqual(result["contacts"], {1: 320, 3: 330})
        self.assertLessEqual(result["norms"][1], 0.52)
        self.assertLessEqual(result["norms"][3], 0.71)
        self.assertAlmostEqual(result["norms"][2], DEFAULT_CLOSE_NORM)

    def test_thumb_rotation_contact_prevents_further_thumb_bending(self):
        hand = FakeHand(contacts={5: (0.25, 340)}, response_step=0.1)

        result = self.run_fast(hand)

        self.assertEqual(result["contacts"], {5: 340})
        self.assertEqual(result["norms"][0], 0.0)
        self.assertGreaterEqual(result["norms"][5], 0.25)
        self.assertLessEqual(result["norms"][5], 0.35)

    def test_requires_an_open_start_pose(self):
        hand = FakeHand(start=[0.0, 0.2, 0.0, 0.0, 0.0, 0.0])

        with self.assertRaises(TaskPreconditionError):
            self.run_fast(hand)

        self.assertEqual(hand.move_targets, [])

    def test_emergency_cancel_interrupts_an_active_phase(self):
        hand = FakeHand()

        with self.assertRaises(InterruptedError):
            self.run_fast(
                hand, should_cancel=lambda: bool(hand.move_targets))

        self.assertGreaterEqual(len(hand.move_targets), 1)
        self.assertEqual(hand.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
