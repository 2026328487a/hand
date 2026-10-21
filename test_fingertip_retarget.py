"""Offline geometry, optimization and fallback checks; never open CAN or cameras."""

import unittest
from unittest.mock import Mock

import numpy as np

from fingertip_retarget import (FingertipRetargeter, LiteS001Kinematics, PAIRS,
                               UPPER, palm_coordinates)
from vision_hand import DEFAULT_MAP, LandmarkMapper, features


def landmarks():
    p = np.zeros((21, 3))
    p[1:5] = ((.035, .025, 0), (.05, .035, 0), (.06, .048, 0), (.065, .061, 0))
    for base, x in ((5, .025), (9, 0), (13, -.022), (17, -.042)):
        for j in range(4):
            p[base + j] = (x, .06 + j * .026, 0)
    return p


def baseline(p):
    f = features(p, False)
    return np.array([m.norm(f) for m in DEFAULT_MAP])


class FingertipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = LiteS001Kinematics()

    def test_closed_linkages_and_calibrated_joint_limits_across_range(self):
        rng = np.random.default_rng(31)
        for action in np.vstack((np.zeros(6), UPPER, rng.random((30, 6)) * UPPER)):
            self.assertLess(self.kin.closure_error(action), .00015)
            m, d = self.kin.model, self.kin.data
            for j in np.flatnonzero(m.jnt_limited):
                value = d.qpos[m.jnt_qposadr[j]]
                self.assertGreaterEqual(value + 1e-6, m.jnt_range[j, 0])
                self.assertLessEqual(value - 1e-6, m.jnt_range[j, 1])
        self.assertTrue(self.kin.model_adjustments)
        expected = self.kin._motor_angle(0, .5)
        self.kin.fingertips(np.full(6, .5))
        self.assertAlmostEqual(self.kin.data.qpos[self.kin.active[0]], expected)
        self.assertGreater(abs(self.kin.data.qpos[1]), .1)

    def test_palm_targets_ignore_translation_rotation_scale_and_mirror(self):
        p = landmarks()
        a = .71
        rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                             [-np.sin(a), 0, np.cos(a)]])
        points, width = palm_coordinates(p)
        transformed, transformed_width = palm_coordinates(p @ rotation.T * 1.8 + [3, -2, 4])
        np.testing.assert_allclose(points / width, transformed / transformed_width, atol=1e-12)
        mirrored = p * [-1, 1, 1]
        reflected, _ = palm_coordinates(mirrored, left=True)
        np.testing.assert_allclose(points, reflected, atol=1e-12)

    def test_zero_gap_pinch_is_finite_and_hysteresis_releases(self):
        p = landmarks()
        p[8] = p[4]
        retarget = FingertipRetargeter(self.kin)
        vectors, _, _ = retarget.targets(p, baseline=baseline(p))
        self.assertTrue(np.isfinite(vectors).all())
        self.assertTrue(retarget.pinching[0])
        self.assertAlmostEqual(np.linalg.norm(vectors[0]), retarget.pinch_distance)
        width = np.linalg.norm(p[5] - p[17])
        p[8] = p[4] + [width * .32, 0, 0]
        retarget.targets(p)
        self.assertTrue(retarget.pinching[0])
        p[8] = p[4] + [width * .5, 0, 0]
        retarget.targets(p)
        self.assertFalse(retarget.pinching[0])

    def test_optimizer_reduces_task_error_and_respects_six_command_bounds(self):
        p = landmarks()
        p[8] = p[4] + [-.006, .006, .002]
        retarget = FingertipRetargeter(self.kin, budget_ms=100)
        before = baseline(p)
        after = retarget.refine(p, before)
        self.assertEqual(after.shape, (6,))
        self.assertTrue(np.all((after >= 0) & (after <= UPPER)))
        self.assertLess(retarget.status['final_cost'], retarget.status['initial_cost'])
        self.assertLess(retarget.status['pair_rmse_mm'], retarget.status['baseline_pair_rmse_mm'])
        tips_before, tips_after = self.kin.fingertips(before), self.kin.fingertips(after)
        self.assertLess(np.linalg.norm(tips_after[1] - tips_after[0]),
                        np.linalg.norm(tips_before[1] - tips_before[0]))
        retarget.reset()
        self.assertIsNone(retarget.previous)
        self.assertFalse(retarget.pinching.any())

    def test_zero_iteration_budget_keeps_baseline(self):
        p = landmarks()
        retarget = FingertipRetargeter(self.kin, budget_ms=0)
        before = np.clip(baseline(p), 0, UPPER)
        np.testing.assert_allclose(retarget.refine(p, before), before)
        self.assertEqual(retarget.status['state'], 'budget')

    def test_model_failure_falls_back_once_and_keeps_original_filter(self):
        factory = Mock(side_effect=FileNotFoundError('missing model'))
        hybrid = LandmarkMapper(kinematics_factory=factory)
        old = LandmarkMapper(retargeting='features')
        p = landmarks()
        for _ in range(3):
            np.testing.assert_allclose(hybrid.update(p, False), old.update(p, False))
        factory.assert_called_once()
        self.assertEqual(hybrid.retarget_status['state'], 'fallback')
        self.assertIn('missing model', hybrid.retarget_status['reason'])

    def test_invalid_landmarks_never_replace_last_valid_target(self):
        mapper = LandmarkMapper(retargeting='features')
        valid = mapper.update(landmarks(), False)
        for invalid in (np.full((21, 3), np.nan), np.zeros((21, 3)), np.zeros((20, 3))):
            with self.assertRaises(ValueError):
                mapper.update(invalid, False)
        np.testing.assert_allclose(mapper.miss(), valid)

    def test_solver_failure_returns_baseline_and_clears_temporal_state(self):
        mapper = LandmarkMapper(kinematics_factory=Mock(return_value=self.kin),
                                ema=1, median_window=1, deadband=0)
        p = landmarks()
        mapper.update(p, False)
        optimizer = mapper._optimizers[False]
        optimizer.refine = Mock(side_effect=np.linalg.LinAlgError('singular solve'))
        np.testing.assert_allclose(mapper.update(p, False), baseline(p))
        self.assertIsNone(optimizer.previous)
        self.assertEqual(mapper.retarget_status['state'], 'fallback')

    def test_prolonged_tracking_loss_clears_pinch_and_optimizer_history(self):
        mapper = LandmarkMapper(kinematics_factory=Mock(return_value=self.kin), hold_frames=1)
        p = landmarks()
        p[8] = p[4] + [.002, .002, 0]
        mapper.update(p, False)
        optimizer = mapper._optimizers[False]
        self.assertTrue(optimizer.pinching[0])
        self.assertIsNotNone(mapper.miss())
        self.assertIsNone(mapper.miss())
        self.assertIsNone(optimizer.previous)
        self.assertFalse(optimizer.pinching.any())

    def test_left_model_and_side_switch_reset_tracking_history(self):
        left_kin = LiteS001Kinematics(left=True)
        self.assertLess(left_kin.closure_error(UPPER), .00015)
        for channel in range(6):
            action = np.zeros(6)
            action[channel] = .7
            right = self.kin.fingertips(action)
            left = left_kin.fingertips(action)
            self.assertLess(np.max(np.abs(left - right)), .006)
        mapper = LandmarkMapper(retargeting='features', ema=.01)
        p = landmarks()
        mapper.update(p, False)
        p[8] = p[4]
        expected = LandmarkMapper(retargeting='features').update(p, True)
        np.testing.assert_allclose(mapper.update(p, True), expected)

    def test_human_handedness_does_not_select_a_different_robot(self):
        factory = Mock(return_value=self.kin)
        mapper = LandmarkMapper(kinematics_factory=factory, ema=1, median_window=1)
        p = landmarks()
        right = mapper.update(p, False)
        left = mapper.update(p * [-1, 1, 1], True)
        factory.assert_called_once_with(left=False, calibration=None)
        np.testing.assert_allclose(right, left, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
