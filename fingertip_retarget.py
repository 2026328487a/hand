"""Six-motor fingertip retargeting for LiteS001, without a viewer or CAN calls.

This adapts DexPilot's task-space idea, not its Allegro joint parameterization.
Closed-loop passive joints are solved from the local MJCF connect constraints
once, then interpolated along each calibrated motor's one-dimensional curve.
The CAD endpoint markers and encoder-to-model alignment still need physical
validation before interpreting model error as real fingertip accuracy.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent.parent / "rohand_mujoco/LiteS001/model"
TIP_BODIES = ("th_distal_link", "if_distal_link", "mf_distal_link",
              "rf_distal_link", "lf_distal_link")
HUMAN_TIPS = np.array([4, 8, 12, 16, 20])
PAIRS = np.array(list(itertools.combinations(range(5), 2)))
UPPER = np.array([1., 1., 1., 1., 1., .90])  # same thumb cap as control_core


def _unit(vector):
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError("degenerate palm geometry")
    return vector / length


def palm_coordinates(points, left=False):
    """Orthogonal, anatomical palm frame; mirrored hands yield equal targets."""
    p = np.asarray(points, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        raise ValueError("expected 21 finite 3D landmarks")
    along = _unit(p[9] - p[0])
    across = p[5] - p[17]
    across = _unit(across - np.dot(across, along) * along)
    normal = np.cross(across, along) * (-1 if left else 1)
    width = float(np.linalg.norm(p[5] - p[17]))
    if width < 1e-5:
        raise ValueError("degenerate palm width")
    return (p[HUMAN_TIPS] - p[0]) @ np.column_stack((across, along, normal)), width


class LiteS001Kinematics:
    """Calibrated commands in simulation order -> coupled fingertip positions.

    Each four-finger channel solves two passive hinges; thumb bend solves its
    three passive hinges and slider. No passive joint is an optimization DOF.
    Loading uses absolute mesh paths and never changes the process cwd.
    """

    def __init__(self, left=False, calibration=None, model_path=None, samples=129):
        import mujoco
        from calibration import Calibration
        from joint_mapping import SIM_TO_REAL

        self.mj = mujoco
        self.cal = calibration if calibration is not None else Calibration.load()
        path = Path(model_path) if model_path else MODEL_DIR / (
            "rohand_lites_left.xml" if left else "rohand_lites_right.xml")
        self.model = m = mujoco.MjModel.from_xml_path(str(path.resolve()))
        self.data = d = mujoco.MjData(m)
        self.left = bool(left)
        self.motor_order = SIM_TO_REAL
        self.grid = np.linspace(0., 1., max(33, int(samples)))
        self.tables = []
        self.max_table_closure_m = 0.
        self.model_adjustments = []
        names = ["if_proximal_link", "mf_proximal_link", "rf_proximal_link",
                 "lf_proximal_link", "th_root_link", "th_slider_connecting"]
        self.active = [int(m.jnt_qposadr[m.joint(n).id]) for n in names]
        groups = [([int(m.jnt_qposadr[m.joint(f"{prefix}_{suffix}").id])
                    for suffix in ("distal_link", "connecting_link")], [i])
                  for i, prefix in enumerate(("if", "mf", "rf", "lf"))]
        groups.extend([([], []), ([int(m.jnt_qposadr[m.joint(n).id]) for n in
                                  ("th_slider_link", "th_connecting_link",
                                   "th_distal_link", "th_proximal_link")], [4, 5])])
        for channel, (passive, eqs) in enumerate(groups):
            d.qpos[:] = m.qpos0
            addresses = [self.active[channel]] + passive
            rows = []
            for value in self.grid:
                d.qpos[self.active[channel]] = self._motor_angle(channel, value)
                if passive:
                    self._solve_closure(passive, eqs)
                    error, _ = self._closure(passive, eqs)
                    self.max_table_closure_m = max(self.max_table_closure_m,
                                                   float(np.max(np.abs(error))))
                rows.append(d.qpos[addresses].copy())
            self.tables.append((addresses, np.array(rows)))
        # The imported thumb connecting hinge has a negative-only limit, while
        # its connect constraints and positive actuator range require positive
        # angles even at 0.2 rad. Repair only this private kinematics model to
        # the calibrated closure branch; do not modify the MJCF or controller.
        thumb_joint = m.joint("th_connecting_link").id
        thumb_column = self.tables[5][0].index(int(m.jnt_qposadr[thumb_joint]))
        thumb_curve = self.tables[5][1][:, thumb_column]
        if thumb_curve.max() > m.jnt_range[thumb_joint, 1] + 1e-6:
            self.model_adjustments.append("thumb passive hinge limit aligned to calibrated closure branch")
            m.jnt_range[thumb_joint, 1] = float(thumb_curve.max()) + 1e-6
        self.tip_ids = np.array([m.body(n).id for n in TIP_BODIES])
        self.tip_local = np.array([self._endpoint(n) for n in TIP_BODIES])
        d.qpos[:] = m.qpos0
        self._forward()
        roots = np.array([d.xpos[m.body(f"{f}_proximal_link").id].copy()
                          for f in ("if", "mf", "rf", "lf")])
        along = _unit(roots[1] - d.xpos[m.body("base_link").id])
        across = roots[0] - roots[3]
        across = _unit(across - np.dot(across, along) * along)
        normal = np.cross(across, along) * (-1 if left else 1)
        self.frame = np.column_stack((across, along, normal))
        self.palm_width = float(np.linalg.norm(roots[0] - roots[3]))
        # Anatomical wrist reference: 60 mm proximal to the middle MCP.
        self.wrist = roots[1] - along * .060

    def _motor_angle(self, channel, normalized):
        motor = self.motor_order[channel]
        curve = self.cal.curves[motor]
        open_deg = self.cal.deg_from_pos(motor, curve.open_pos)
        actual_deg = self.cal.deg_from_pos(motor, self.cal.pos_from_norm(motor, normalized))
        sign = 1. if curve.closed_deg > open_deg else -1.
        return float(np.radians(sign * (actual_deg - open_deg)))

    def _forward(self):
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_comPos(self.model, self.data)

    def _closure(self, passive, eqs):
        m, d = self.model, self.data
        self._forward()
        residuals, jacobians = [], []
        for e in eqs:
            b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            anchors = m.eq_data[e]
            p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ anchors[:3]
            p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ anchors[3:6]
            j1, j2 = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            self.mj.mj_jac(m, d, j1, None, p1, b1)
            self.mj.mj_jac(m, d, j2, None, p2, b2)
            residuals.extend(p1 - p2)
            jacobians.append((j1 - j2)[:, passive])
        return np.asarray(residuals), np.concatenate(jacobians)

    def _solve_closure(self, passive, eqs):
        for _ in range(30):
            r, j = self._closure(passive, eqs)
            if np.max(np.abs(r)) < 1e-8:
                return
            step = np.linalg.solve(j.T @ j + np.eye(len(passive)) * 1e-10, -j.T @ r)
            limits = np.array([.001 if self.model.jnt_type[a] == 2 else .15
                               for a in passive])
            self.data.qpos[passive] += np.clip(step, -limits, limits)
            if np.max(np.abs(step)) < 1e-8:
                break
        r, _ = self._closure(passive, eqs)
        # Imported CAD hinge axes are slightly noncoplanar. Their least-squares
        # closure has an irreducible residual (about 0.08 mm across the range).
        if not np.isfinite(r).all() or np.max(np.abs(r)) > .00015:
            raise RuntimeError("LiteS001 linkage closure exceeds 0.15 mm")

    def _endpoint(self, name):
        """Centre of the distal CAD end, estimated from body-local mesh bounds."""
        m = self.model
        gid = int(m.body_geomadr[m.body(name).id])
        mid = int(m.geom_dataid[gid])
        start, count = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        rotation = np.zeros(9)
        self.mj.mju_quat2Mat(rotation, m.geom_quat[gid])
        verts = m.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T + m.geom_pos[gid]
        low, high = verts.min(axis=0), verts.max(axis=0)
        point = (low + high) / 2
        point[0] = low[0] + .004
        return point

    def fingertips(self, commands):
        action = np.asarray(commands, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("expected six finite motor commands")
        action = np.clip(action, 0., 1.)
        self.data.qpos[:] = self.model.qpos0
        for value, (addresses, rows) in zip(action, self.tables):
            self.data.qpos[addresses] = [np.interp(value, self.grid, rows[:, i])
                                         for i in range(len(addresses))]
        limited = self.model.jnt_limited.astype(bool)
        values = self.data.qpos[self.model.jnt_qposadr[limited]]
        ranges = self.model.jnt_range[limited]
        if np.any(values < ranges[:, 0] - 1e-6) or np.any(values > ranges[:, 1] + 1e-6):
            raise RuntimeError("coupled solution violates a model joint limit")
        self._forward()
        rotations = self.data.xmat[self.tip_ids].reshape(5, 3, 3)
        world = self.data.xpos[self.tip_ids] + np.einsum("nij,nj->ni", rotations, self.tip_local)
        return (world - self.wrist) @ self.frame

    def closure_error(self, commands):
        self.fingertips(commands)
        r, _ = self._closure(list(range(self.model.nv)), list(range(self.model.neq)))
        return float(np.max(np.abs(r)))


class FingertipRetargeter:
    """Bounded damped least squares, with prior, temporal term and pinch hysteresis."""

    def __init__(self, kinematics, *, max_iterations=12, budget_ms=20.,
                 prior_weight=.20, temporal_weight=.04, pinch_enter=.25,
                 pinch_exit=.40, pinch_distance=.014):
        self.kinematics = kinematics
        self.max_iterations = max_iterations
        self.budget_ms = budget_ms
        self.prior_weight = prior_weight
        self.temporal_weight = temporal_weight
        self.pinch_enter, self.pinch_exit = pinch_enter, pinch_exit
        self.pinch_distance = pinch_distance
        self.reset()

    def reset(self):
        self.previous = None
        self.pinching = np.zeros(4, dtype=bool)
        self.status = {"state": "waiting"}

    def targets(self, points, left=False, baseline=None):
        tips, width = palm_coordinates(points, left)
        tips *= self.kinematics.palm_width / width
        vectors = tips[PAIRS[:, 1]] - tips[PAIRS[:, 0]]
        distance = np.linalg.norm(vectors[:4], axis=1)
        relative = distance / self.kinematics.palm_width
        self.pinching[relative < self.pinch_enter] = True
        self.pinching[relative > self.pinch_exit] = False
        # Hysteresis avoids toggling at a single threshold; output is filtered.
        weights = np.where(PAIRS[:, 0] == 0, 1., .25)
        robot_tips = self.kinematics.fingertips(np.zeros(6) if baseline is None else baseline)
        for i in np.flatnonzero(self.pinching):
            direction = vectors[i] / distance[i] if distance[i] > 1e-7 else _unit(
                robot_tips[i + 1] - robot_tips[0])
            vectors[i] = direction * self.pinch_distance
            weights[i] = 20.
        return vectors, tips, weights

    def refine(self, points, baseline, left=False):
        started = time.perf_counter()
        baseline = np.asarray(baseline, dtype=float)
        if baseline.shape != (6,) or not np.isfinite(baseline).all():
            raise ValueError("invalid baseline")
        baseline = np.clip(baseline, 0., UPPER)
        vectors, tips, weights = self.targets(points, left, baseline)
        previous = baseline if self.previous is None else self.previous
        scale = self.kinematics.palm_width

        def residual(action):
            robot = self.kinematics.fingertips(action)
            diff = (robot[PAIRS[:, 1]] - robot[PAIRS[:, 0]] - vectors) / scale
            # Radial pseudo-Huber residual: large tracking errors have less influence.
            magnitude = np.linalg.norm(diff, axis=1)
            robust = np.sqrt(2. / (np.sqrt(1. + (magnitude / .25) ** 2) + 1.))
            separation = np.linalg.norm(robot[PAIRS[:, 1]] - robot[PAIRS[:, 0]], axis=1)
            return np.concatenate(((diff * np.sqrt(weights)[:, None] * robust[:, None]).ravel(),
                                   ((robot - tips) / scale * np.sqrt(.10)).ravel(),
                                   np.maximum(.012 - separation, 0.) / scale * np.sqrt(10.),
                                   (action - baseline) * np.sqrt(self.prior_weight),
                                   (action - previous) * np.sqrt(self.temporal_weight)))

        r = residual(baseline)
        initial_cost = float(r @ r)
        x, best = baseline.copy(), initial_cost
        if self.previous is not None:
            trial = np.clip(self.previous, 0., UPPER)
            trial_r = residual(trial)
            if float(trial_r @ trial_r) < best:
                x, r, best = trial, trial_r, float(trial_r @ trial_r)
        damping, iterations, budget_hit = .02, 0, False
        for iterations in range(1, self.max_iterations + 1):
            if (time.perf_counter() - started) * 1000 >= self.budget_ms:
                budget_hit = True
                break
            columns = []
            for j in range(6):
                if (time.perf_counter() - started) * 1000 >= self.budget_ms:
                    budget_hit = True
                    break
                delta = .002 if x[j] <= UPPER[j] - .002 else -.002
                perturbed = x.copy()
                perturbed[j] += delta
                columns.append((residual(perturbed) - r) / delta)
            if budget_hit:
                break
            jacobian = np.column_stack(columns)
            step = np.linalg.solve(jacobian.T @ jacobian + np.eye(6) * damping,
                                   -jacobian.T @ r)
            step = np.clip(step, -.15, .15)
            trial = np.clip(x + step, 0., UPPER)
            trial_r = residual(trial)
            cost = float(trial_r @ trial_r)
            if np.isfinite(cost) and cost < best:
                improvement = best - cost
                x, r, best = trial, trial_r, cost
                damping = max(.0001, damping * .5)
                if improvement < 1e-6 or np.max(np.abs(step)) < .0005:
                    break
            else:
                damping *= 5.
                if damping > 100.:
                    break
        if not np.isfinite(x).all():
            raise RuntimeError("non-finite fingertip solution")
        self.previous = x.copy()
        final_tips = self.kinematics.fingertips(x)
        error = final_tips[PAIRS[:, 1]] - final_tips[PAIRS[:, 0]] - vectors
        baseline_tips = self.kinematics.fingertips(baseline)
        baseline_error = baseline_tips[PAIRS[:, 1]] - baseline_tips[PAIRS[:, 0]] - vectors
        elapsed_ms = (time.perf_counter() - started) * 1000
        budget_hit = budget_hit or elapsed_ms >= self.budget_ms
        self.status = dict(state="budget" if budget_hit else "optimized",
                           solve_ms=elapsed_ms,
                           iterations=iterations, initial_cost=initial_cost, final_cost=best,
                           pair_rmse_mm=float(np.sqrt(np.mean(error ** 2)) * 1000),
                           baseline_pair_rmse_mm=float(np.sqrt(np.mean(baseline_error ** 2)) * 1000),
                           model_adjustments=list(self.kinematics.model_adjustments),
                           pinching=self.pinching.tolist())
        return x.copy()
