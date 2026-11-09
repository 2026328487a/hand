"""Optional thumb clearance and finger sequencing, separate from current protection.

Both planners use control order: index, middle, ring, little, thumb rotation,
thumb bend. Disable this policy to preserve all six mapped target commands.
"""
import numpy as np

from gestures import THUMB_BEND_SAFE, THUMB_ROT_MOVE
from joint_mapping import SIM_TO_REAL

# Measured LiteS001 thresholds: bend .95 stalls against its own limit;
# bend .45 already interferes with closing fingers, while .30 clears them.
THUMB_BEND_MAX = 0.90
FINGER_GATE_TOL = 0.05
FINGER_LAG_COUNTS = 2000
THUMB_BEND_WHILE_MOVING = 0.30
SIM_TR, SIM_TB = 4, 5


class CommandAvoidance:
    """Retract before rotation; let fingers close before bending the thumb."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.interlock = False

    def apply(self, target, current):
        out = target.copy()
        gap = abs(target[SIM_TR] - current[SIM_TR])
        if self.interlock:
            if gap <= THUMB_ROT_MOVE * 0.5:
                self.interlock = False
        elif current[SIM_TB] > THUMB_BEND_SAFE and gap > THUMB_ROT_MOVE:
            self.interlock = True
        if self.interlock:
            out[SIM_TB] = 0.0
            if current[SIM_TB] > THUMB_BEND_SAFE:
                out[SIM_TR] = current[SIM_TR]
        if float(np.max(target[:4] - current[:4])) > FINGER_GATE_TOL:
            out[SIM_TB] = min(out[SIM_TB], THUMB_BEND_WHILE_MOVING)
        out[SIM_TB] = min(out[SIM_TB], THUMB_BEND_MAX)
        return out


class FeedbackAvoidance:
    """Apply the same clearance rule using measured physical positions."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.waiting_thumb = False
        self.rotation_clearance = False

    def apply(self, cal, want, positions, out):
        out = list(out)
        tb, tr = SIM_TO_REAL[SIM_TB], SIM_TO_REAL[SIM_TR]
        lag = max(want[r] - positions[r] for r in SIM_TO_REAL[:4])
        self.waiting_thumb = lag > FINGER_LAG_COUNTS
        if self.waiting_thumb:
            out[tb] = min(out[tb], cal.pos_from_norm(tb, THUMB_BEND_WHILE_MOVING))
        gap = abs(cal.norm_from_pos(tr, want[tr]) - cal.norm_from_pos(tr, positions[tr]))
        bend = cal.norm_from_pos(tb, positions[tb])
        if self.rotation_clearance:
            if gap <= THUMB_ROT_MOVE * 0.5:
                self.rotation_clearance = False
        elif bend > THUMB_BEND_SAFE and gap > THUMB_ROT_MOVE:
            self.rotation_clearance = True
        if self.rotation_clearance:
            out[tb] = cal.pos_from_norm(tb, 0.0)
            if bend > THUMB_BEND_SAFE:
                out[tr] = positions[tr]
        return out
