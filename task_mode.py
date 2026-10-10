"""Task-oriented control primitives for the ROH-LiteS001 GUI.

The first physical task is deliberately narrow: close the four fingers
together, pin each finger independently at first contact, oppose the thumb,
and hold the presented object. It reports contact depths and motor currents;
it does not claim to identify the object or verify that it remained grasped.
"""

from __future__ import annotations

import time

from rohand import N_MOTORS


FINGERS = (1, 2, 3, 4)
THUMB_BEND = 0
THUMB_ROT = 5
DEFAULT_CONTACT_MA = 300
DEFAULT_CLOSE_NORM = 0.85
DEFAULT_THUMB_ROT_NORM = 0.55
DEFAULT_THUMB_BEND_NORM = 0.65
DEFAULT_SLEW_NORM_S = 0.5
DEFAULT_HZ = 20
APPROACH_SPEED = 80
ARRIVE_TOL_NORM = 0.03
PHASE_TIMEOUT_S = 10.0


class TaskPreconditionError(ValueError):
    """The hand is not in a safe starting pose for the task."""


def _read_norms(hand, cal):
    return [cal.norm_from_pos(i, p) for i, p in enumerate(hand.read_pos())]


def _move_phase(hand, cal, commanded, targets, motors, contact_ma, contacts,
                *, should_cancel, slew, hz, timeout, on_progress, phase,
                label):
    """Move selected channels together and pin each channel on first contact."""
    pinned = {}
    start = previous = time.monotonic()
    period = 1.0 / hz

    while True:
        loop_started = time.monotonic()
        if should_cancel is not None and should_cancel():
            hand.stop()
            raise InterruptedError("自适应抓取已取消")
        if loop_started - start > timeout:
            hand.stop()
            raise TimeoutError(f"{label}超时，已停止手部运动")

        # Sample actual feedback before taking the next small step. A channel
        # that reaches contact is held at its measured position while the
        # remaining channels continue toward their own targets.
        currents = [int(x) for x in hand.read_currents()]
        positions = hand.read_pos()
        actual = [cal.norm_from_pos(i, p) for i, p in enumerate(positions)]
        for motor in motors:
            if motor not in pinned and currents[motor] > contact_ma:
                pinned[motor] = actual[motor]
                contacts[motor] = currents[motor]
                commanded[motor] = pinned[motor]

        now = time.monotonic()
        dt = min(max(now - previous, 0.0), 0.1)
        previous = now
        max_step = slew * dt
        for motor in motors:
            if motor in pinned:
                commanded[motor] = pinned[motor]
            else:
                delta = targets[motor] - commanded[motor]
                commanded[motor] += max(-max_step, min(max_step, delta))

        hand.move(cal.pose_from_norm(commanded), speed=APPROACH_SPEED)

        done = all(
            motor in pinned
            or (abs(targets[motor] - commanded[motor]) <= 1e-6
                and abs(targets[motor] - actual[motor]) <= ARRIVE_TOL_NORM)
            for motor in motors)
        if done:
            # Take one final sample so the report describes the physical hand,
            # not only the last commanded target.
            positions = hand.read_pos()
            actual = [cal.norm_from_pos(i, p) for i, p in enumerate(positions)]
            touched = bool(pinned)
            if on_progress is not None:
                on_progress(phase, label, touched, actual, dict(contacts))
            return actual, touched

        time.sleep(max(0.0, period - (time.monotonic() - loop_started)))


def perform_adaptive_grasp(hand, cal, *, contact_ma=DEFAULT_CONTACT_MA,
                           close_norm=DEFAULT_CLOSE_NORM,
                           on_progress=None, should_cancel=None,
                           slew=DEFAULT_SLEW_NORM_S, hz=DEFAULT_HZ):
    """Close four fingers in parallel, then bring the thumb into opposition.

    Independent current monitoring lets each finger stop at its own contact
    depth while the other fingers continue. The thumb rotates only after the
    four fingers reach contact or their configured travel limit; bending is
    skipped if thumb rotation itself encounters an obstruction.
    """
    contact_ma = int(contact_ma)
    close_norm = float(close_norm)
    if not 200 <= contact_ma <= 600:
        raise ValueError("接触电流阈值需在 200–600 mA 之间")
    if not 0.25 <= close_norm <= 0.9:
        raise ValueError("四指闭合上限需在 0.25–0.90 之间")
    if slew <= 0 or hz <= 0:
        raise ValueError("动作速度和采样频率必须大于 0")

    def check_cancelled():
        if should_cancel is not None and should_cancel():
            hand.stop()
            raise InterruptedError("自适应抓取已取消")

    check_cancelled()
    norms = _read_norms(hand, cal)
    if len(norms) != N_MOTORS or max(norms) > 0.15:
        raise TaskPreconditionError("请先张开手，再把物体放到手掌前")

    commanded = list(norms)
    contacts = {}
    finger_targets = list(commanded)
    for motor in FINGERS:
        finger_targets[motor] = close_norm

    norms, _ = _move_phase(
        hand, cal, commanded, finger_targets, FINGERS, contact_ma, contacts,
        should_cancel=should_cancel, slew=slew, hz=hz,
        timeout=PHASE_TIMEOUT_S, on_progress=on_progress,
        phase=1, label="四指自适应抓取")
    commanded[:] = norms

    rot_targets = list(commanded)
    rot_targets[THUMB_ROT] = DEFAULT_THUMB_ROT_NORM
    norms, rotation_touched = _move_phase(
        hand, cal, commanded, rot_targets, (THUMB_ROT,), contact_ma, contacts,
        should_cancel=should_cancel, slew=slew, hz=hz,
        timeout=PHASE_TIMEOUT_S, on_progress=on_progress,
        phase=2, label="拇指对掌")
    commanded[:] = norms

    if not rotation_touched:
        bend_targets = list(commanded)
        bend_targets[THUMB_BEND] = DEFAULT_THUMB_BEND_NORM
        norms, _ = _move_phase(
            hand, cal, commanded, bend_targets, (THUMB_BEND,), contact_ma,
            contacts, should_cancel=should_cancel, slew=slew, hz=hz,
            timeout=PHASE_TIMEOUT_S, on_progress=on_progress,
            phase=3, label="拇指压紧")
        commanded[:] = norms

    check_cancelled()
    return {
        "contact_ma": contact_ma,
        "close_norm": close_norm,
        "norms": _read_norms(hand, cal),
        "contacts": contacts,
        "currents_at_hold_ma": [int(x) for x in hand.read_currents()],
    }
