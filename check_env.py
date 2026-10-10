"""Verify the GUI environment and local assets without opening a hardware bus.

Run with --viewer to also open and automatically close the MuJoCo viewer.
Missing PCANBasic is reported separately; it does not block offline use.
"""

from __future__ import annotations

import argparse
import ctypes
from importlib.metadata import version
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def check_python_and_tk():
    import platform
    import tkinter as tk

    expected = (ROOT / '.python-version').read_text().strip()
    if platform.python_version() != expected or sys.maxsize <= 2**32:
        raise RuntimeError(f'Expected 64-bit Python {expected}; got {sys.version}')
    root = tk.Tk()
    root.withdraw()
    try:
        print(f'  Python {platform.python_version()}, Tcl/Tk {root.tk.call("info", "patchlevel")}')
    finally:
        root.destroy()


def check_gui():
    import gui_control

    for name, error in (('simulation', gui_control.SIM_IMPORT_ERROR),
                        ('vision', gui_control.VISION_IMPORT_ERROR)):
        if error is not None:
            print(f'  Optional {name} integration unavailable: {error}')
    gui_control.Calibration.load()
    for name in ('ohand-serial-sdkpy', 'python-can', 'pyserial', 'numpy',
                 'mujoco', 'mediapipe', 'opencv-contrib-python', 'cv2-enumerate-cameras', 'pillow'):
        print(f'  {name}=={version(name)}')


def check_simulation():
    import mujoco
    import numpy as np
    # Importing sim_bridge adds the parent simulation workspace to sys.path.
    import sim_bridge  # noqa: F401
    from lesson1_gestures import load

    for left in (False, True):
        model, data = load(left=left)
        for _ in range(100):
            mujoco.mj_step(model, data)
        if model.nu != 6 or not np.isfinite(data.qpos).all():
            raise RuntimeError('Invalid simulation state')
        print(f'  {"Left" if left else "Right"} hand: 6 actuators, 100 steps OK')


def check_vision():
    import cv2
    from vision_hand import DEFAULT_MODEL, HandTracker

    frame = cv2.imread(str(DEFAULT_MODEL.parent / 'victory.jpg'))
    if frame is None:
        raise FileNotFoundError('Missing or unreadable models/victory.jpg')
    tracker = HandTracker()
    try:
        result = tracker.detect(frame, 0)
        if result is None or result[0].shape != (21, 3):
            raise RuntimeError('Expected 21 hand landmarks in models/victory.jpg')
        print('  Local hand_landmarker.task: 21 landmarks detected')
    finally:
        tracker.close()


def check_sample_video():
    import cv2

    capture = cv2.VideoCapture(str(ROOT / '1.mp4'))
    try:
        success, _ = capture.read()
        if not success:
            raise RuntimeError('Cannot decode real/1.mp4')
        print('  real/1.mp4: video decoding OK')
    finally:
        capture.release()


def check_viewer():
    from sim_sync import SimSync

    events = []
    sim = SimSync(on_event=lambda *args: events.append(args))
    sim.start()
    try:
        if not sim.wait_ready(20):
            raise RuntimeError(f'Viewer startup failed: {sim._err}; events={events}')
        time.sleep(2)
        if not sim.is_alive() or any(event[0] == 'sim_error' for event in events):
            raise RuntimeError(f'Viewer stopped unexpectedly: {events}')
    finally:
        sim.stop()
        sim.join(timeout=8)
    if sim.is_alive():
        raise RuntimeError('Viewer did not shut down')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--core-only', action='store_true',
                        help='skip optional simulation and vision asset checks')
    args = parser.parse_args()
    print(f'Interpreter: {sys.executable}', flush=True)
    checks = [('Python / Tk', check_python_and_tk), ('GUI / SDK / calibration', check_gui),
              ('Sample video', check_sample_video)]
    if not args.core_only:
        checks.extend([('MuJoCo models', check_simulation),
                       ('MediaPipe / OpenCV', check_vision)])
    if args.viewer:
        checks.append(('MuJoCo viewer', check_viewer))
    failures = 0
    for label, check in checks:
        try:
            check()
        except Exception:
            failures += 1
            print(f'[FAIL] {label}', flush=True)
            traceback.print_exc()
        else:
            print(f'[OK] {label}', flush=True)
    try:
        ctypes.WinDLL('PCANBasic.dll')
    except (OSError, AttributeError):
        print('[INFO] PCANBasic.dll unavailable: GUI, simulation and video remain usable; '
              'install the PEAK driver / PCAN-Basic before connecting a physical hand.')
    else:
        print('[INFO] PCANBasic.dll loads. Hardware connection was not tested.')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
