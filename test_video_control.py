"""Camera lifecycle and video-follow safety tests. Never access real hardware."""

import queue
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from calibration import Calibration
from gui_control import HandWorker
from control_core import SafeSender as FollowSender
from vision_hand import VisionSource


class Capture:
    def __init__(self, frames=1):
        self.frames = frames
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, _):
        return 30.0

    def set(self, *_):
        self.index = 0
        return True

    def read(self):
        if self.index >= self.frames:
            return False, None
        self.index += 1
        return True, np.zeros((80, 120, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class VideoSourceTests(unittest.TestCase):
    def test_live_thread_is_not_ready_until_first_frame(self):
        source = VisionSource(0)
        with patch.object(source, 'is_alive', return_value=True):
            self.assertFalse(source.wait_ready(0.01))

    def test_empty_looping_file_fails_and_releases_capture(self):
        capture = Capture(frames=0)
        with patch('cv2.VideoCapture', return_value=capture), \
                patch('vision_hand.HandTracker') as tracker:
            source = VisionSource('empty.mp4', loop=True)
            source.start()
            source.join(2)
        self.assertFalse(source.is_alive())
        self.assertIsNotNone(source._err)
        self.assertTrue(capture.released)
        tracker.return_value.close.assert_called_once()

    def test_stop_during_inference_cannot_publish_a_late_target(self):
        entered, release = threading.Event(), threading.Event()
        sync = Mock()
        sync.action.return_value = np.zeros(6)
        tracker = Mock()

        def detect(*_):
            entered.set()
            release.wait(2)
            return np.zeros((21, 3)), False, np.zeros((21, 2))

        tracker.detect.side_effect = detect
        capture = Capture()
        with patch('cv2.VideoCapture', return_value=capture), \
                patch('vision_hand.HandTracker', return_value=tracker):
            source = VisionSource('test.mp4', sync)
            source.start()
            try:
                self.assertTrue(entered.wait(2))
                source.stop()
                sync.set_target.reset_mock()
            finally:
                release.set()
                source.join(2)
        # Final cleanup may freeze once more, but can never publish the detected pose.
        for call in sync.set_target.call_args_list:
            np.testing.assert_array_equal(call.args[0], np.zeros(6))
        self.assertIsNone(source.preview_frame())
        self.assertFalse(source.is_alive())

    def test_looping_video_timestamps_keep_increasing(self):
        timestamps = []
        capture = Capture()
        tracker = Mock()
        source = VisionSource('loop.mp4', realtime=False)

        def detect(_frame, timestamp):
            timestamps.append(timestamp)
            if len(timestamps) == 4:
                source.stop()
            return None

        tracker.detect.side_effect = detect
        with patch('cv2.VideoCapture', return_value=capture), \
                patch('vision_hand.HandTracker', return_value=tracker):
            source.start()
            source.join(2)
        self.assertFalse(source.is_alive())
        self.assertEqual(len(timestamps), 4)
        self.assertTrue(all(b > a for a, b in zip(timestamps, timestamps[1:])))

    def test_old_camera_frame_cannot_enable_follow(self):
        source = VisionSource(0)
        source._ready_ok = source.detected = True
        source._last_frame = source._last_detection = time.perf_counter() - 1.0
        self.assertFalse(source.snapshot()['control_ready'])
        self.assertTrue(source.snapshot()['tracking_lost'])


class VideoFollowTests(unittest.TestCase):
    def setUp(self):
        self.worker = HandWorker(Calibration.load(), queue.Queue(), queue.Queue())
        self.worker.sim = Mock()
        self.worker.sim.action.return_value = np.zeros(6)
        self.worker.sim.alive.return_value = True
        self.worker.hand = Mock()
        self.worker.hand.read_pos.return_value = self.worker.cal.pose_from_norm([0.0] * 6)
        self.worker.hand.read_currents.return_value = [0] * 6
        self.worker._connected = True
        self.worker.vision = Mock()
        self.worker.vision.snapshot.return_value = dict(
            error=None, tracking_lost=True, control_ready=False, state='missing')
        self.worker.vision.alive.return_value = True
        self.worker.vision.is_alive.return_value = True

    def test_no_hand_refuses_follow_without_sending_motion(self):
        self.worker._set_control_mode('vision')
        self.assertEqual(self.worker.control_mode, 'off')
        self.worker.hand.move.assert_not_called()

    def test_tracking_loss_holds_follow_and_keeps_last_target(self):
        self.worker.control_mode = 'vision'
        self.worker.control_source = 'vision'
        sender = self.worker.sender = Mock()
        sender.stalled = []
        sender.positions = self.worker.hand.read_pos.return_value
        self.worker._control_tick()
        self.assertEqual(self.worker.control_mode, 'vision')
        self.worker.hand.stop.assert_not_called()
        sender.tick.assert_called_once_with(self.worker.hand, should_cancel=self.worker._task_cancel.is_set)

    def test_video_stop_does_not_join_on_can_thread(self):
        self.worker.control_mode = 'vision'
        self.worker.control_source = 'vision'
        self.worker._stop_vision()
        self.worker.vision.stop.assert_called_once()
        self.worker.vision.join.assert_not_called()
        self.worker.hand.stop.assert_called_once()

    def test_startup_does_not_wait_on_can_thread(self):
        self.worker.vision = None
        source = Mock()
        with patch('gui_control.VisionSource', return_value=source):
            self.worker._start_vision(0)
        source.start.assert_called_once()
        source.wait_ready.assert_not_called()

    def test_still_closing_source_cannot_be_replaced(self):
        previous = self.worker.vision
        previous.alive.return_value = False
        with patch('gui_control.VisionSource') as factory:
            self.worker._start_vision(0)
        factory.assert_not_called()
        self.assertIs(self.worker.vision, previous)

    def test_old_source_event_cannot_stop_new_follow(self):
        self.worker.control_mode = 'vision'
        self.worker.control_source = 'vision'
        self.worker._on_vision_event(Mock(), 'vision_error', ('old error',))
        self.assertEqual(self.worker.control_mode, 'vision')
        self.worker.hand.stop.assert_not_called()
        self.assertTrue(self.worker.events.empty())

    def test_old_sim_event_cannot_close_new_simulation(self):
        current = self.worker.sim
        self.worker._handle_sim_lifecycle(Mock(), 'sim_closed', ())
        self.assertIs(self.worker.sim, current)
        self.worker.vision.stop.assert_not_called()

    def test_sim_start_is_embedded_and_does_not_wait_on_can_thread(self):
        self.worker.sim = None
        with patch('gui_control.SimSync') as factory:
            self.worker._start_sim()
        self.assertEqual(factory.call_args.kwargs['display'], 'embedded')
        factory.return_value.start.assert_called_once()
        factory.return_value.wait_ready.assert_not_called()
        self.worker.vision.set_sync.assert_not_called()
        self.worker._handle_sim_lifecycle(factory.return_value, 'sim_ready', ())
        self.worker.vision.set_sync.assert_not_called()

    def test_follow_waits_for_simulation_ready(self):
        self.worker.sim.wait_ready.return_value = False
        self.worker._set_control_mode('follow')
        self.assertEqual(self.worker.control_mode, 'off')
        self.worker.hand.move.assert_not_called()

    def test_failed_current_feedback_prevents_motion(self):
        sync = Mock()
        sync.action.return_value = np.zeros(6)
        sync.hi = np.ones(6)
        sender = FollowSender(self.worker.cal, sync, monitor_period=0)
        self.worker.hand.read_currents.side_effect = RuntimeError('CAN timeout')
        with self.assertRaisesRegex(RuntimeError, 'CAN timeout'):
            sender.tick(self.worker.hand)
        self.worker.hand.move.assert_not_called()

    def test_can_duration_is_included_in_follow_period(self):
        self._check_worker_period(0.0314, 1 / 30)

    def test_slow_can_does_not_add_another_full_wait_period(self):
        self._check_worker_period(0.040, 0.040)

    def _check_worker_period(self, can_duration, expected_period):
        # 仓库实测：电流 18.6ms + 位置 11ms + 下发 1.8ms。
        # 用虚拟时钟验证完整 worker 循环，无需接手也不依赖系统 sleep 精度。
        clock = [10.0]
        ticks = []

        def get(timeout):
            if len(ticks) >= 30:
                return '__quit__'
            clock[0] += timeout
            raise queue.Empty

        def tick():
            ticks.append(clock[0])
            clock[0] += can_duration

        self.worker.commands = Mock()
        self.worker.commands.get.side_effect = get
        with patch('gui_control.time.monotonic', side_effect=lambda: clock[0]), \
                patch.object(self.worker, '_control_tick', side_effect=tick), \
                patch.object(self.worker, '_maybe_report_state'), \
                patch.object(self.worker, '_do_disconnect'), \
                patch.object(self.worker, '_stop_sim'):
            self.worker.run()
        np.testing.assert_allclose(np.diff(ticks), expected_period, atol=1e-9)

    def test_follow_ui_reuses_feedback_without_extra_can_reads(self):
        self.worker.control_mode = 'vision'
        self.worker.control_source = 'vision'
        self.worker.sender = Mock(positions=[100] * 6, currents=[20] * 6)
        self.worker._maybe_report_state()
        self.worker.hand.read_pos.assert_not_called()
        self.worker.hand.read_currents.assert_not_called()
        self.assertEqual(self.worker.events.get_nowait(),
                         ('state', [100] * 6, [20] * 6))

    def test_speed_setting_updates_active_sender_and_next_session(self):
        self.worker.sender = Mock()
        self.worker._do_sim_command(('follow_speed', 120))
        self.assertEqual(self.worker.follow_speed, 120)
        self.assertEqual(self.worker.sender.speed, 120)


if __name__ == '__main__':
    unittest.main()
