"""Compare feature/hybrid retargeting on a local video, without hardware/GUI.

Usage: python benchmark_retargeting.py --video 1.mp4 --frames 180
Reports modeled geometric error, NOT real fingertip error or grasp success.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fingertip_retarget import FingertipRetargeter, LiteS001Kinematics, PAIRS
from vision_hand import DEFAULT_MAP, HandTracker, LandmarkMapper, features


def benchmark(video, frames=180, robot_left=False):
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    tracker = None
    try:
        tracker = HandTracker()
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.
        indices = np.unique(np.linspace(0, max(0, count - 1), min(frames, max(1, count)), dtype=int))
        engines, rows, initialization = {}, [], {}
        old = LandmarkMapper(retargeting='features')
        filtered = LandmarkMapper(retargeting='features')
        previous_side = None
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            found = tracker.detect(frame, int(index * 1000 / fps))
            if found is None:
                old.miss()
                continue
            p, left, _ = found
            if previous_side is not None and left != previous_side:
                filtered.reset()
                for engine in engines.values():
                    engine.reset()
            previous_side = left
            if left not in engines:
                start = time.perf_counter()
                engines[left] = FingertipRetargeter(LiteS001Kinematics(left=robot_left))
                initialization[str(left)] = (time.perf_counter() - start) * 1000
            engine = engines[left]
            measured = features(p, left)
            base = np.array([m.norm(measured) for m in DEFAULT_MAP])
            after = engine.refine(p, base, left)
            # Compare the same post-mapping median/EMA/deadband on both paths.
            before_filtered = old.update(p, left)
            after_filtered = filtered.filter_action(after)
            targets, _, weights = engine.targets(p, left, base)
            def rmse(action):
                tips = engine.kinematics.fingertips(action)
                error = tips[PAIRS[:, 1]] - tips[PAIRS[:, 0]] - targets
                return float(np.sqrt(np.sum(weights[:, None] * error ** 2) / (3 * weights.sum())) * 1000)
            rows.append(dict(frame=int(index), left=bool(left), **engine.status,
                             weighted_baseline_mm=rmse(base), weighted_hybrid_mm=rmse(after),
                             filtered_baseline_mm=rmse(before_filtered),
                             filtered_hybrid_mm=rmse(after_filtered),
                             baseline_command=before_filtered.tolist(), hybrid_command=after_filtered.tolist()))
        if not rows:
            raise RuntimeError('no valid hand detections in sampled video')
        times = np.array([r['solve_ms'] for r in rows])
        summary = dict(video=str(Path(video).resolve()), sampled_frames=len(indices), detected_frames=len(rows),
                       robot_side='left' if robot_left else 'right',
                       initialization_ms=initialization, real_hardware_connected=False,
                       measurement='CAD/encoder model error, not measured real fingertip error or grasp success',
                       pinch_frames=sum(any(r['pinching']) for r in rows),
                       objective_nonincrease_frames=sum(r['final_cost'] <= r['initial_cost'] + 1e-10 for r in rows),
                       budget_frames=sum(r['state'] == 'budget' for r in rows),
                       solve_ms_median=float(np.median(times)), solve_ms_p95=float(np.percentile(times, 95)),
                       solve_ms_max=float(times.max()))
        for key in ('weighted_baseline_mm', 'weighted_hybrid_mm',
                    'filtered_baseline_mm', 'filtered_hybrid_mm'):
            summary[key + '_mean'] = float(np.mean([r[key] for r in rows]))
        # Average sampled command change, not a real-time velocity or jerk.
        for key in ('baseline_command', 'hybrid_command'):
            differences = [np.mean(np.abs(np.array(b[key]) - a[key]))
                           for a, b in zip(rows, rows[1:]) if a['left'] == b['left']]
            summary[key + '_mean_sample_change'] = float(np.mean(differences)) if differences else 0.
        return dict(summary=summary, frames=rows)
    finally:
        cap.release()
        if tracker is not None:
            tracker.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--frames', type=int, default=180)
    ap.add_argument('--left', action='store_true', help='use left robot model (human hand may be either side)')
    ap.add_argument('--output', type=Path, default=Path('test_artifacts/retargeting_comparison.json'))
    args = ap.parse_args()
    if args.frames < 1:
        ap.error('--frames must be positive')
    result = benchmark(args.video, args.frames, args.left)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
