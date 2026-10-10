"""录制用于视觉映射标定的受控摄像头视频。

用法::

    ..\\.venv\\Scripts\\python.exe record_calibration_video.py --camera 0

脚本会在每个姿态开始前倒计时，并把阶段信息写入同名 ``.json`` 文件。
录制过程中按 ``q`` 可提前结束。生成的视频可以直接交给
``vision_hand.py --calibrate --video ...`` 做初始映射标定。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PHASES = (
    ("open", "五指完全伸直，拇指自然放松，不要刻意外展"),
    ("half", "四指自然半弯，保持拇指张开"),
    ("fist", "四指完全弯曲握拳，拇指保持自然位置"),
    ("thumb_open", "四指伸直，只把拇指向外张开"),
    ("thumb_across", "四指伸直，只把拇指横扫到掌心"),
)


def main() -> int:
    # Windows 中文代码页经常把 Python UTF-8 输出显示成“???”；日志和
    # 终端统一使用 UTF-8，避免采集提示乱码。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0, help="摄像头编号")
    ap.add_argument("--output", type=Path,
                    default=Path("calibration_capture.mp4"),
                    help="输出视频路径")
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="每个姿态保持时间")
    ap.add_argument("--countdown", type=float, default=3.0,
                    help="每个姿态前倒计时")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("缺少 opencv，请先运行 setup_env.cmd") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"打不开摄像头 {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (1.0 < fps < 120.0):
        fps = args.fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"无法创建视频文件 {args.output}")

    phases = []
    frame_no = 0
    aborted = False

    def draw(frame, title, detail, countdown=None):
        cv2.rectangle(frame, (0, 0), (width, 104), (0, 0, 0), -1)
        cv2.putText(frame, title, (24, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, detail, (24, 72), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 1, cv2.LINE_AA)
        if countdown is not None:
            cv2.putText(frame, f"{countdown:.1f}", (width - 150, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 230, 255), 3,
                        cv2.LINE_AA)
        cv2.putText(frame, "按 q 提前结束", (width - 220, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1,
                    cv2.LINE_AA)

    try:
        for index, (name, detail) in enumerate(PHASES, 1):
            start = frame_no
            deadline = time.perf_counter() + args.countdown
            while time.perf_counter() < deadline:
                ok, frame = cap.read()
                if not ok:
                    continue
                left = deadline - time.perf_counter()
                draw(frame, f"准备姿态 {index}/{len(PHASES)}: {name}",
                     detail, max(0.0, left))
                cv2.imshow("Calibration capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    aborted = True
                    break
            if aborted:
                break

            deadline = time.perf_counter() + args.seconds
            while time.perf_counter() < deadline:
                ok, frame = cap.read()
                if not ok:
                    continue
                draw(frame, f"录制中 {index}/{len(PHASES)}: {name}", detail)
                writer.write(frame)
                frame_no += 1
                cv2.imshow("Calibration capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    aborted = True
                    break
            phases.append({"name": name, "detail": detail,
                           "start_frame": start, "end_frame": frame_no})
            if aborted:
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    meta = args.output.with_suffix(".json")
    meta.write_text(json.dumps({
        "video": str(args.output), "camera": args.camera,
        "fps": fps, "width": width, "height": height,
        "seconds_per_phase": args.seconds, "aborted": aborted,
        "phases": phases,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"视频已保存: {args.output.resolve()}")
    print(f"阶段信息已保存: {meta.resolve()}")
    if not aborted and len(phases) == len(PHASES):
        print("下一步: ..\\.venv\\Scripts\\python.exe vision_hand.py --calibrate --video "
              f"\"{args.output}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
