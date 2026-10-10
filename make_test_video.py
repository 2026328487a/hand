"""用已知手势的样图合成一段测试视频。

为什么需要它:手边既没有摄像头、也没有人手视频,但 vision_hand.py 整条链路
(检测 → 映射 → 平滑 → 仿真 → 真机)需要能跑起来才敢往下做。这段视频让链路
今天就能端到端验证。

**它不能替代真实视频**:每个手势是静止画面硬切,没有连续的过渡过程,换镜头
那一瞬 MediaPipe 的跨帧跟踪会丢一下再重新锁定(链路里的丢帧保持正好也被它
测到了)。拿到真实视频后直接换掉即可。

用法::

    python make_test_video.py                     # 生成 ../out/vision_test.mp4
    python make_test_video.py --hold 2.0          # 每个手势停 2 秒
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "models"
OUT = ROOT.parent / "out" / "vision_test.mp4"

# 顺序挑过:从张开开始,逐步收拢,中间插一个拇指张开的,让六路都被扫到。
SEQUENCE = ("woman_hands", "victory", "pointing_up", "fist", "thumb_up")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hold", type=float, default=1.5, help="每个手势停留秒数")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--size", default="640x480", help="输出分辨率 WxH")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    frames_each = max(1, int(args.hold * args.fps))

    imgs = []
    for name in SEQUENCE:
        p = SRC / f"{name}.jpg"
        img = cv2.imread(str(p))
        if img is None:
            print(f"✗ 读不到 {p}")
            return 1
        # 等比缩放后居中裁切,别把手拉变形 —— 关键点角度本身对尺度免疫,但
        # 非等比拉伸会改变真实的关节角。
        scale = max(w / img.shape[1], h / img.shape[0])
        img = cv2.resize(img, (int(img.shape[1] * scale + 0.5),
                               int(img.shape[0] * scale + 0.5)))
        y0 = (img.shape[0] - h) // 2
        x0 = (img.shape[1] - w) // 2
        imgs.append((name, img[y0:y0 + h, x0:x0 + w]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
    if not writer.isOpened():
        print("✗ VideoWriter 打不开")
        return 1
    try:
        for name, img in imgs:
            frame = img.copy()
            cv2.putText(frame, name, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            for _ in range(frames_each):
                writer.write(frame)
    finally:
        writer.release()

    total = len(imgs) * frames_each
    print(f"已写出 {out}  {w}x{h} {args.fps}fps  "
          f"{total} 帧 / {total / args.fps:.1f}s")
    print("手势顺序: " + " → ".join(n for n, _ in imgs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
