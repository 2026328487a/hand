"""探针:在已知手势的样图上量各种候选特征,用来定 vision_hand.py 里的映射区间。

不是最终代码,是**定常数的依据**。拿到自己的视频后可以重跑一遍,看区间要不要调。

用法::

    python probe_vision.py                    # 跑 models/ 下的样图
    python probe_vision.py --dir some/dir     # 换一批图
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions,
                                           RunningMode)

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "models" / "hand_landmarker.task"

WRIST = 0
T_CMC, T_MCP, T_IP, T_TIP = 1, 2, 3, 4
I_MCP, I_PIP, I_DIP, I_TIP = 5, 6, 7, 8
M_MCP, M_PIP, M_DIP, M_TIP = 9, 10, 11, 12
R_MCP, R_PIP, R_DIP, R_TIP = 13, 14, 15, 16
P_MCP, P_PIP, P_DIP, P_TIP = 17, 18, 19, 20

FOUR = [("index", I_MCP, I_PIP, I_DIP, I_TIP),
        ("middle", M_MCP, M_PIP, M_DIP, M_TIP),
        ("ring", R_MCP, R_PIP, R_DIP, R_TIP),
        ("little", P_MCP, P_PIP, P_DIP, P_TIP)]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0.0


def angle(a, b, c) -> float:
    """在 b 处、ba 与 bc 之间的夹角(度)。"""
    v1, v2 = unit(a - b), unit(c - b)
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2), -1.0, 1.0))))


def turn(a, b, c, d) -> float:
    """相邻两段骨的转折角(度):(b-a) 到 (d-c) 需要拐多少。伸直=0。"""
    v1, v2 = unit(b - a), unit(d - c)
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2), -1.0, 1.0))))


def palm_frame(p, left: bool):
    """手掌坐标系。left 手把横轴翻过来,让左右手算出来同号。"""
    along = unit(p[M_MCP] - p[WRIST])          # 掌根 → 中指根,掌的长轴
    across = unit(p[I_MCP] - p[P_MCP])         # 小指根 → 食指根(拇指那一侧)
    if left:
        across = -across
    normal = unit(np.cross(across, along))     # 掌面法向
    return along, across, normal


def features(p, left: bool) -> dict:
    """算出所有候选特征。p 是 (21,3) 的 world landmarks(米)。"""
    f = {}
    for name, mcp, pip, dip, tip in FOUR:
        # 三段折角相加。掌骨那一段用 wrist→mcp 当参考,把 MCP 关节的屈曲也算进来。
        a0 = turn(p[WRIST], p[mcp], p[mcp], p[pip])
        a1 = turn(p[mcp], p[pip], p[pip], p[dip])
        a2 = turn(p[pip], p[dip], p[dip], p[tip])
        f[f"{name}_curl"] = a0 + a1 + a2
        f[f"{name}_a0"] = a0
        # 指尖到掌根的距离 / 掌长 —— 另一个候选,伸直时大、蜷起时小
        palm = np.linalg.norm(p[M_MCP] - p[WRIST])
        f[f"{name}_reach"] = np.linalg.norm(p[tip] - p[WRIST]) / max(palm, 1e-9)

    # 拇指弯曲:MCP + IP 两个关节的折角
    b1 = turn(p[T_CMC], p[T_MCP], p[T_MCP], p[T_IP])
    b2 = turn(p[T_MCP], p[T_IP], p[T_IP], p[T_TIP])
    f["thumb_bend"] = b1 + b2

    # 拇指对掌(旋转)的几个候选
    along, across, normal = palm_frame(p, left)
    t = p[T_MCP] - p[WRIST]                    # 掌根 → 拇指 MCP
    t_in = t - np.dot(t, normal) * normal      # 投影到掌面内
    f["rot_inplane"] = math.degrees(math.atan2(float(np.dot(t_in, along)),
                                               float(np.dot(t_in, across))))
    f["rot_outplane"] = math.degrees(math.asin(float(np.clip(
        np.dot(unit(t), normal), -1.0, 1.0))))
    # 拇指尖到小指根的距离 / 掌宽 —— 对掌时拇指扫向小指侧,这个值变小
    width = np.linalg.norm(p[I_MCP] - p[P_MCP])
    f["rot_tip2pinky"] = np.linalg.norm(p[T_TIP] - p[P_MCP]) / max(width, 1e-9)
    # 拇指尖到食指根,张开时大
    f["rot_tip2index"] = np.linalg.norm(p[T_TIP] - p[I_MCP]) / max(width, 1e-9)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT.parent / "models"))
    args = ap.parse_args()

    det = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=RunningMode.IMAGE, num_hands=1))

    rows = []
    for path in sorted(Path(args.dir).glob("*.jpg")):
        img = cv2.imread(str(path))
        if img is None:
            continue
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if not res.hand_world_landmarks:
            print(f"{path.stem:14s} 未检出")
            continue
        left = res.handedness[0][0].category_name == "Left"
        p = np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[0]])
        rows.append((path.stem, "L" if left else "R", features(p, left)))

    if not rows:
        print("没有可用样图")
        return

    keys = list(rows[0][2])
    print(f"{'特征':16s}" + "".join(f"{n[:11]:>12s}" for n, _, _ in rows))
    print(f"{'':16s}" + "".join(f"{'(' + h + ')':>12s}" for _, h, _ in rows))
    print("-" * (16 + 12 * len(rows)))
    for k in keys:
        print(f"{k:16s}" + "".join(f"{r[2][k]:12.1f}" for r in rows))


if __name__ == "__main__":
    main()
