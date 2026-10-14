"""Joint order and calibration adapters; no simulator or model dependency."""

import numpy as np
from calibration import Calibration
from rohand import N_MOTORS

# 仿真 DOF index → 真机电机 index。
# 仿真的 thumb_root 是**旋转**、thumb 是**弯曲**,别被名字骗了。
SIM_TO_REAL = [
    1,   # index_finger  → index
    2,   # middle_finger → middle
    3,   # ring_finger   → ring
    4,   # little_finger → little
    5,   # thumb_root    → thumb_rot
    0,   # thumb         → thumb_bend
]
REAL_TO_SIM = [SIM_TO_REAL.index(i) for i in range(N_MOTORS)]


def reorder_sim_to_real(vec):
    """长度 6 的仿真序向量 → 真机序。"""
    out = [0.0] * N_MOTORS
    for sim_i, real_i in enumerate(SIM_TO_REAL):
        out[real_i] = vec[sim_i]
    return out


def pose_normalized(cal: Calibration, action) -> list:
    """仿真归一化 action(0..1,仿真序) → 真机指令位置。"""
    return cal.pose_from_norm(reorder_sim_to_real(action))


def pose_by_angle(cal: Calibration, action, hi) -> list:
    """按实际关节角对齐。action 归一化,hi 是各执行器 ctrlrange 上限(rad)。

    仿真角是相对伸直的弯曲量,真机报绝对角。四指和拇指弯曲的真机角随弯曲
    **减小**,拇指旋转**增大**,所以方向由标定里的 open/closed 决定,不能写死。
    """
    sim_rad = np.asarray(action) * np.asarray(hi)
    sim_deg = np.degrees(sim_rad)
    degs = [0.0] * N_MOTORS
    for sim_i, real_i in enumerate(SIM_TO_REAL):
        open_deg, closed_deg = cal.deg_span(real_i)
        sign = 1.0 if closed_deg > open_deg else -1.0   # 弯曲时角度的走向
        degs[real_i] = open_deg + sign * sim_deg[sim_i]
    return cal.pose_from_deg(degs)


def build_pose(cal, action, hi, mode: str) -> list:
    return (pose_by_angle(cal, action, hi) if mode == "angle"
            else pose_normalized(cal, action))



def reorder_real_to_sim(vec):
    """Motor order -> control order (index, middle, ring, little, rotation, bend)."""
    return [float(vec[i]) for i in SIM_TO_REAL]
