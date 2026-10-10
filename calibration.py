"""把 calibration.json 变成可用的正反向映射。

标定测到的是**离散点位**,这个模块在点之间做线性插值,提供三套坐标之间的
互转:

    指令位置 pos   0-65535,直接发给手的量
    归一化   norm  0.0-1.0,0=张开 1=闭合。跨电机统一,写手势最顺手
    角度     deg   实际关节角,单位度。和仿真对接时用这个

为什么不用一条直线公式:实测位置→角度的非线性残差最大 3.9°(食指),
虽然不大,但做角度级对接时会体现出来。查表插值几乎不花什么代价。

用法::

    from calibration import Calibration
    cal = Calibration.load()
    cal.pos_from_norm(1, 0.5)      # 食指半闭合 → 32768
    cal.pos_from_deg(1, 140.0)     # 食指 140° → 对应指令位置
    cal.deg_from_pos(1, 32768)     # 反过来
    cal.norm_span_deg(1)           # (173.83, 100.47) 张开角/闭合角
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

from rohand import MOTORS, N_MOTORS, POS_MAX

DEFAULT_PATH = Path(__file__).with_name("calibration.json")


def _interp(x: float, xs: list, ys: list) -> float:
    """在 (xs, ys) 上线性插值。xs 必须单调递增,超界则夹到端点。"""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x) - 1
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[i], ys[i + 1]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class MotorCurve:
    motor: int
    name: str
    cmds: list          # 指令位置,递增
    degs: list          # 对应角度(可能递增也可能递减)
    open_pos: int       # 张开姿态下该电机的指令位置

    @property
    def open_deg(self) -> float:
        return self.degs[0]

    @property
    def closed_deg(self) -> float:
        return self.degs[-1]

    def deg_from_pos(self, pos: float) -> float:
        return _interp(pos, self.cmds, self.degs)

    def pos_from_deg(self, deg: float) -> float:
        """角度反查位置。

        四指和拇指弯曲的角度随位置**递减**,拇指旋转是递增 —— 插值要求 x
        单调递增,所以递减的曲线得先整体反转。
        """
        if self.degs[0] <= self.degs[-1]:
            return _interp(deg, self.degs, self.cmds)
        return _interp(deg, self.degs[::-1], self.cmds[::-1])


class Calibration:
    def __init__(self, data: dict):
        self.raw = data
        self.open_pose = data["open_pose"]
        self.curves = {}
        for name, m in data["motors"].items():
            # 只用上行数据建表。回差实测 <0.1°,不值得为方向各建一张表;
            # 下行点位留在 json 里备查。
            ups = [r for r in m["records"] if r["direction"] == "up"]
            ups.sort(key=lambda r: r["cmd"])
            self.curves[m["motor"]] = MotorCurve(
                motor=m["motor"], name=name,
                cmds=[r["cmd"] for r in ups],
                degs=[r["angle_deg"] for r in ups],
                open_pos=self.open_pose[m["motor"]],
            )
        missing = [i for i in range(N_MOTORS) if i not in self.curves]
        if missing:
            raise ValueError(f"标定数据缺电机 {missing},请先跑 calibrate.py")

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Calibration":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 不存在 —— 先跑 `python calibrate.py` 做标定")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    # ---------- 单电机 ----------

    def pos_from_norm(self, motor: int, norm: float) -> int:
        """0=张开位 1=完全闭合。张开位取标定时用的 open_pose,不一定是 0
        (拇指旋转的张开位是 728)。"""
        norm = max(0.0, min(1.0, norm))
        lo = self.curves[motor].open_pos
        return round(lo + (POS_MAX - lo) * norm)

    def norm_from_pos(self, motor: int, pos: float) -> float:
        lo = self.curves[motor].open_pos
        if POS_MAX == lo:
            return 0.0
        return max(0.0, min(1.0, (pos - lo) / (POS_MAX - lo)))

    def pos_from_deg(self, motor: int, deg: float) -> int:
        return round(self.curves[motor].pos_from_deg(deg))

    def deg_from_pos(self, motor: int, pos: float) -> float:
        return self.curves[motor].deg_from_pos(pos)

    def deg_span(self, motor: int):
        """(张开角, 闭合角)。注意四指是从大到小,拇指旋转是从小到大。"""
        c = self.curves[motor]
        return c.open_deg, c.closed_deg

    # ---------- 整手 ----------

    def pose_from_norm(self, norms) -> list:
        """6 个归一化量 → 6 个指令位置。手势库就靠这个。"""
        if len(norms) != N_MOTORS:
            raise ValueError(f"need {N_MOTORS} values, got {len(norms)}")
        return [self.pos_from_norm(i, v) for i, v in enumerate(norms)]

    def pose_from_deg(self, degs) -> list:
        if len(degs) != N_MOTORS:
            raise ValueError(f"need {N_MOTORS} values, got {len(degs)}")
        return [self.pos_from_deg(i, v) for i, v in enumerate(degs)]

    def summary(self) -> str:
        rows = ["电机          张开°    闭合°   行程°   张开位"]
        for i in range(N_MOTORS):
            c = self.curves[i]
            rows.append(f"{MOTORS[i]:11s} {c.open_deg:7.2f} {c.closed_deg:8.2f} "
                        f"{abs(c.closed_deg - c.open_deg):7.2f} {c.open_pos:7d}")
        return "\n".join(rows)


if __name__ == "__main__":
    cal = Calibration.load()
    print(cal.summary())
    print("\n往返检查(pos → deg → pos,应该回到原值):")
    for i in range(N_MOTORS):
        errs = []
        for p in range(0, POS_MAX + 1, 4096):
            back = cal.pos_from_deg(i, cal.deg_from_pos(i, p))
            errs.append(abs(back - p))
        print(f"  {MOTORS[i]:11s} 最大往返误差 {max(errs):4d} counts "
              f"({max(errs) / POS_MAX * 100:.3f}%)")
