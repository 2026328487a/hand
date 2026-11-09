"""Simulator-independent continuous motion and feedback protection.

Control order: index, middle, ring, little, thumb rotation, thumb bend.
The caller advances MotionController with wall time; only the CAN owner calls
SafeSender.tick. SimSync reuses the same planner for simulation/CLI operation.
"""
from __future__ import annotations
import threading
import time
import numpy as np
from calibration import Calibration
from gestures import resolve
from collision_avoidance import (CommandAvoidance, FeedbackAvoidance,
                                 THUMB_BEND_SAFE, THUMB_ROT_MOVE, THUMB_BEND_MAX,
                                 FINGER_GATE_TOL, FINGER_LAG_COUNTS,
                                 THUMB_BEND_WHILE_MOVING)
from rohand import MOTOR_CN, N_MOTORS, ROHand
from joint_mapping import (SIM_TO_REAL, build_pose, reorder_real_to_sim,
                           reorder_sim_to_real)

# 仿真侧 DOF 的中文名,顺序同 lesson1_gestures.DOF。
# 注意 thumb_root 是**旋转**、thumb 是**弯曲**,名字反直觉。
DOF_CN = ("食指", "中指", "无名指", "小指", "拇指旋转", "拇指弯曲")
SIM_IX, SIM_MD, SIM_RG, SIM_LT, SIM_TR, SIM_TB = range(6)

SCRIPT_TOL = 0.02        # 手势脚本判"这一步到位"的归一化容差
DEFAULT_SLEW = 1.2       # 归一化/秒。满行程约 0.83s,和手势库 speed=160 相当
DEFAULT_HZ = 30          # 真机下发频率。move() 往返 1.8ms,余量很大
DEFAULT_SPEED = 140      # 电机速度上限，独立于仿真目标的 slew
DEADBAND = 200           # counts。小于此值不重发,省掉无意义的总线往返
CURRENT_WARN_MA = 400    # 同 gestures.py:正常运动 <180mA,顶限位/夹住明显更高
STALL_MA = 450           # 判「顶住了」的电流。真机实测自由运动峰值 145mA,
                         # 出厂单路上限 1199mA,450 两头都有余量
STALL_RELEASE = 1500     # counts。目标要退回这么多才解除钉住,避免在阈值上抖


class MotionController:
    """Thread-safe target mailbox and motion planner; never opens hardware."""

    def __init__(self, slew=DEFAULT_SLEW, *, rate_limit=True, collision_avoidance=True):
        self.slew = slew
        self.rate_limit = bool(rate_limit)
        self.collision_avoidance = bool(collision_avoidance)
        self._avoidance = CommandAvoidance()
        self._lock = threading.Lock()
        self._target = np.zeros(N_MOTORS)
        self._ctrl = np.zeros(N_MOTORS)
        self._script = []
        self._dwell = 0.0
        self._external = None

    @staticmethod
    def _vector(values):
        values = np.asarray(values, dtype=float)
        if values.shape != (N_MOTORS,) or not np.isfinite(values).all():
            raise ValueError("目标必须是 6 路有限数值")
        return np.clip(values, 0.0, 1.0)

    def _advance(self, dt: float) -> np.ndarray:
        """算出本步的归一化 ctrl。返回长度 6 的数组(仿真序)。"""
        with self._lock:
            if self._external is not None:
                # reverse:真机是主。真机报上来的是它的**实际位置**,本身已经
                # 受机械速度限制,不需要再限速,直接顶上去。
                self._ctrl = np.clip(self._external, 0.0, 1.0)
                self._target = self._ctrl.copy()
                return self._ctrl.copy()

            if self._script:
                tgt = self._script[0][0]
                if np.max(np.abs(self._ctrl - tgt)) <= SCRIPT_TOL:
                    self._dwell += dt
                    if self._dwell >= self._script[0][1]:
                        self._script.pop(0)
                        self._dwell = 0.0
                else:
                    self._dwell = 0.0
                if self._script:
                    self._target = self._script[0][0].copy()

            cur = self._ctrl
            eff = self._target.copy()

            if self.collision_avoidance:
                eff = self._avoidance.apply(eff, cur)

            if self.rate_limit:
                step = self.slew * dt
                self._ctrl = np.clip(cur + np.clip(eff - cur, -step, step), 0.0, 1.0)
            else:
                self._ctrl = eff.copy()
            return self._ctrl.copy()

    # ---------------- 外部接口(线程安全) ----------------

    def set_rate_limit(self, enabled):
        """Toggle interpolation independently of the optional avoidance policy."""
        with self._lock:
            self.rate_limit = bool(enabled)

    def set_collision_avoidance(self, enabled):
        """Toggle optional pose changes independently of interpolation/current protection."""
        with self._lock:
            if self.collision_avoidance == bool(enabled):
                return
            self.collision_avoidance = bool(enabled)
            self._avoidance.reset()

    def set_target(self, norms_sim):
        """设定目标姿态(归一化,仿真序)。会打断正在播的手势。"""
        v = self._vector(norms_sim)
        with self._lock:
            self._script.clear()
            self._dwell = 0.0
            self._target = v

    def set_one(self, i: int, value: float):
        """只改一路。滑杆走这个。"""
        if i not in range(N_MOTORS) or not np.isfinite(value):
            raise ValueError("关节编号或目标无效")
        with self._lock:
            self._script.clear()
            self._dwell = 0.0
            self._target = self._target.copy()
            self._target[i] = min(1.0, max(0.0, float(value)))

    def set_state(self, norms_sim):
        """**硬置**当前状态,不经限速。

        用于使能跟随前的对齐:真机此刻的姿态和仿真里的不一定一样,不先对齐就
        使能,第一帧就是一个大跳变 —— 六路电机同时全速冲过去。
        """
        v = self._vector(norms_sim)
        with self._lock:
            self._script.clear()
            self._dwell = 0.0
            self._avoidance.reset()
            self._ctrl = v.copy()
            self._target = v.copy()

    def push_external(self, norms_sim):
        """reverse 模式:真机报上来的姿态。传 None 退出 reverse。"""
        with self._lock:
            if norms_sim is None:
                # 退出时把目标钉在当前位置,免得切回 follow 时朝旧目标窜过去
                self._external = None
                self._target = self._ctrl.copy()
            else:
                self._external = np.asarray(norms_sim, dtype=float)

    def play_gesture(self, gesture):
        """播一个 gestures.py 的分步手势 —— 仿真先演,真机跟着走。

        手势里的 `stop_on_current`(碰到就停)在这里**不生效**:接触停止要靠
        分段逼近 + 每段读电流(见 ROHand.approach),而跟随模式是仿真定节奏、
        真机只管跟。空捏类手势照样会顶到限位,由 SafeSender 的逐路电流保护兜底。

        返回:该手势是否有被忽略掉的「碰到就停」步骤。
        """
        with self._lock:
            start = reorder_sim_to_real(self._ctrl)
        script, skipped = [], False
        norms = list(start)
        for s in gesture.steps:
            norms = resolve(s, norms)
            skipped = skipped or bool(s.stop_on_current)
            # dwell 至少给 0.15s,否则脚本判到位后立刻跳下一步,看不出分步
            script.append((np.asarray(reorder_real_to_sim(norms)),
                           max(s.dwell, 0.15)))
        with self._lock:
            self._script = script
            self._dwell = 0.0
            if script:
                self._target = script[0][0].copy()
        return skipped

    def busy(self) -> bool:
        """手势脚本是否还在播。"""
        with self._lock:
            return bool(self._script)

    def action(self) -> np.ndarray:
        """当前实际 ctrl(归一化,仿真序)。真机跟随的就是这个。"""
        with self._lock:
            return self._ctrl.copy()

    def status(self) -> dict:
        with self._lock:
            return {
                "action": self._ctrl.copy(),
                "target": self._target.copy(),
                "interlock": self._avoidance.interlock,
                "collision_avoidance": self.collision_avoidance,
                "script": len(self._script),
                "external": self._external is not None,
                "rate_limit": self.rate_limit,
            }


class SafeSender:
    """把连续控制器的 action 下发真机,带死区 + 逐路「顶住就钉住」保护。

    GUI 的 worker 线程和命令行都用它,保证两边行为一致。**只能被独占 CAN 的
    那条线程调用** —— 它自己不加锁。

    为什么必须有过流保护(真机实测):视频里人手握拳那一帧会被映射成六路全满,
    而 ROH LiteS001 的拇指对掌到 1.0 再弯到 1.0 时,落点正好压在食指上 ——
    实测食指只能走到 0.53,电机顶在那里拉 **1174mA**(出厂上限 1199mA),拇指
    弯曲同时 1171mA。同样的食指指令在拇指让开时是 1.00 / 87mA,差 13 倍。
    gestures.py 的 fist 靠 thumb_bend=0.75 + 分步避开了,连续跟随没这层保护。

    保护做法是**逐路**的:哪一路顶住了,就把那一路的下发位置钉在它当前的实际
    位置上,不再往前顶;其余五路照常跟随。目标退回来(比钉住点松 STALL_RELEASE
    以上)就自动解除。这样抓东西时也是对的 —— 停在物体上、不再加力,正是
    ROHand.approach 的「碰到就停」推广到连续跟随。
    """

    def __init__(self, cal: Calibration, sync: MotionController,
                 mode: str = "normalized", speed: int = DEFAULT_SPEED,
                 deadband: int = DEADBAND, stall_ma: int = STALL_MA,
                 monitor_period: float = 0.04, on_warn=None,
                 stagger_currents: bool = False, collision_avoidance: bool = True,
                 stop_on_stall: bool = False):
        self.cal = cal
        self.sync = sync
        self.collision_avoidance = bool(collision_avoidance)
        self.stop_on_stall = bool(stop_on_stall)
        self._avoidance = FeedbackAvoidance()
        self.mode = mode
        # slew 限制的是离散目标变化量，不是电机速度；满速追短目标可能走停。
        self.speed = speed
        self.deadband = deadband
        self.stall_ma = stall_ma
        # 六路电流约 18.6ms，位置约 11ms；调用方须把 CAN 耗时计入周期。
        self.monitor_period = monitor_period
        self.stagger_currents = bool(stagger_currents)
        self._current_group = None
        self.feedback_ready = False
        self.on_warn = on_warn or (lambda msg: None)
        self.currents = [0] * N_MOTORS
        self.positions = [0] * N_MOTORS  # 最近一次读到的真机位置
        self.stalled: list = []          # 当前被钉住的电机 index,给界面看
        self.waiting_thumb = False       # 拇指是否正在等四指跟上
        self._clamp: list = [None] * N_MOTORS
        self._last_pose = None
        self._last_speed = None
        self._cur_t = 0.0
        # 逐路节流。**之前是一个全局 _warn_t**,结果拇指弯曲的告警把食指的挤
        # 掉了 —— 实测食指顶到 1167mA 却一条告警都没发出来。
        self._warn_t = [0.0] * N_MOTORS

    def pose(self):
        return build_pose(self.cal, self.sync.action(), getattr(self.sync, "hi", None), self.mode)

    def tick(self, hand: ROHand, should_cancel=None) -> bool:
        """下发一帧。返回 True 表示真的发了(没被死区挡掉)。"""
        # 先读反馈再取最新目标，避免把 CAN 读取前的旧仿真帧下发出去。
        pose = self._guard(hand, None)
        if should_cancel is not None and should_cancel():
            return False
        sent = False
        if (self._last_pose is None or self.speed != self._last_speed or
                max(abs(a - b) for a, b in zip(pose, self._last_pose))
                > self.deadband):
            hand.move(pose, speed=self.speed)
            self._last_pose = pose
            self._last_speed = self.speed
            sent = True
        return sent

    def _guard(self, hand: ROHand, want) -> list:
        """采电流,更新钉住状态,返回实际该下发的位置。"""
        now = time.monotonic()
        if now - self._cur_t >= self.monitor_period:
            self._cur_t = now
            # 没有新反馈就不能继续按旧电流/位置下发；交由调用方停止跟随。
            sampled = list(range(N_MOTORS))
            if self.stagger_currents and self._current_group is not None:
                sampled = list(range(self._current_group * 3, self._current_group * 3 + 3))
                cur = list(self.currents)
                for i in sampled:
                    cur[i] = hand.read_current(i)
            else:
                cur = hand.read_currents()
            pos = hand.read_pos()
            if (cur is None or pos is None or len(cur) != N_MOTORS
                    or len(pos) != N_MOTORS
                    or not np.isfinite(cur).all() or not np.isfinite(pos).all()):
                raise RuntimeError("真机反馈缺失或无效，已禁止继续下发")
            if cur is not None and pos is not None:
                self.currents = cur
                self.positions = pos
                self.feedback_ready = True
                if self.stagger_currents:
                    self._current_group = 0 if self._current_group is None else 1 - self._current_group
                hot = [i for i in sampled if cur[i] > self.stall_ma]
                if hot and self.stop_on_stall:
                    names = "、".join(f"{MOTOR_CN[i]} {cur[i]}mA" for i in hot)
                    raise RuntimeError(f"原样跟随过流（{names}），已停止全部六路；请退回姿态后重新启用")
                for i in hot:
                    new = (pos[i] if self._clamp[i] is None
                           else min(self._clamp[i], pos[i]))
                    first = self._clamp[i] is None
                    self._clamp[i] = new
                    if first or now - self._warn_t[i] > 2.0:
                        self._warn_t[i] = now
                        self.on_warn(
                            f"{MOTOR_CN[i]} 顶住了({cur[i]}mA)，"
                            f"已钉在 pos={new}，其余各路照常跟随")

        if want is None:
            want = self.pose()
        out = list(want)
        for i, c in enumerate(self._clamp):
            if c is None:
                continue
            if want[i] < c - STALL_RELEASE:
                self._clamp[i] = None       # 目标退回来了,松开
            else:
                out[i] = c
        self.stalled = [i for i, c in enumerate(self._clamp) if c is not None]

        if self.collision_avoidance:
            out = self._avoidance.apply(self.cal, want, self.positions, out)
        self.waiting_thumb = self._avoidance.waiting_thumb
        return out

    def set_collision_avoidance(self, enabled):
        if self.collision_avoidance == bool(enabled):
            return
        self.collision_avoidance = bool(enabled)
        self._avoidance.reset()
        self.waiting_thumb = False
        self._last_pose = None

    def resync(self):
        """强制下一帧无视死区,并清掉钉住状态。切换映射模式后要调。"""
        self._last_pose = None
        self._clamp = [None] * N_MOTORS
        self.stalled = []
        self.waiting_thumb = False
        self._current_group = None
        self.feedback_ready = False
        self._avoidance.reset()
        self._cur_t = 0.0


def align_control_to_hand(sync: MotionController, cal: Calibration, hand: ROHand):
    """从真机当前位置接管，避免第一帧跳变。"""
    norms_real = [cal.norm_from_pos(i, p) for i, p in enumerate(hand.read_pos())]
    sync.set_state(reorder_real_to_sim(norms_real))
