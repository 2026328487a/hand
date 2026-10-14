"""ROH-LiteS001 真机的薄封装 —— 把官方 SDK 的 C 风格 API 包成 Python 用法。

官方 `OHandSerialAPI` 是从 C 头文件直译过来的:每个 Get 都要调用方预先造好
输出容器,返回的是元组 `(err, ...)` 而不是裸错误码,而且容器类型有讲究
(见下面 CONTAINER 注释)。这个模块把这些细节收在一处,后面的标定 / 手势 /
仿真桥接都只用这里的方法。

典型用法::

    from rohand import ROHand

    with ROHand() as hand:
        print(hand.read_pos())          # 6 个通道当前位置 0-65535
        hand.move([0, 0, 0, 0, 0, 728]) # 张开
        hand.wait_settled()

CONTAINER: SDK 往输出容器的 `[i]` 里写的是**整数**(位置 0-65535、角度带符号),
所以容器必须是 `list` 而不是 `bytearray` —— bytearray 只能存 0-255,一超界
就抛 ValueError。另外带 `motor_cnt` 的接口要把它预置成 `[MAX_MOTOR_CNT]`,
SDK 会拿它当"调用方缓冲区有多大"来校验,传 0 直接 DATA_SIZE_TOO_BIG。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ohand.constants import (HAND_PROTOCOL_UART, HAND_RESP_SUCCESS,
                             MAX_MOTOR_CNT, MAX_THUMB_ROOT_POS)
from ohand.interface.can import (delay_milli_seconds_impl,
                                 get_milli_seconds_impl, recv_data_impl,
                                 send_data_impl)
from ohand.OHandSerialAPI import OHandSerialAPI

__all__ = ["ROHand", "HandError", "MOTORS", "MOTOR_CN", "N_MOTORS", "POS_MAX"]

N_MOTORS = MAX_MOTOR_CNT            # 6
POS_MAX = 65535

# 实测顺序:GetFingerPosAll 返回的通道 0..5。
# 注意拇指弯曲在**最前**,而 MuJoCo XML 的执行器把拇指放在最后 —— sim2real
# 映射时要重排,见 sim_bridge.py 的 SIM_TO_REAL。
MOTORS = ("thumb_bend", "index", "middle", "ring", "little", "thumb_rot")
MOTOR_CN = ("拇指弯曲", "食指", "中指", "无名指", "小指", "拇指旋转")

DEFAULT_HAND_ID = 0x02
DEFAULT_MASTER_ID = 0x01
DEFAULT_BITRATE = 1_000_000

_ERR_NAMES = {
    1: "TIMER_FUNC_NOT_SET", 2: "INVALID_CONTEXT", 3: "TIMEOUT",
    4: "INVALID_OUT_BUFFER_SIZE", 5: "UNMATCHED_ADDR", 6: "UNMATCHED_CMD",
    7: "DATA_SIZE_TOO_BIG", 8: "DATA_INVALID", 0xFF: "HAND_ERROR",
}


class HandError(RuntimeError):
    """SDK 返回了非 0 错误码。"""

    def __init__(self, op: str, err: int, remote_err: int = 0):
        self.op, self.err, self.remote_err = op, err, remote_err
        detail = _ERR_NAMES.get(err, f"0x{err:02X}")
        extra = f", remote_err={remote_err}" if err == 0xFF else ""
        super().__init__(f"{op} failed: {detail}{extra}")


@dataclass
class CaliData:
    """手自己存的标定端点(出厂写入,不是我们测的)。"""
    start_pos: list        # 每个电机的原始编码器起点
    end_pos: list          # 原始编码器终点
    thumb_root_pos: list   # 拇指旋转的预设档位原始值


def _err(ret) -> int:
    """SDK 的 Get* 返回 (err, ...) 元组,Set* 返回裸码。统一取出错误码。"""
    return ret[0] if isinstance(ret, tuple) else ret


class ROHand:
    def __init__(self, hand_id: int = DEFAULT_HAND_ID,
                 master_id: int = DEFAULT_MASTER_ID,
                 bitrate: int = DEFAULT_BITRATE,
                 channel: str = "PCAN_USBBUS1",
                 timeout_ms: int = 255):
        self.hand_id = hand_id
        self.master_id = master_id
        self.bitrate = bitrate
        self.channel = channel
        self.timeout_ms = timeout_ms
        self.bus = None
        self.api = None
        self._rerr = bytearray(1)

    # ---------- 连接 ----------

    def _release_channel(self):
        """PCAN 通道是独占的,而且进程崩溃时驱动不一定回收。

        上次跑脚本时只要在 bus 用着的时候抛了异常,下次开总线就会报
        "A PCAN Channel has not been initialized yet",很容易误判成被别的
        程序占用。所以开之前无条件 Uninitialize 一次。
        """
        try:
            from can.interfaces.pcan import basic
            basic.PCANBasic().Uninitialize(getattr(basic, self.channel))
        except Exception:                       # noqa: BLE001 - 清理失败不该阻断
            pass

    def open(self):
        import can
        self._release_channel()
        self.bus = can.interface.Bus(interface="pcan", channel=self.channel,
                                     bitrate=self.bitrate)
        # protocol 传 UART 不是笔误 —— 那是帧格式标识,和物理层无关,
        # 走 CAN 时官方示例也传这个值。
        self.api = OHandSerialAPI(self.bus, HAND_PROTOCOL_UART, self.master_id,
                                  send_data_impl, recv_data_impl)
        self.api.HAND_SetTimerFunction(get_milli_seconds_impl,
                                       delay_milli_seconds_impl)
        self.api.HAND_SetCommandTimeOut(self.timeout_ms)
        return self

    def close(self):
        if self.bus is not None:
            self.bus.shutdown()
            self.bus = None
        self.api = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ---------- 只读 ----------

    def versions(self) -> dict:
        a = self.api
        major, minor, rev = [0], [0], [0]
        ret = a.HAND_GetProtocolVersion(self.hand_id, major, minor, self._rerr)
        if _err(ret) != HAND_RESP_SUCCESS:
            raise HandError("GetProtocolVersion", _err(ret), self._rerr[0])
        out = {"protocol": f"{major[0]}.{minor[0]}"}
        if _err(a.HAND_GetFirmwareVersion(self.hand_id, major, minor, rev,
                                          self._rerr)) == HAND_RESP_SUCCESS:
            out["firmware"] = f"{major[0]}.{minor[0]}.{rev[0]}"
        w = ([0], [0], [0])
        if _err(a.HAND_GetUID(self.hand_id, w[0], w[1], w[2],
                              self._rerr)) == HAND_RESP_SUCCESS:
            out["uid"] = "-".join(f"{x[0]:08X}" for x in w)
        return out

    def read_pos(self) -> list:
        """六个电机当前位置,0-65535(0 = 张开)。"""
        return self._read_all(self.api.HAND_GetFingerPosAll,
                              "GetFingerPosAll")[1]

    def read_pos_target(self) -> list:
        return self._read_all(self.api.HAND_GetFingerPosAll,
                              "GetFingerPosAll")[0]

    def read_angle(self) -> list:
        """六个关节当前角度,单位度(SDK 原始值是 0.01°)。"""
        _, cur = self._read_all(self.api.HAND_GetFingerAngleAll,
                                "GetFingerAngleAll")
        return [v / 100 for v in cur]

    def read_pos_abs(self) -> list:
        """原始编码器读数,不做 0-65535 归一化。标定时用来看真实行程。"""
        return self._read_all(self.api.HAND_GetFingerPosAbsAll,
                              "GetFingerPosAbsAll")[1]

    def _read_all(self, fn, name):
        tgt = [0] * N_MOTORS
        cur = [0] * N_MOTORS
        cnt = [N_MOTORS]                    # 见模块头 CONTAINER 注释
        ret = fn(self.hand_id, tgt, cur, cnt, self._rerr)
        if _err(ret) != HAND_RESP_SUCCESS:
            raise HandError(name, _err(ret), self._rerr[0])
        n = cnt[0]
        return tgt[:n], cur[:n]

    def read_current(self, motor: int) -> int:
        """单个电机的电流(mA)。夹持力的间接指标。"""
        cur = [0]
        ret = self.api.HAND_GetFingerCurrent(self.hand_id, motor, cur,
                                             self._rerr)
        if _err(ret) != HAND_RESP_SUCCESS:
            raise HandError(f"GetFingerCurrent({motor})", _err(ret),
                            self._rerr[0])
        return cur[0]

    def read_currents(self) -> list:
        return [self.read_current(i) for i in range(N_MOTORS)]

    def read_pos_limit(self, motor: int):
        low, high = [0], [0]
        ret = self.api.HAND_GetFingerPosLimit(self.hand_id, motor, low, high,
                                              self._rerr)
        if _err(ret) != HAND_RESP_SUCCESS:
            raise HandError(f"GetFingerPosLimit({motor})", _err(ret),
                            self._rerr[0])
        return low[0], high[0]

    def read_cali_data(self) -> CaliData:
        end = [0] * N_MOTORS
        start = [0] * N_MOTORS
        cnt = [N_MOTORS]
        troot = [0] * MAX_THUMB_ROOT_POS
        tcnt = [MAX_THUMB_ROOT_POS]
        ret = self.api.HAND_GetCaliData(self.hand_id, end, start, cnt, troot,
                                        tcnt, self._rerr)
        if _err(ret) != HAND_RESP_SUCCESS:
            raise HandError("GetCaliData", _err(ret), self._rerr[0])
        return CaliData(start[:cnt[0]], end[:cnt[0]], troot[:tcnt[0]])

    def read_battery_mv(self) -> int:
        v = [0]
        ret = self.api.HAND_GetBatteryVoltage(self.hand_id, v, self._rerr)
        if _err(ret) != HAND_RESP_SUCCESS:
            raise HandError("GetBatteryVoltage", _err(ret), self._rerr[0])
        return v[0]

    # ---------- 运动 ----------

    def move(self, pos, speed=255):
        """一条命令下发全部 6 个电机。pos 是 0-65535 的整数序列。

        SDK 的 SetFingerPosAll 会校验类型,所以这里强制转 int 并夹紧范围 ——
        浮点(例如从仿真归一化算出来的)会被判 DATA_INVALID。
        """
        pos = [max(0, min(POS_MAX, int(round(p)))) for p in pos]
        if len(pos) != N_MOTORS:
            raise ValueError(f"need {N_MOTORS} positions, got {len(pos)}")
        if isinstance(speed, int):
            speeds = [speed] * N_MOTORS
        else:
            speeds = [max(0, min(255, int(s))) for s in speed]
        err = self.api.HAND_SetFingerPosAll(self.hand_id, pos, speeds,
                                            N_MOTORS, self._rerr)
        if _err(err) != HAND_RESP_SUCCESS:
            raise HandError("SetFingerPosAll", _err(err), self._rerr[0])

    def move_one(self, motor: int, pos: int, speed: int = 255):
        err = self.api.HAND_SetFingerPos(self.hand_id, motor,
                                         max(0, min(POS_MAX, int(pos))),
                                         speed, self._rerr)
        if _err(err) != HAND_RESP_SUCCESS:
            raise HandError(f"SetFingerPos({motor})", _err(err), self._rerr[0])

    def stop(self, motors=None):
        """停在当前位置。motors=None 表示全停。"""
        bits = 0x3F if motors is None else sum(1 << m for m in motors)
        err = self.api.HAND_FingerStop(self.hand_id, bits, self._rerr)
        if _err(err) != HAND_RESP_SUCCESS:
            raise HandError("FingerStop", _err(err), self._rerr[0])

    def wait_settled(self, tol: int = 200, timeout: float = 4.0,
                     poll: float = 0.05, should_cancel=None) -> bool:
        """等到位置不再变化。

        不用"到达目标"作判据 —— 手指被挡住时永远到不了目标位,那样会一直等到
        超时。改成看**连续两次读数是否稳定**,挡住的情况也能及时返回。
        """
        deadline = time.perf_counter() + timeout
        prev = self.read_pos()
        stable = 0
        while time.perf_counter() < deadline:
            if should_cancel is not None and should_cancel():
                self.stop()
                raise InterruptedError("动作已取消")
            time.sleep(poll)
            cur = self.read_pos()
            if max(abs(a - b) for a, b in zip(cur, prev)) <= tol:
                stable += 1
                if stable >= 2:
                    return True
            else:
                stable = 0
            prev = cur
        return False

    def approach(self, target, current_limit_ma: int, speed: int = 90,
                 sub_steps: int = 12, step_timeout: float = 0.6,
                 reach_tol: int = 400, watch=None):
        """分段逼近目标,任一被监视电机电流超阈值就停在当前位置。

        位置执行器直接发终点会一路顶到底 —— 空捏时实测 852mA,而出厂上限是
        1199mA。抓真实物体要的是"碰到就停",所以把行程切成若干小段,每段后读
        电流。这是这只手唯一的力反馈途径:它没有力矩传感器,电流是夹持力的
        间接量。

        每个子步必须**等实际位置跟上来**再测电流。`move()` 只是下发目标就
        返回,如果固定 sleep 一小段就采样,指令点会一直跑在实际位置前面,电机
        全程都在正常加速、电流不高,直到最后一步才堵转 —— 实测就是这样漏掉
        接触、事后才看到 816mA。判据取"到位 **或** 位置不再变化",后者正是
        被挡住的特征。

        watch: 要监视的电机 index;None 表示监视本次会动的那几路。
        返回 (是否触发接触, 停下时的位置, 触发的 (电机, 电流) 列表)。
        """
        start = self.read_pos()
        target = [max(0, min(POS_MAX, int(round(p)))) for p in target]
        if watch is None:
            watch = [i for i in range(N_MOTORS)
                     if abs(target[i] - start[i]) > 200]
        if not watch:
            self.move(target, speed=speed)
            self.wait_settled(timeout=2.0)
            return False, self.read_pos(), []

        for k in range(1, sub_steps + 1):
            mid = [s + (t - s) * k / sub_steps for s, t in zip(start, target)]
            self.move(mid, speed=speed)

            # 等到位或卡住,再采电流
            deadline = time.perf_counter() + step_timeout
            prev = None
            while time.perf_counter() < deadline:
                time.sleep(0.03)
                cur = self.read_pos()
                reached = all(abs(cur[i] - mid[i]) <= reach_tol for i in watch)
                stalled = prev is not None and all(
                    abs(cur[i] - prev[i]) <= 40 for i in watch)
                if reached or stalled:
                    break
                prev = cur

            hot = [(i, c) for i in watch
                   if (c := self.read_current(i)) > current_limit_ma]
            if hot:
                self.stop()                 # 停在这里:不回退,也不继续压
                return True, self.read_pos(), hot

        self.wait_settled(timeout=2.0)
        return False, self.read_pos(), []
