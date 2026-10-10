"""FDC2214 触觉链路的薄封装 —— 和 rohand.py 对称的用法。

串口那头是 STM32F103 + FDC2214,CH0 单通道高速模式,每帧 9 字节 ASCII::

    c0F63889a
    ↑ ──┬── ↑
   帧头  28位 帧尾

帧率约 2000/s,所以读串口必须放后台线程,主线程只管取结果。

典型用法::

    from tactile import Tactile

    with Tactile() as tac:
        tac.start()
        ...                      # 做你的实验
        rec = tac.stop()
    print(rec.fps, rec.pf.std())

TIMING: 每帧的时间戳不是"传感器采样时刻",而是推算出来的到达时刻。串口是
成批到的(一次 read 拿回几十帧),本模块把一批帧**在上一批到达时刻和本批到达
时刻之间均匀铺开** —— STM32 是定速死循环发送,这个假设成立得很好。

但批次到达时刻本身还落后于真实采样时刻一个**固定延迟**(USB 缓冲,CH340 的
延迟定时器默认 16ms)。做"电容比电流早多少毫秒发现接触"这类测量前,必须:
  1. 设备管理器 → 端口 → CH340 → 属性 → 端口设置 → 高级 → 延迟计时器 改 1ms
  2. 或用一次已知事件标定出固定偏移,分析时减掉
不做这步,"谁更早"的结论不成立。

CONSTANTS: 换算常数必须和这三处保持一致,改一处就要改三处 ——
  Hardware/FDC2214/fdc2214.h     (STM32 侧)
  GUI/fdc_gui/conversion.py      (原上位机)
  本文件的 SensorConfig
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import serial
import serial.tools.list_ports

__all__ = ["Tactile", "TactileError", "Recording", "SensorConfig",
           "raw_to_frequency_hz", "raw_to_capacitance_pf", "find_ch340"]

BAUDRATE = 460800
FRAME_SIZE = 9
_START = ord("c")
_END = ord("a")
_HEX = frozenset(b"0123456789abcdefABCDEF")

RAW_MASK = 0x0FFFFFFF
_TWO_POW_28 = float(1 << 28)

# CH340/CH341 的 USB 厂商号。沁恒全系都是 0x1A86。
_CH34X_VID = 0x1A86


class TactileError(RuntimeError):
    pass


@dataclass(frozen=True)
class SensorConfig:
    """LC 传感回路参数。

    警告:默认值是 TI Demo 板的,**不是那块手形传感器的真实参数**。
    所以算出来的 pF 绝对值不可信,但相对变化量 ΔpF 是可信的 —— 研究里
    基本只看 ΔpF,所以通常不影响。要绝对值就得实测 L 和固定 C 再改这里。
    """
    clock_hz: float = 40_000_000.0
    fref_divider: float = 1.0
    fin_divider: float = 2.0
    inductance_h: float = 18e-6
    fixed_capacitance_pf: float = 33.0


DEFAULT_SENSOR = SensorConfig()


def raw_to_frequency_hz(raw, config: SensorConfig = DEFAULT_SENSOR):
    """28 位原始值 → LC 回路振荡频率 (Hz)。"""
    raw_values = np.asarray(raw, dtype=np.uint32) & RAW_MASK
    reference_hz = config.clock_hz / config.fref_divider
    return raw_values.astype(np.float64) * reference_hz * config.fin_divider / _TWO_POW_28


def raw_to_capacitance_pf(raw, config: SensorConfig = DEFAULT_SENSOR):
    """28 位原始值 → 传感电容 (pF)。C = 1/(L·w^2) − C_fixed"""
    frequency = raw_to_frequency_hz(raw, config)
    angular = 2.0 * np.pi * frequency
    with np.errstate(divide="ignore", invalid="ignore"):
        total_pf = 1e12 / (config.inductance_h * angular * angular)
    return total_pf - config.fixed_capacitance_pf


def find_ch340():
    """按 VID 找 CH340/CH341,找不到再按描述兜底。返回 COM 口名或 None。"""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.vid == _CH34X_VID:
            return p.device
    for p in ports:
        text = "{} {}".format(p.description, p.manufacturer or "").upper()
        if "CH34" in text or "USB-SERIAL" in text:
            return p.device
    return None


class _FrameParser:
    """增量解析定长 cXXXXXXXa 帧,丢字节后自动重新对齐。

    照搬 GUI/fdc_gui/protocol.py 的逻辑 —— 那份写得很稳,失步后会向后
    找下一个 c 重新同步,不会一路错到底。
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self.valid_frames = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes):
        if data:
            self._buf.extend(data)
        out = []
        while True:
            start = self._buf.find(_START)
            if start < 0:
                self.discarded_bytes += len(self._buf)
                self._buf.clear()
                break
            if start:
                self.discarded_bytes += start
                del self._buf[:start]
            if len(self._buf) < FRAME_SIZE:
                break
            cand = self._buf[:FRAME_SIZE]
            payload = cand[1:8]
            if cand[8] == _END and all(c in _HEX for c in payload):
                out.append(int(payload.decode("ascii"), 16))
                self.valid_frames += 1
                del self._buf[:FRAME_SIZE]
            else:
                self.discarded_bytes += 1
                del self._buf[0]
        return out


@dataclass
class Recording:
    """一段采集。t 是秒(perf_counter 基准),raw 是 28 位整数,pf 由 raw 换算。"""
    t: np.ndarray
    raw: np.ndarray
    config: SensorConfig = DEFAULT_SENSOR
    label: str = ""
    discarded_bytes: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def pf(self) -> np.ndarray:
        return raw_to_capacitance_pf(self.raw, self.config)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def fps(self) -> float:
        return (len(self.t) - 1) / self.duration if self.duration > 0 else 0.0

    def detrend(self, seconds: float = 0.5) -> np.ndarray:
        """减掉慢漂移,只留高频噪声。

        EMI 分析要看的是"电机开关带来的抖动",不是几十秒尺度的温漂,所以
        用滑动均值做高通。窗口取 0.5s 量级。
        """
        pf = self.pf
        if len(pf) < 8:
            return pf - pf.mean() if len(pf) else pf
        n = max(3, int(seconds * self.fps) | 1)
        if n >= len(pf):
            return pf - pf.mean()
        kernel = np.ones(n) / n
        baseline = np.convolve(pf, kernel, mode="same")
        edge = n // 2
        return (pf - baseline)[edge:-edge] if len(pf) > 2 * edge else pf - baseline

    def __len__(self) -> int:
        return len(self.raw)

    def __repr__(self) -> str:
        return ("<Recording {!r} n={} {:.1f}s @{:.0f}fps>"
                .format(self.label, len(self), self.duration, self.fps))


class Tactile:
    """后台线程持续收帧。start() 开始累积,stop() 返回 Recording。"""

    def __init__(self, port=None, baudrate: int = BAUDRATE,
                 config: SensorConfig = DEFAULT_SENSOR) -> None:
        self.port = port
        self.baudrate = baudrate
        self.config = config
        self._ser = None
        self._parser = _FrameParser()
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._collecting = False
        self._t = []
        self._raw = []
        self._last_arrival = 0.0
        self._latest = None

    # ---- 连接 ----------------------------------------------------------

    def open(self) -> "Tactile":
        if self.port is None:
            self.port = find_ch340()
            if self.port is None:
                raise TactileError(
                    "没找到 CH340。检查 USB 转串口是否插好;"
                    "也可以手动指定 Tactile(port='COM3')")
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
        except serial.SerialException as e:
            raise TactileError("打不开 {}: {}".format(self.port, e)) from e
        self._ser.reset_input_buffer()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def __enter__(self) -> "Tactile":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 采集 ----------------------------------------------------------

    def _loop(self) -> None:
        self._last_arrival = time.perf_counter()
        while self._running:
            try:
                n = self._ser.in_waiting or 1
                data = self._ser.read(n)
            except (serial.SerialException, AttributeError, OSError):
                break
            if not data:
                continue
            arrival = time.perf_counter()
            values = self._parser.feed(data)
            if not values:
                continue
            # 把这一批帧在 [上批到达, 本批到达] 之间均匀铺开。
            # STM32 是定速死循环发送,均匀假设成立得很好。
            span = arrival - self._last_arrival
            k = len(values)
            stamps = [self._last_arrival + span * (i + 1) / k for i in range(k)]
            self._last_arrival = arrival
            with self._lock:
                self._latest = (stamps[-1], values[-1])
                if self._collecting:
                    self._t.extend(stamps)
                    self._raw.extend(values)

    def start(self) -> None:
        """清空缓存并开始累积。"""
        with self._lock:
            self._t.clear()
            self._raw.clear()
            self._collecting = True

    def stop(self, label: str = "") -> Recording:
        """停止累积并返回这一段数据。"""
        with self._lock:
            self._collecting = False
            t = np.asarray(self._t, dtype=np.float64)
            raw = np.asarray(self._raw, dtype=np.uint32)
        return Recording(t=t, raw=raw, config=self.config, label=label,
                         discarded_bytes=self._parser.discarded_bytes)

    def record(self, seconds: float, label: str = "",
               progress: bool = True) -> Recording:
        """阻塞采集固定时长。"""
        self.start()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            if progress:
                left = seconds - (time.perf_counter() - t0)
                with self._lock:
                    n = len(self._raw)
                print("\r    采集中 {:4.1f}s  已收 {:6d} 帧".format(left, n),
                      end="", flush=True)
            time.sleep(0.1)
        rec = self.stop(label)
        if progress:
            print("\r    完成 {:4.1f}s  {:6d} 帧  实测 {:.0f} fps            "
                  .format(rec.duration, len(rec), rec.fps))
        return rec

    def latest(self):
        """最近一帧 (时间戳, raw),没有数据返回 None。"""
        with self._lock:
            return self._latest

    def wait_for_data(self, timeout: float = 3.0) -> bool:
        """等到真的有帧进来。返回 False 说明链路不通。"""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.latest() is not None:
                return True
            time.sleep(0.05)
        return False


def _selftest() -> int:
    """python tactile.py —— 只读自检,确认触觉链路通不通。"""
    print("触觉链路自检")
    print("-" * 52)
    port = find_ch340()
    print("  CH340        : {}".format(port or "未找到"))
    if port is None:
        print("\n  串口列表:")
        for p in serial.tools.list_ports.comports():
            print("    {:8s} VID={} PID={}  {}"
                  .format(p.device, p.vid, p.pid, p.description))
        print("\n  插上 USB 转串口再试。")
        return 1
    try:
        with Tactile() as tac:
            print("  波特率       : {}".format(tac.baudrate))
            if not tac.wait_for_data(3.0):
                print("\n  x 打开了串口但收不到帧。检查:")
                print("    - STM32 是否上电、程序是否烧进去")
                print("    - TX/RX 是否交叉、是否共地")
                print("    - 波特率是否 460800")
                return 1
            rec = tac.record(3.0, "selftest")
            pf = rec.pf
            print("-" * 52)
            print("  帧率         : {:.0f} fps".format(rec.fps))
            print("  丢弃字节     : {}".format(rec.discarded_bytes))
            print("  RAW 范围     : {} ~ {}".format(rec.raw.min(), rec.raw.max()))
            print("  频率         : {:.3f} MHz"
                  .format(raw_to_frequency_hz(rec.raw).mean() / 1e6))
            print("  电容         : {:.3f} pF  (sigma={:.4f})"
                  .format(pf.mean(), pf.std()))
            print("  高通后噪声   : sigma={:.4f} pF".format(rec.detrend().std()))
            print("-" * 52)
            if rec.raw.min() == 0 or rec.raw.max() >= RAW_MASK:
                print("  ! RAW 触到边界,LC 回路可能没搭好或电极没接")
            elif rec.fps < 500:
                print("  ! 帧率偏低({:.0f}),预期 ~2000".format(rec.fps))
            else:
                print("  OK 链路正常")
    except TactileError as e:
        print("  x {}".format(e))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
