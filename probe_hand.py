"""ROH-LiteS001 真机连接自检 —— **只读,不发任何运动指令**。

按顺序检查:
    PCAN-USB 驱动 → CAN 总线能否打开 → 总线上有没有报文 → 手是否应答 → 读状态

用法:
    python probe_hand.py                 # 全套自检
    python probe_hand.py --bitrate 500000
    python probe_hand.py --hand-id 2

刻意不包含任何 HAND_Set* 调用。确认这个脚本能读到手之后,再谈动作。
"""

import argparse
import ctypes
import sys

HAND_ID_DEFAULT = 0x02      # 官方示例里手的节点地址
MASTER_ID = 0x01
BITRATES = (1000000, 500000, 250000)
# 实测顺序(GetFingerPosAll 返回的 6 个通道,index 0..5)
MOTOR_NAMES = ["拇指弯曲", "食指", "中指", "无名指", "小指", "拇指旋转"]


def code(ret):
    """SDK 的 HAND_Get* 返回 (err, ...) 元组,取出真正的错误码。"""
    return ret[0] if isinstance(ret, tuple) else ret


def check_driver():
    """PCANBasic.dll 在不在。不在就没必要往下走了。"""
    print("[1/5] 检查 PCAN 驱动 ...")
    try:
        ctypes.WinDLL("PCANBasic.dll")
    except OSError:
        print("      × PCANBasic.dll 未找到 —— PEAK 驱动没装")
        print("        装 PEAK 的 'PCAN-USB Driver for Windows'(含 PCANBasic.dll)后再跑")
        return False
    print("      ○ PCANBasic.dll 已就位")
    return True


def release_channel(channel="PCAN_USBBUS1"):
    """强制释放通道。

    PCAN 通道是独占的,而且**进程崩溃时驱动不一定会回收** —— 上一次跑脚本时
    只要在 bus 用着的时候抛了异常,下一次开总线就会报
    "A PCAN Channel has not been initialized yet",很容易误判成被别的程序占用。
    所以每次开之前先无条件 Uninitialize 一次。
    """
    try:
        from can.interfaces.pcan import basic
        basic.PCANBasic().Uninitialize(getattr(basic, channel))
    except Exception:                              # noqa: BLE001 - 清理失败不该阻断主流程
        pass


def open_bus(bitrate, channel="PCAN_USBBUS1"):
    import can
    release_channel(channel)
    try:
        bus = can.interface.Bus(interface="pcan", channel=channel, bitrate=bitrate)
    except Exception as exc:                       # noqa: BLE001 - 想看到全部失败原因
        print(f"      × {bitrate // 1000}k 打不开: {type(exc).__name__}: {str(exc)[:120]}")
        return None
    print(f"      ○ 总线已打开 @ {bitrate // 1000} kbit/s")
    return bus


def sniff(bus, seconds=2.0):
    """被动听总线。手上电后通常会周期性发状态帧,能听到就说明物理链路通了。"""
    print(f"[3/5] 被动监听 {seconds:.0f} 秒(不发送任何东西) ...")
    seen = {}
    deadline = None
    import time
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        msg = bus.recv(timeout=0.2)
        if msg is None:
            continue
        seen.setdefault(msg.arbitration_id, 0)
        seen[msg.arbitration_id] += 1
    if not seen:
        print("      · 没听到报文。手可能不主动发帧(正常),继续问它")
    else:
        for arb, n in sorted(seen.items()):
            print(f"      ○ ID 0x{arb:03X}  {n} 帧")
    return seen


def query(bus, hand_id):
    """用官方 SDK 问手要版本和当前位置。全部是读命令。"""
    from ohand.constants import HAND_PROTOCOL_UART, HAND_RESP_SUCCESS, MAX_MOTOR_CNT
    from ohand.interface.can import (delay_milli_seconds_impl, get_milli_seconds_impl,
                                     recv_data_impl, send_data_impl)
    from ohand.OHandSerialAPI import OHandSerialAPI

    print(f"[4/5] 向节点 0x{hand_id:02X} 发读命令 ...")
    api = OHandSerialAPI(bus, HAND_PROTOCOL_UART, MASTER_ID,
                         send_data_impl, recv_data_impl)
    api.HAND_SetTimerFunction(get_milli_seconds_impl, delay_milli_seconds_impl)
    api.HAND_SetCommandTimeOut(255)

    remote_err = bytearray(1)
    major, minor, revision = bytearray(1), bytearray(1), bytearray(2)

    # 注意:SDK 的 HAND_Get* 返回的是元组 (err, ...),不是裸错误码。
    # 直接和 HAND_RESP_SUCCESS 比会永远不等 —— 第一版就栽在这。
    err = code(api.HAND_GetProtocolVersion(hand_id, major, minor, remote_err))
    if err != HAND_RESP_SUCCESS:
        print(f"      × 协议版本读取失败,err={err}"
              f"{' (3 = 超时,手没应答)' if err == 3 else ''}")
        return None
    print(f"      ○ 协议版本 {major[0]}.{minor[0]}")

    if code(api.HAND_GetFirmwareVersion(hand_id, major, minor, revision, remote_err)) == HAND_RESP_SUCCESS:
        print(f"      ○ 固件版本 {major[0]}.{minor[0]}.{int.from_bytes(revision, 'little')}")

    # UID 是 3 个 32 位字,SDK 会往容器 [0] 里写整数,所以得用 list 不能用 bytearray
    w0, w1, w2 = [0], [0], [0]
    if code(api.HAND_GetUID(hand_id, w0, w1, w2, remote_err)) == HAND_RESP_SUCCESS:
        print(f"      ○ UID {w0[0]:08X}-{w1[0]:08X}-{w2[0]:08X}")

    # 容器必须是 list,不能是 bytearray —— SDK 往 container[i] 写的是**整数**
    # (位置 0-65535、角度带符号),bytearray 只能存 0-255,一超界就抛
    # ValueError,被上层吞掉表现为"读取失败"。第二版栽在这。
    print("[5/5] 读当前手指位置 ...")
    target, current, motor_cnt = [0] * MAX_MOTOR_CNT, [0] * MAX_MOTOR_CNT, [MAX_MOTOR_CNT]
    if code(api.HAND_GetFingerPosAll(hand_id, target, current, motor_cnt, remote_err)) == HAND_RESP_SUCCESS:
        for i in range(motor_cnt[0]):
            label = MOTOR_NAMES[i] if i < len(MOTOR_NAMES) else f"电机{i}"
            print(f"      {label:8s} 当前 {current[i]:5d} / 目标 {target[i]:5d}   (0-65535)")
    else:
        print("      · 位置读取失败")

    tgt_a, cur_a, cnt_a = [0] * MAX_MOTOR_CNT, [0] * MAX_MOTOR_CNT, [MAX_MOTOR_CNT]
    if code(api.HAND_GetFingerAngleAll(hand_id, tgt_a, cur_a, cnt_a, remote_err)) == HAND_RESP_SUCCESS:
        print("      角度(单位 0.01°):")
        for i in range(cnt_a[0]):
            label = MOTOR_NAMES[i] if i < len(MOTOR_NAMES) else f"电机{i}"
            print(f"      {label:8s} {cur_a[i] / 100:8.2f}°")
    return api


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitrate", type=int, default=None,
                    help="指定波特率;默认依次试 1M / 500k / 250k")
    ap.add_argument("--hand-id", type=lambda x: int(x, 0), default=HAND_ID_DEFAULT)
    ap.add_argument("--channel", default="PCAN_USBBUS1")
    args = ap.parse_args()

    if not check_driver():
        return 1

    print("[2/5] 打开 CAN 总线 ...")
    bus = None
    for bitrate in ([args.bitrate] if args.bitrate else BITRATES):
        bus = open_bus(bitrate, args.channel)
        if bus is not None:
            break
    if bus is None:
        print("      总线始终打不开。检查适配器是否被别的程序占用(PCAN-View 等)")
        return 1

    try:
        sniff(bus)
        if query(bus, args.hand_id) is None:
            print("\n手没有应答。逐项排查:")
            print("  · 手的 24V 电源是否已上电(只插 CAN 不供电是不会应答的)")
            print("  · CAN_H / CAN_L 有没有接反,120Ω 终端电阻是否接了")
            print("  · 波特率是否匹配(用 --bitrate 逐个试)")
            print("  · 节点地址是否是 0x02(用 --hand-id 试 0x01..0x07)")
            return 1
        print("\n真机链路正常,可以读到状态。")
    finally:
        bus.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
