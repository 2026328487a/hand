"""手指卡滞诊断 —— 判断"回不到张开位"是机械卡住还是控制器放弃。

背景:probe 发现无名指(pos 3556)、小指(3243) 目标是 0 却停在半路,而且
**电流是 0**;拇指旋转停在 562(目标 728),持续 33mA。标定时这六个通道
都能到 [0,0,0,0,0,728],所以是标定之后新出现的。

判读逻辑:

    命令张开,然后看 ——
    位置动了、回到 0                 -> 正常,之前只是停在半路
    位置不动、电流高                 -> 机械卡住(有东西挡着)
    位置不动、电流仍是 0             -> 控制器根本没出力(保护latch/失步)
    闭合方向能动、张开方向不能动      -> 阻力在伸直方向(典型:背侧贴了东西)

安全措施:
  - 低速(默认 60,正常手势用 150~255)
  - 电流超过 --abort(默认 450mA)立刻停这一路。
    参考:正常运动 0~180mA,碰到物体 350~600mA,顶死 852~1168mA
  - 先测闭合(顺着阻力,安全),再测张开(逆着阻力,靠限流保护)
  - 任何异常都会把手停下

用法::

    python probe_jam.py                 # 默认测 无名指/小指/拇指旋转 + 食指对照
    python probe_jam.py --abort 350     # 更保守
    python probe_jam.py --motors 3 4    # 只测某几路
"""

from __future__ import annotations

import argparse
import time

from rohand import ROHand, MOTOR_CN, MOTORS

# 各路的"张开位"。只有拇指旋转不是 0。
OPEN_POS = [0, 0, 0, 0, 0, 728]
# 测试时往闭合方向走多少(拇指旋转反过来,它的闭合是往小走)
PROBE_DELTA = 9000

SUSPECT = [3, 4, 5]     # 无名指、小指、拇指旋转
CONTROL = 1             # 食指 —— 已确认 0.00° 偏差,拿来当健康对照


def trace(hand, motor, target, seconds, abort_ma, tag, speed=60):
    """命令一路到 target,采样位置和电流,返回轨迹。"""
    t0 = time.perf_counter()
    hand.move_one(motor, target, speed=speed)
    samples = []
    aborted = False
    while time.perf_counter() - t0 < seconds:
        pos = hand.read_pos()[motor]
        cur = hand.read_currents()[motor]
        samples.append((time.perf_counter() - t0, pos, cur))
        if cur > abort_ma:
            hand.stop([motor])
            aborted = True
            break
        time.sleep(0.05)
    moved = samples[-1][1] - samples[0][1] if samples else 0
    peak = max((c for _, _, c in samples), default=0)
    print("    {:<6s} 目标{:>6d}  起点{:>6d} -> 终点{:>6d}  位移{:>+7d}  "
          "峰值电流{:>5d}mA{}".format(
              tag, target, samples[0][1] if samples else -1,
              samples[-1][1] if samples else -1, moved, peak,
              "  [限流中止]" if aborted else ""))
    return {"samples": samples, "moved": moved, "peak": peak, "aborted": aborted}


def verdict(close_r, open_r, at_open_after):
    """根据两个方向的响应给结论。"""
    c_moved, o_moved = abs(close_r["moved"]), abs(open_r["moved"])
    o_peak = open_r["peak"]
    if at_open_after:
        return "正常", "能回到张开位,之前只是停在半路(可能是上次指令没走完)"
    if o_moved < 300 and o_peak < 30:
        return "未出力", "张开方向位置没动而且电流≈0 —— 控制器根本没使劲,像保护latch或失步"
    if o_moved < 300 and o_peak >= 30:
        return "机械卡住", "张开方向在出力但动不了 —— 有东西挡着(查 FPC/线束/胶带)"
    if c_moved > 1000 and o_moved < c_moved * 0.5:
        return "单向受阻", "闭合能动、张开动不了 —— 阻力在伸直方向,典型是背侧贴了东西"
    return "部分恢复", "动了一些但没完全到位,阻力偏大"


def main() -> int:
    ap = argparse.ArgumentParser(description="手指卡滞诊断")
    ap.add_argument("--abort", type=int, default=450,
                    help="单路电流超过就中止(mA),默认 450")
    ap.add_argument("--motors", type=int, nargs="*", default=None,
                    help="只测这几路(0-5),默认 3 4 5 加食指对照")
    ap.add_argument("--speed", type=int, default=60, help="运动速度,默认 60")
    args = ap.parse_args()

    motors = args.motors if args.motors is not None else SUSPECT + [CONTROL]

    print("=" * 76)
    print("手指卡滞诊断    限流 {}mA   速度 {}".format(args.abort, args.speed))
    print("=" * 76)

    results = {}
    with ROHand() as hand:
        before = hand.read_pos()
        cur0 = hand.read_currents()
        print("\n起始状态:")
        for i in range(6):
            flag = "  <- 偏离张开位" if abs(before[i] - OPEN_POS[i]) > 200 else ""
            print("  {:<8s} pos={:>6d}  (张开位 {:>4d})  电流 {:>4d}mA{}".format(
                MOTOR_CN[i], before[i], OPEN_POS[i], cur0[i], flag))

        try:
            for m in motors:
                is_ctrl = (m == CONTROL and args.motors is None)
                print("\n[{}] {}{}".format(
                    m, MOTOR_CN[m], "   ★健康对照组" if is_ctrl else ""))

                # 1) 闭合方向 —— 顺着阻力走,安全
                # 六路的闭合方向都是 pos 变大。拇指旋转的张开位是 728
                # (不是 0),norm->pos 是 728 + (65535-728)*norm,所以它的
                # 闭合同样是往大走 —— 第一版这里写反了,命令 0 只会让它
                # 停在 728,等于没测到闭合方向。
                close_target = min(65535, before[m] + PROBE_DELTA)
                close_r = trace(hand, m, close_target, 2.5, args.abort, "闭合", args.speed)

                # 2) 张开方向 —— 逆着阻力,靠限流保护
                open_r = trace(hand, m, OPEN_POS[m], 3.5, args.abort, "张开", args.speed)

                final = hand.read_pos()[m]
                at_open = abs(final - OPEN_POS[m]) <= 200
                tag, why = verdict(close_r, open_r, at_open)
                print("    => [{}] {}".format(tag, why))
                results[m] = (tag, why, final)

        except KeyboardInterrupt:
            print("\n  中断,正在停手 ...")
        finally:
            try:
                hand.stop()
            except Exception:
                pass

        after = hand.read_pos()

    print("\n" + "=" * 76)
    print("汇总")
    print("=" * 76)
    print("  {:<10s}{:>10s}{:>10s}{:>10s}   {}".format(
        "通道", "测试前", "测试后", "张开位", "结论"))
    print("  " + "-" * 72)
    for m in sorted(results):
        tag = results[m][0]
        print("  {:<10s}{:>10d}{:>10d}{:>10d}   {}".format(
            MOTOR_CN[m], before[m], after[m], OPEN_POS[m], tag))
    print()
    stuck = [m for m in results if results[m][0] in ("机械卡住", "单向受阻")]
    dead = [m for m in results if results[m][0] == "未出力"]
    if stuck:
        print("  机械受阻: {}".format("、".join(MOTOR_CN[m] for m in stuck)))
        print("    -> 断电检查这几根手指上有没有 FPC、引线、胶带挡住伸直方向")
        print("    -> 传感器可能正在被挤压,有损坏风险,尽快拆下确认")
    if dead:
        print("  未出力  : {}".format("、".join(MOTOR_CN[m] for m in dead)))
        print("    -> 电流为 0 说明不是卡住。试试重新上电,或看 SDK 有没有报错状态")
    if not stuck and not dead:
        print("  未发现卡滞,三路都能回到张开位。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
