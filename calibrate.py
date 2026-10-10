"""标定 ROH-LiteS001:测出 0-65535 指令位置到实际行程的对应关系。

为什么需要这一步:手接受的是 0-65535 的归一化位置,但读回的角度不是简单
正比 —— 张开(pos=0)时四指角度是 ~173° 而不是 0°。要把仿真的关节角或
归一化 action 映射到真机,必须先测出每个电机的实测曲线。

做法:**一次只动一个电机**,其余保持张开,沿 0→65535→0 走若干点位,每点
等稳定后记录 位置 / 角度 / 原始编码器 / 电流。往回走是为了看回差
(backlash)—— 同一个指令位置在上行和下行读到的角度差多少。

用法::

    python calibrate.py                      # 全部 6 个电机
    python calibrate.py --motors 1 2         # 只标食指和中指
    python calibrate.py --steps 5 --speed 80 # 更少点位、更慢
    python calibrate.py --dry-run            # 只打印计划,不动手

输出 `calibration.json`,后续 gestures.py / sim_bridge.py 直接读它。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rohand import MOTOR_CN, MOTORS, N_MOTORS, POS_MAX, HandError, ROHand

OUT_PATH = Path(__file__).with_name("calibration.json")

# 张开姿态。拇指旋转不是 0 —— 它的机械零点在 2983(原始编码器),归一化后
# 静止读数是 728,直接给 0 会让它顶到内侧限位,所以张开态就用实测值。
OPEN_POSE = [0, 0, 0, 0, 0, 728]


def sweep_points(steps: int):
    """上行 + 下行的指令位置序列。两端都包含。"""
    up = [round(POS_MAX * i / (steps - 1)) for i in range(steps)]
    down = list(reversed(up[:-1]))          # 顶点不重复采
    return [(p, "up") for p in up] + [(p, "down") for p in down]


def sample(hand: ROHand, motor: int) -> dict:
    """读一个点位的全部状态量。"""
    pos = hand.read_pos()
    angle = hand.read_angle()
    abs_enc = hand.read_pos_abs()
    return {
        "pos_read": pos[motor],
        "angle_deg": angle[motor],
        "abs_enc": abs_enc[motor],
        "current_ma": hand.read_current(motor),
        # 同时记全部 6 路,用来发现耦合(动一个手指时别的是否被带动)
        "all_pos": pos,
        "all_angle": angle,
    }


def calibrate_motor(hand: ROHand, motor: int, steps: int, speed: int,
                    settle_tol: int, settle_timeout: float) -> dict:
    name = MOTORS[motor]
    print(f"\n=== 电机 {motor} {name} ({MOTOR_CN[motor]}) ===")

    # 其余电机回张开态,避免手指互相挡住影响行程
    hand.move(OPEN_POSE, speed=speed)
    hand.wait_settled(tol=settle_tol, timeout=settle_timeout)

    records = []
    for cmd, direction in sweep_points(steps):
        pose = list(OPEN_POSE)
        pose[motor] = cmd
        hand.move(pose, speed=speed)
        settled = hand.wait_settled(tol=settle_tol, timeout=settle_timeout)
        time.sleep(0.15)                    # 稳定判据过后再让机械静一下
        s = sample(hand, motor)
        s.update(cmd=cmd, direction=direction, settled=settled)
        records.append(s)
        flag = "" if settled else "  [未稳定/可能被挡]"
        print(f"  cmd={cmd:6d} {direction:4s} → 读回 {s['pos_read']:6d}  "
              f"角度 {s['angle_deg']:7.2f}°  enc {s['abs_enc']:6d}  "
              f"电流 {s['current_ma']:4d}mA{flag}")

    hand.move(OPEN_POSE, speed=speed)
    hand.wait_settled(tol=settle_tol, timeout=settle_timeout)

    ups = [r for r in records if r["direction"] == "up"]
    downs = [r for r in records if r["direction"] == "down"]
    angles = [r["angle_deg"] for r in records]

    # 回差:同一指令位置,上行与下行的角度差
    down_by_cmd = {r["cmd"]: r["angle_deg"] for r in downs}
    hysteresis = [abs(r["angle_deg"] - down_by_cmd[r["cmd"]])
                  for r in ups if r["cmd"] in down_by_cmd]

    return {
        "motor": motor,
        "name": name,
        "name_cn": MOTOR_CN[motor],
        "pos_limit": list(hand.read_pos_limit(motor)),
        "angle_at_min_cmd": ups[0]["angle_deg"],
        "angle_at_max_cmd": max(angles, key=lambda a: abs(a - ups[0]["angle_deg"])),
        "angle_range_deg": [min(angles), max(angles)],
        "max_hysteresis_deg": max(hysteresis) if hysteresis else None,
        "max_current_ma": max(r["current_ma"] for r in records),
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motors", type=int, nargs="*", default=list(range(N_MOTORS)),
                    help="要标定的电机 index,默认全部 0-5")
    ap.add_argument("--steps", type=int, default=9,
                    help="上行采样点数(含两端),默认 9 → 步长 8192")
    ap.add_argument("--speed", type=int, default=120,
                    help="运动速度 0-255,默认 120(比全速慢,更稳)")
    ap.add_argument("--settle-tol", type=int, default=150)
    ap.add_argument("--settle-timeout", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要发的指令,不连手")
    args = ap.parse_args()

    if args.steps < 2:
        print("--steps 至少 2")
        return 2
    bad = [m for m in args.motors if not 0 <= m < N_MOTORS]
    if bad:
        print(f"电机 index 越界: {bad}")
        return 2

    plan = sweep_points(args.steps)
    print(f"计划:{len(args.motors)} 个电机 × {len(plan)} 个点位 "
          f"(上行 {args.steps} + 下行 {args.steps - 1}),速度 {args.speed}")
    print(f"指令序列:{[p for p, _ in plan]}")
    if args.dry_run:
        for m in args.motors:
            print(f"  电机 {m} {MOTORS[m]}: 其余保持 {OPEN_POSE},该路走上述序列")
        return 0

    result = {
        "device": {},
        "open_pose": OPEN_POSE,
        "speed": args.speed,
        "steps": args.steps,
        "motors": {},
    }

    with ROHand() as hand:
        result["device"] = hand.versions()
        cali = hand.read_cali_data()
        result["factory_cali"] = {
            "start_pos": cali.start_pos,
            "end_pos": cali.end_pos,
            "thumb_root_pos": cali.thumb_root_pos,
        }
        print(f"设备 {result['device']}")
        print(f"出厂标定 start={cali.start_pos} end={cali.end_pos}")

        try:
            for m in args.motors:
                result["motors"][MOTORS[m]] = calibrate_motor(
                    hand, m, args.steps, args.speed,
                    args.settle_tol, args.settle_timeout)
        except (HandError, KeyboardInterrupt) as exc:
            # 中断时务必把手停下,不要留在夹紧状态
            print(f"\n中断: {exc}")
            try:
                hand.stop()
                hand.move(OPEN_POSE, speed=args.speed)
                hand.wait_settled()
            except Exception:                   # noqa: BLE001
                pass
            if result["motors"]:
                args.out.write_text(json.dumps(result, indent=2,
                                               ensure_ascii=False), "utf-8")
                print(f"已把中断前的数据写入 {args.out}")
            return 1

    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
    print(f"\n写入 {args.out}")

    print("\n=== 汇总 ===")
    print(f"{'电机':10s} {'指令0角度':>10s} {'指令满角度':>11s} "
          f"{'行程°':>8s} {'回差°':>7s} {'峰值电流':>9s}")
    for name, d in result["motors"].items():
        lo, hi = d["angle_range_deg"]
        hy = d["max_hysteresis_deg"]
        print(f"{name:10s} {d['angle_at_min_cmd']:10.2f} "
              f"{d['angle_at_max_cmd']:11.2f} {hi - lo:8.2f} "
              f"{(hy if hy is not None else float('nan')):7.2f} "
              f"{d['max_current_ma']:7d}mA")
    return 0


if __name__ == "__main__":
    sys.exit(main())
