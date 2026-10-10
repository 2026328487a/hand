"""电机 EMI 基线实验 —— 灵巧手触觉项目的**一票否决门槛**。

问题:要测的是 fF 量级的电容变化,而传感器要贴在一根内部有电机、单路电流
能到 1168mA、PWM 在高速开关的手指上。先搞清楚 —— 手一动,读数会不会被
噪声淹掉?

四个条件(逐级加重):

    A  手完全断电,传感器静止        本底噪声,标尺
    B  手上电但不动                 静态保持电流的影响
    C  手指空载往复运动,不碰东西     ★关键:实际工作状态
    D  拇指对食指顶死(~1000mA)       ★最恶劣,可跳过

判据:看**高通后**的 sigma(慢漂移是温漂,不算 EMI)。

注意 —— sigma_C / sigma_A 这个比值**不是合格判据**,它只说明电机干扰有多大。
噪声涨了 20 倍,只要接触信号比噪声大得多,照样能用。真正的判据是信噪比:

    SNR = dC_接触 / sigma_C   >= 10   通过
                              3 ~ 10  勉强,建议先降噪
                              <  3    不通过,必须先解决 EMI

所以跑完本脚本后还要单独测一次"真实接触能带来多大 dC",再用 --contact-dc
重跑就能自动判定。只跑本脚本只能得到 sigma,得不出结论。

用法::

    python emi_baseline.py                  # 完整四条件,每条 30 秒
    python emi_baseline.py --seconds 15     # 缩短
    python emi_baseline.py --skip-jam       # 跳过顶死(保护机械)
    python emi_baseline.py --only A         # 只跑某一条件
    python emi_baseline.py --contact-dc 0.5 # 已知接触能带来 0.5pF,自动判定

条件 A/B 需要你手动断电和上电,脚本会停下来等回车。C/D 由脚本驱动。

输出到 emi_out/:  .npz 原始数据  .png 图  .txt 报告
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")            # 只存图,不开窗口
import matplotlib.font_manager   # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# 图里的条件名是中文,没有 CJK 字体会全变成方框。挑一个系统里装了的。
_HAVE = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
for _f in ("Microsoft YaHei", "DengXian", "SimHei", "Microsoft JhengHei",
           "Noto Sans CJK SC", "Source Han Sans SC", "KaiTi", "FangSong"):
    if _f in _HAVE:
        plt.rcParams["font.sans-serif"] = [_f]
        plt.rcParams["axes.unicode_minus"] = False
        break

from tactile import Tactile, TactileError  # noqa: E402

OUT_DIR = Path(__file__).parent / "emi_out"

# 条件 C 的往复运动:四指在 0.0 <-> 0.6 之间来回,拇指保持张开避免自碰。
SWEEP_OPEN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
SWEEP_CLOSE = [0.0, 0.6, 0.6, 0.6, 0.6, 0.0]
# 条件 D 的顶死姿态:拇指旋转到对掌 + 拇指弯曲 + 食指闭合,空捏。
JAM_POSE = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------- 分析

def analyse(rec, hp_seconds: float = 0.5) -> dict:
    """算出一段采集的噪声指标。"""
    pf = rec.pf
    hp = rec.detrend(hp_seconds)
    out = {
        "label": rec.label,
        "n": len(rec),
        "duration": rec.duration,
        "fps": rec.fps,
        "mean_pf": float(pf.mean()) if len(pf) else float("nan"),
        "sigma_raw": float(pf.std()) if len(pf) else float("nan"),
        "sigma_hp": float(hp.std()) if len(hp) else float("nan"),
        "p2p_hp": float(hp.max() - hp.min()) if len(hp) else float("nan"),
        "discarded": rec.discarded_bytes,
    }
    out["peaks"] = spectrum_peaks(hp, rec.fps)
    return out


def spectrum_peaks(signal: np.ndarray, fps: float, top: int = 4,
                   min_sep_hz: float = 5.0):
    """找高通信号里最突出的几个频率成分(电机 PWM 基频/谐波会在这里现形)。

    同一个峰会占好几个相邻 FFT bin,所以要按 min_sep_hz 去重,否则输出四行
    全是同一个频率。
    """
    if len(signal) < 64 or not np.isfinite(fps) or fps <= 0:
        return []
    n = len(signal)
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(signal * window)) / n
    freq = np.fft.rfftfreq(n, d=1.0 / fps)
    spec[0] = 0.0                      # 扔掉直流
    keep = freq > 1.0                  # 1Hz 以下算漂移,不算 EMI
    if not keep.any():
        return []
    spec, freq = spec[keep], freq[keep]
    picked = []
    for i in np.argsort(spec)[::-1]:
        f = float(freq[i])
        if any(abs(f - pf) < min_sep_hz for pf, _ in picked):
            continue
        picked.append((f, float(spec[i])))
        if len(picked) >= top:
            break
    return picked


def spectrum(signal: np.ndarray, fps: float):
    """返回 (freq, amp) 供画图用。"""
    if len(signal) < 64 or fps <= 0:
        return np.array([]), np.array([])
    n = len(signal)
    spec = np.abs(np.fft.rfft(signal * np.hanning(n))) / n
    freq = np.fft.rfftfreq(n, d=1.0 / fps)
    spec[0] = 0.0
    return freq, spec


# ---------------------------------------------------------------- 采集

def ask(prompt: str) -> None:
    try:
        input("\n  >> {} ,然后按回车 ".format(prompt))
    except (EOFError, KeyboardInterrupt):
        print("\n  中断。")
        raise SystemExit(130)


def cond_a(tac: Tactile, seconds: float):
    print("\n[A] 本底噪声 —— 手完全断电")
    ask("请**切断手的电源**(拔 24V 或拔 PCAN)")
    if not tac.wait_for_data(2.0):
        raise TactileError("收不到触觉帧,先跑 python tactile.py 自检")
    return tac.record(seconds, "A 手断电")


def cond_b(hand, seconds: float, tac: Tactile):
    print("\n[B] 静态上电 —— 手上电但不动")
    currents = hand.read_currents()
    print("    静止电流: {} mA".format(currents))
    return tac.record(seconds, "B 上电静止")


def cond_c(hand, cal, seconds: float, tac: Tactile):
    """四指空载往复,同时记录电流范围。"""
    print("\n[C] 空载运动 —— 四指往复,不接触任何东西  ★关键条件")
    print("    确保手周围没有障碍物,手指能自由张合")
    ask("准备好了")
    open_pos = cal.pose_from_norm(SWEEP_OPEN)
    close_pos = cal.pose_from_norm(SWEEP_CLOSE)

    tac.start()
    t0 = time.perf_counter()
    peak_current = 0
    cycles = 0
    target_close = True
    while time.perf_counter() - t0 < seconds:
        hand.move(close_pos if target_close else open_pos, speed=150)
        target_close = not target_close
        # 边运动边采电流,顺便给出这次实验的电流区间
        deadline = time.perf_counter() + 1.2
        while time.perf_counter() < deadline and time.perf_counter() - t0 < seconds:
            try:
                peak_current = max(peak_current, max(hand.read_currents()))
            except Exception:
                pass
            left = seconds - (time.perf_counter() - t0)
            print("\r    运动中 {:4.1f}s  峰值电流 {:4d} mA  往复 {:2d} 次"
                  .format(left, peak_current, cycles), end="", flush=True)
            time.sleep(0.05)
        cycles += 1
    rec = tac.stop("C 空载运动")
    hand.move(open_pos, speed=120)
    print("\r    完成 {:4.1f}s  {:6d} 帧  峰值电流 {} mA  往复 {} 次        "
          .format(rec.duration, len(rec), peak_current, cycles))
    rec.meta["peak_current_ma"] = peak_current
    rec.meta["cycles"] = cycles
    return rec


def cond_d(hand, cal, seconds: float, tac: Tactile):
    """拇指对食指空捏顶死。电流会到 ~1000mA,时间必须短。"""
    print("\n[D] 顶死 —— 拇指对食指空捏  ★最恶劣条件")
    print("    警告:这会让电流升到 850~1170mA。硬件自己会限流不会烧,")
    print("    但长期顶死伤机械,所以本条件固定只跑 {:.0f} 秒。".format(seconds))
    ans = input("\n  >> 确认执行?(y/N) ").strip().lower()
    if ans != "y":
        print("    已跳过。")
        return None
    jam_pos = cal.pose_from_norm(JAM_POSE)
    open_pos = cal.pose_from_norm(SWEEP_OPEN)

    hand.move(jam_pos, speed=120)
    time.sleep(1.5)                     # 等顶上
    tac.start()
    t0 = time.perf_counter()
    peak_current = 0
    while time.perf_counter() - t0 < seconds:
        try:
            peak_current = max(peak_current, max(hand.read_currents()))
        except Exception:
            pass
        left = seconds - (time.perf_counter() - t0)
        print("\r    顶死中 {:4.1f}s  峰值电流 {:4d} mA"
              .format(left, peak_current), end="", flush=True)
        time.sleep(0.05)
    rec = tac.stop("D 顶死")
    hand.move(open_pos, speed=120)      # 立刻松开
    print("\r    完成 {:4.1f}s  {:6d} 帧  峰值电流 {} mA            "
          .format(rec.duration, len(rec), peak_current))
    rec.meta["peak_current_ma"] = peak_current
    return rec


# ---------------------------------------------------------------- 报告

def build_report(results, contact_dc) -> str:
    lines = []
    w = lines.append
    w("=" * 74)
    w("电机 EMI 基线实验报告        {}".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    w("=" * 74)
    w("")
    w("{:<14s}{:>7s}{:>8s}{:>11s}{:>11s}{:>11s}".format(
        "条件", "帧数", "fps", "均值(pF)", "sigma_hp", "峰峰(pF)"))
    w("-" * 74)
    for r in results:
        w("{:<14s}{:>7d}{:>8.0f}{:>11.3f}{:>11.5f}{:>11.5f}".format(
            r["label"], r["n"], r["fps"], r["mean_pf"], r["sigma_hp"], r["p2p_hp"]))
    w("-" * 74)
    w("sigma_hp = 高通后标准差(扣掉慢漂移),这才是 EMI 噪声")
    w("")

    by = {r["label"][0]: r for r in results}
    base = by.get("A")
    w("频谱主峰(高通信号,>1Hz):")
    for r in results:
        if r["peaks"]:
            txt = "  ".join("{:.0f}Hz({:.2e})".format(f, a) for f, a in r["peaks"])
        else:
            txt = "(无显著峰)"
        w("  {:<14s}{}".format(r["label"], txt))
    w("")

    w("=" * 74)
    w("判定")
    w("=" * 74)
    if base is None:
        w("  缺少条件 A,无法给出比值判定。请完整跑一遍。")
        return "\n".join(lines)

    sa = base["sigma_hp"]
    w("  本底 sigma_A = {:.5f} pF".format(sa))
    w("")
    w("  各条件的干扰强度(相对本底,**这不是合格判据**,只说明电机干扰有多大):")
    for key, name in (("B", "上电静止"), ("C", "空载运动"), ("D", "顶死")):
        r = by.get(key)
        if r is None:
            continue
        ratio = r["sigma_hp"] / sa if sa > 0 else float("inf")
        level = "轻微" if ratio < 2 else ("明显" if ratio < 10 else "强烈")
        w("    {:<10s} sigma = {:.5f} pF   = {:6.1f} x 本底   {}"
          .format(name, r["sigma_hp"], ratio, level))
    w("")

    c = by.get("C")
    if c is None:
        w("  缺少条件 C,无法判定。条件 C(空载运动)才是实际工作状态。")
        w("=" * 74)
        return "\n".join(lines)

    w("  --- 合格判定(以条件 C 为准,那才是实际工作状态) ---")
    sc = c["sigma_hp"]
    if contact_dc is None:
        w("  还不能判定 —— 噪声大小本身说明不了问题,要和**接触信号**比才有意义。")
        w("")
        w("  下一步:测一次真实接触能带来多大的 dC(拿物体压电极,看电容跳多少),")
        w("  然后按这个标准看:")
        w("    dC >= {:.4f} pF  (SNR>=10)  ->  通过".format(sc * 10))
        w("    dC >= {:.4f} pF  (SNR>=3 )  ->  勉强,建议先降噪".format(sc * 3))
        w("    dC <  {:.4f} pF             ->  不通过,必须先解决 EMI".format(sc * 3))
        w("")
        w("  测出来后用 --contact-dc <pF> 重跑本脚本,会自动判定。")
        need_fix = None
    else:
        snr = contact_dc / sc if sc > 0 else float("inf")
        w("  已知接触信号 dC = {:.4f} pF".format(contact_dc))
        w("  运动时信噪比 SNR = dC / sigma_C = {:.1f}".format(snr))
        w("")
        if snr >= 10:
            w("  ==> 通过。接触信号远大于运动噪声,可以往下走。")
            need_fix = False
        elif snr >= 3:
            w("  ==> 勉强。能用但不稳,建议先降噪把 SNR 提到 10 以上。")
            need_fix = True
        else:
            w("  ==> 不通过。接触信号被电机噪声淹没,必须先解决 EMI。")
            need_fix = True

    if need_fix is not False:
        w("")
        w("  {}降噪按这个顺序试:".format("" if need_fix else "如果后面判定不通过,"))
        w("    1. 主动屏蔽层驱动(跟随感测电位,不是简单接地)")
        w("    2. 传感器地与电机地单点共地,走线避开电机线束")
        w("    3. 缩短 SCL/SDA,加磁环")
        w("    4. 提高 RCOUNT(牺牲速度换分辨率)")
        w("    5. 采样与电机 PWM 同步,在关断相采样")
    w("=" * 74)
    return "\n".join(lines)


def plot(recs, results, path: Path) -> None:
    n = len(recs)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7), squeeze=False)
    for i, (rec, res) in enumerate(zip(recs, results)):
        hp = rec.detrend()
        t = np.arange(len(hp)) / rec.fps if rec.fps > 0 else np.arange(len(hp))
        ax = axes[0][i]
        ax.plot(t, hp, lw=0.4)
        ax.set_title("{}\nsigma={:.5f} pF".format(rec.label, res["sigma_hp"]),
                     fontsize=9)
        ax.set_xlabel("time (s)", fontsize=8)
        if i == 0:
            ax.set_ylabel("high-passed dC (pF)", fontsize=8)
        ax.tick_params(labelsize=7)

        ax = axes[1][i]
        freq, spec = spectrum(hp, rec.fps)
        if len(freq):
            ax.semilogy(freq, spec, lw=0.5)
            ax.set_xlim(0, min(500, freq.max()))
        ax.set_xlabel("freq (Hz)", fontsize=8)
        if i == 0:
            ax.set_ylabel("amplitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    # 统一纵轴,让四个条件肉眼可比
    ymax = max((abs(a.get_ylim()[0]), abs(a.get_ylim()[1])) for a in axes[0])
    lim = max(ymax)
    for a in axes[0]:
        a.set_ylim(-lim, lim)
    fig.suptitle("Motor EMI baseline - tactile noise vs hand state", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="电机 EMI 基线实验")
    ap.add_argument("--seconds", type=float, default=30.0, help="每条件采集秒数")
    ap.add_argument("--jam-seconds", type=float, default=8.0, help="顶死条件秒数")
    ap.add_argument("--port", default=None, help="CH340 串口,默认自动找")
    ap.add_argument("--skip-jam", action="store_true", help="跳过条件 D")
    ap.add_argument("--only", default=None, help="只跑某些条件,如 A 或 AC")
    ap.add_argument("--contact-dc", type=float, default=None,
                    help="已知接触带来的 dC(pF),给了就自动判定通过与否")
    ap.add_argument("--out", default=str(OUT_DIR), help="输出目录")
    args = ap.parse_args()

    want = set((args.only or "ABCD").upper())
    if args.skip_jam:
        want.discard("D")
    need_hand = bool(want & set("BCD"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 74)
    print("电机 EMI 基线实验")
    print("=" * 74)
    print("  条件      : {}".format(" ".join(sorted(want))))
    print("  每条时长  : {:.0f} s  (顶死 {:.0f} s)".format(args.seconds, args.jam_seconds))
    print("  输出       : {}".format(out_dir))

    recs, results = [], []
    try:
        with Tactile(port=args.port) as tac:
            print("  触觉串口  : {}".format(tac.port))

            if "A" in want:
                recs.append(cond_a(tac, args.seconds))

            if need_hand:
                print("\n[连接手]")
                ask("请给手**上电**并接好 PCAN")
                from calibration import Calibration
                from rohand import ROHand
                cal = Calibration.load()
                with ROHand() as hand:
                    print("    已连接。固件 {}".format(
                        hand.versions().get("firmware", "?")))
                    if "B" in want:
                        recs.append(cond_b(hand, args.seconds, tac))
                    if "C" in want:
                        recs.append(cond_c(hand, cal, args.seconds, tac))
                    if "D" in want:
                        r = cond_d(hand, cal, args.jam_seconds, tac)
                        if r is not None:
                            recs.append(r)
    except TactileError as e:
        print("\n  x 触觉链路: {}".format(e))
        return 1
    except KeyboardInterrupt:
        print("\n  中断,已采集的数据仍会保存。")
    except Exception as e:
        print("\n  x {}: {}".format(e.__class__.__name__, e))
        if not recs:
            return 1
        print("    已采集的数据仍会保存。")

    if not recs:
        print("\n  没有采到任何数据。")
        return 1

    results = [analyse(r) for r in recs]
    report = build_report(results, args.contact_dc)
    print("\n" + report)

    npz_path = out_dir / "emi_{}.npz".format(stamp)
    payload = {}
    for i, r in enumerate(recs):
        payload["t_{}".format(i)] = r.t
        payload["raw_{}".format(i)] = r.raw
        payload["label_{}".format(i)] = np.array(r.label)
    np.savez_compressed(npz_path, **payload)

    txt_path = out_dir / "emi_{}.txt".format(stamp)
    txt_path.write_text(report, encoding="utf-8")

    png_path = out_dir / "emi_{}.png".format(stamp)
    try:
        plot(recs, results, png_path)
    except Exception as e:
        print("  (画图失败: {})".format(e))
        png_path = None

    print("\n  已保存:")
    print("    {}".format(npz_path))
    print("    {}".format(txt_path))
    if png_path:
        print("    {}".format(png_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
