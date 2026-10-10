"""接触信号幅度测量 —— 测 dC,EMI 门槛判定缺的就是这个数。

EMI 实验只能告诉你噪声 sigma 有多大。但噪声大不等于不能用 —— 真正的判据是
信噪比 SNR = dC / sigma。所以必须单独测一次"真实接触能带来多大的电容变化"。

流程:

    1. 静置若干秒            -> 基线 mu 和 sigma
    2. 你反复按压/靠近电极    -> 脚本自动识别每次事件,实时报出 dC
    3. 结束后给统计           -> dC 均值、重复性、以及换算出的 SNR

降噪:平滑窗口默认取**整数个工频周期**。滑动平均在 k*fps/L 处有零点,窗口取
一个 50Hz 周期(fps/50 ≈ 49 点)就能把 50/100/150Hz 整族一起陷掉 —— 实测这
一族占总噪声功率的 58%,所以这一步很值。

用法::

    python contact_dc.py                    # 基线5s + 检测60s
    python contact_dc.py --duration 30      # 缩短检测时间
    python contact_dc.py --k 4              # 放宽触发阈值(默认6倍sigma)
    python contact_dc.py --mains 0          # 关掉工频对齐,用固定点数平滑

测完把 dC 填给 EMI 实验::

    python emi_baseline.py --contact-dc <测出来的值>
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager   # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_HAVE = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
for _f in ("Microsoft YaHei", "DengXian", "SimHei", "Microsoft JhengHei",
           "Noto Sans CJK SC", "KaiTi", "FangSong"):
    if _f in _HAVE:
        plt.rcParams["font.sans-serif"] = [_f]
        plt.rcParams["axes.unicode_minus"] = False
        break

from tactile import Tactile, TactileError  # noqa: E402

OUT_DIR = Path(__file__).parent / "contact_out"


def smooth_window(fps: float, mains_hz: float, periods: float,
                  fallback: int = 99) -> int:
    """平滑窗口点数。对齐到整数个工频周期,好把 50Hz 整族陷掉。"""
    if mains_hz <= 0 or fps <= 0:
        return fallback if fallback % 2 else fallback + 1
    n = int(round(periods * fps / mains_hz))
    n = max(3, n)
    return n if n % 2 else n + 1


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) < n:
        return x.copy()
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")


class EventDetector:
    """状态机:识别一次"按下 - 保持 - 松开"。

    进入:偏离基线超过 entry=k*sigma,持续约 min_hold 秒
    退出:回落到 exit 以内,持续约 min_gap 秒。
         exit = max(0.4*entry, 0.3*本次事件峰值偏移)

    退出阈值为什么要跟着峰值缩放:事件进行中基线是冻结的,而温漂会一直累积。
    第二版用固定的 0.4*entry(约 0.003 pF),而按压间隔里漂移能到 0.008 pF ——
    于是永远退不出来,8 次按压被连成 2 个几十秒的"事件"。让阈值随信号幅度
    放大后,0.3pF 的按压退出线是 0.09 pF,远高于漂移,就稳了。

    计数用**漏积分**而不是"连续 N 个样本":满足条件 +1,不满足 -1,夹在
    [0, N]。第一版要求连续 366 个样本全部低于阈值,噪声下概率约 3e-12。
    """

    def __init__(self, baseline: float, sigma: float, k: float,
                 fps: float, min_hold: float = 0.15, min_gap: float = 0.15):
        self.baseline = baseline
        self.sigma = sigma
        self.entry = k * sigma
        self.fps = fps
        self.min_hold = max(3, int(min_hold * fps))
        self.min_gap = max(3, int(min_gap * fps))
        self.alpha = 1.0 / max(1.0, 2.0 * fps)   # 基线 EMA,时间常数约 2s
        # 5s 太慢:跟踪滞后 tau*漂移率 已接近 entry,空闲时频繁误触发
        self._in_event = False
        self._above = 0
        self._below = 0
        self._buf: list[float] = []
        self._start_baseline = baseline
        self._start_idx = 0
        self._peak = 0.0

    def _exit_thresh(self) -> float:
        return max(0.4 * self.entry, 0.3 * self._peak)

    def feed(self, idx: int, value: float):
        """喂一个平滑后的样本。返回完成的事件 dict 或 None。"""
        dev = value - self.baseline

        if not self._in_event:
            if abs(dev) > self.entry:
                self._above = min(self.min_hold, self._above + 1)
                self._buf.append(value)
                if len(self._buf) > self.min_hold:
                    self._buf.pop(0)   # pending 阶段只保留最近 min_hold 个,
                                       # 否则噪声让 _above 反复弹跳时 _buf 会
                                       # 一直累积,_start_idx 退到按压前很久
                if self._above >= self.min_hold:
                    self._in_event = True
                    self._below = 0
                    self._start_baseline = self.baseline
                    self._start_idx = idx - len(self._buf)
                    self._peak = max(abs(v - self.baseline) for v in self._buf)
            else:
                self._above = max(0, self._above - 1)
                if self._above == 0:
                    self._buf.clear()
                self.baseline += self.alpha * dev      # 只在空闲时跟踪基线
            return None

        # 事件进行中
        self._buf.append(value)
        self._peak = max(self._peak, abs(value - self._start_baseline))
        if abs(value - self._start_baseline) < self._exit_thresh():
            self._below = min(self.min_gap, self._below + 1)
            if self._below >= self.min_gap:
                return self._finish()
        else:
            self._below = max(0, self._below - 1)
        return None

    def flush(self):
        """数据结束时收尾:把还没闭合的事件交出来(如果够长)。"""
        if self._in_event and len(self._buf) >= self.min_hold:
            return self._finish()
        return None

    def _finish(self) -> dict:
        arr = np.asarray(self._buf, dtype=np.float64)
        dev = arr - self._start_baseline
        # 掐掉尾部已经回落的部分:找最后一个仍然超过 entry 的位置
        hot = np.nonzero(np.abs(dev) > self.entry)[0]
        if len(hot):
            arr, dev = arr[: hot[-1] + 1], dev[: hot[-1] + 1]
        if len(arr) < 5:
            plateau = float(arr.mean()) if len(arr) else self._start_baseline
        else:
            # 中间 60% 的中位数当平台值,避开上升沿和释放沿
            lo, hi = int(0.2 * len(arr)), int(0.8 * len(arr))
            plateau = float(np.median(arr[lo:hi] if hi > lo else arr))
        sign = 1.0 if plateau >= self._start_baseline else -1.0
        ev = {
            "start_idx": self._start_idx,
            "end_idx": self._start_idx + len(arr),
            "n": len(arr),
            "duration": len(arr) / self.fps,
            "baseline": self._start_baseline,
            "plateau": plateau,
            "dC": plateau - self._start_baseline,
            "peak_dC": float(np.abs(dev).max()) * sign if len(dev) else 0.0,
        }
        self._in_event = False
        self._above = 0
        self._below = 0
        self._buf.clear()
        self._peak = 0.0
        # 事件结束后基线回到事件前的水平,让 EMA 从这里继续跟
        self.baseline = self._start_baseline
        return ev


def main() -> int:
    ap = argparse.ArgumentParser(description="接触信号幅度 dC 测量")
    ap.add_argument("--baseline", type=float, default=5.0, help="静置基线秒数")
    ap.add_argument("--duration", type=float, default=60.0, help="检测阶段秒数")
    ap.add_argument("--k", type=float, default=6.0, help="触发阈值(sigma 的倍数)")
    ap.add_argument("--mains", type=float, default=50.0,
                    help="工频(Hz),平滑窗口对齐到它;0 表示不对齐")
    ap.add_argument("--periods", type=float, default=1.0, help="平滑窗口占几个工频周期")
    ap.add_argument("--port", default=None, help="CH340 串口")
    ap.add_argument("--sigma-c", type=float, default=None,
                    help="条件C(运动)的 sigma,给了就直接算最终 SNR")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("接触信号幅度测量")
    print("=" * 72)

    try:
        with Tactile(port=args.port) as tac:
            print("  串口: {}".format(tac.port))
            if not tac.wait_for_data(3.0):
                raise TactileError("收不到触觉帧,先跑 python tactile.py 自检")

            # ---- 基线 ----
            print("\n[1/2] 静置基线 {:.0f} 秒 —— **不要碰电极,手也离远一点**"
                  .format(args.baseline))
            input("      准备好了按回车 ")
            base_rec = tac.record(args.baseline, "baseline")
            fps = base_rec.fps
            win = smooth_window(fps, args.mains, args.periods)
            base_pf = base_rec.pf
            base_sm = moving_average(base_pf, win)
            edge = win
            base_sm_trim = base_sm[edge:-edge] if len(base_sm) > 2 * edge else base_sm
            mu = float(np.median(base_sm_trim))
            sigma_raw = float(base_rec.detrend().std())
            sigma_sm = float((base_sm_trim - np.median(base_sm_trim)).std())

            print("      帧率        : {:.0f} fps".format(fps))
            print("      平滑窗口    : {} 点 = {:.1f} ms = {:.2f} 个工频周期"
                  .format(win, win / fps * 1000, win * args.mains / fps
                          if args.mains > 0 else 0))
            print("      基线        : {:.4f} pF".format(mu))
            print("      sigma 原始  : {:.5f} pF".format(sigma_raw))
            print("      sigma 平滑后: {:.5f} pF   (降噪 {:.1f} 倍)"
                  .format(sigma_sm, sigma_raw / sigma_sm if sigma_sm > 0 else 0))
            print("      触发阈值    : {:.5f} pF  ({:.0f} sigma)"
                  .format(args.k * sigma_sm, args.k))

            # ---- 检测 ----
            print("\n[2/2] 开始检测 {:.0f} 秒 —— **反复按压/靠近电极,每次之间完全松开**"
                  .format(args.duration))
            print("      建议至少做 8~10 次,力度尽量一致")
            input("      准备好了按回车 ")

            det = EventDetector(mu, sigma_sm, args.k, fps)
            events = []
            tac.start()
            t0 = time.perf_counter()
            from tactile import raw_to_capacitance_pf
            half = win // 2
            buf = np.empty(0, dtype=np.float64)   # 全部已收样本(pF)
            emitted = 0                            # 已喂给状态机的样本数
            while time.perf_counter() - t0 < args.duration:
                time.sleep(0.15)
                with tac._lock:                    # 取增量
                    raw_now = np.asarray(tac._raw[len(buf):], dtype=np.uint32)
                if len(raw_now):
                    buf = np.concatenate([buf, raw_to_capacitance_pf(raw_now)])

                # 滑动平均 mode='same' 的有效区间是 [half, len-half)。
                # 每轮只喂"平滑值已经稳定"的那一段,并带上 half 的上下文
                # 重算,避免分块边界漏样本(第一版就漏了 win//2 个/轮)。
                avail_end = len(buf) - half
                if avail_end > emitted:
                    lo = max(0, emitted - half)
                    seg = buf[lo:min(len(buf), avail_end + half)]
                    if len(seg) >= win:
                        sm_seg = moving_average(seg, win)
                        for j in range(emitted - lo, emitted - lo + (avail_end - emitted)):
                            if j >= len(sm_seg):
                                break
                            ev = det.feed(lo + j, float(sm_seg[j]))
                            if ev is not None:
                                events.append(ev)
                                print("\r      #{:<3d} dC = {:+.4f} pF   持续 {:.2f}s   "
                                      "基线 {:.4f}        ".format(
                                          len(events), ev["dC"], ev["duration"],
                                          ev["baseline"]))
                        emitted = avail_end
                left = args.duration - (time.perf_counter() - t0)
                print("\r      剩余 {:4.1f}s   已识别 {} 次      "
                      .format(left, len(events)), end="", flush=True)
            rec = tac.stop("contact")
            tail = det.flush()          # 最后一次按压可能还没闭合
            if tail is not None:
                events.append(tail)
            print("\r      检测结束,共识别 {} 次事件            ".format(len(events)))

    except TactileError as e:
        print("\n  x {}".format(e))
        return 1
    except KeyboardInterrupt:
        print("\n  中断。")
        return 130

    if not events:
        print("\n  没有识别到任何事件。可能原因:")
        print("    - 按压幅度小于触发阈值 {:.5f} pF -> 试试 --k 3".format(
            args.k * sigma_sm))
        print("    - 电极没接好,或按的位置不在电极上")
        print("    - 每次按压持续时间太短(需要 >0.15s)")
        return 1

    # ---- 统计 ----
    dCs = np.array([e["dC"] for e in events])
    durations = np.array([e["duration"] for e in events])
    mean_dC, std_dC = float(dCs.mean()), float(dCs.std(ddof=1)) if len(dCs) > 1 else 0.0
    cv = abs(std_dC / mean_dC) * 100 if mean_dC != 0 else float("nan")

    lines = []
    w = lines.append
    w("=" * 72)
    w("接触信号幅度报告        {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    w("=" * 72)
    w("")
    w("  帧率 {:.0f} fps   平滑 {} 点({:.1f}ms)   基线 sigma(平滑后) {:.5f} pF"
      .format(fps, win, win / fps * 1000, sigma_sm))
    w("")
    w("  {:>4s}{:>12s}{:>12s}{:>12s}".format("#", "dC(pF)", "峰值(pF)", "持续(s)"))
    w("  " + "-" * 40)
    for i, e in enumerate(events, 1):
        w("  {:>4d}{:>12.4f}{:>12.4f}{:>12.2f}".format(
            i, e["dC"], e["peak_dC"], e["duration"]))
    w("  " + "-" * 40)
    w("  次数      : {}".format(len(events)))
    w("  dC 均值   : {:+.4f} pF".format(mean_dC))
    w("  dC 标准差 : {:.4f} pF".format(std_dC))
    w("  重复性 CV : {:.1f} %".format(cv))
    w("  dC 范围   : {:+.4f} ~ {:+.4f} pF".format(dCs.min(), dCs.max()))
    w("  持续时间  : {:.2f} ~ {:.2f} s".format(durations.min(), durations.max()))
    w("")
    w("  方向: 接触使电容{}".format("增大" if mean_dC > 0 else "减小"))
    w("")
    w("=" * 72)
    w("拿去做 EMI 判定")
    w("=" * 72)
    snr_base = abs(mean_dC) / sigma_sm if sigma_sm > 0 else float("inf")
    w("  相对静置基线的 SNR = {:.1f}   (dC / sigma_基线平滑后)".format(snr_base))
    if args.sigma_c is not None:
        snr_c = abs(mean_dC) / args.sigma_c if args.sigma_c > 0 else float("inf")
        w("  相对条件C 的 SNR   = {:.1f}   (dC / {:.5f})".format(snr_c, args.sigma_c))
        w("")
        if snr_c >= 10:
            w("  ==> 通过。")
        elif snr_c >= 3:
            w("  ==> 勉强,建议先降噪。")
        else:
            w("  ==> 不通过,必须先解决 EMI。")
    else:
        w("  注意:上面这个 SNR 是对**静置**基线算的,偏乐观。")
        w("  真正的判据要用条件C(手在运动)的 sigma。下一步:")
        w("")
        w("      python emi_baseline.py --contact-dc {:.4f}".format(abs(mean_dC)))
        w("")
        w("  它会自动算出运动时的 SNR 并给判定。")
    w("=" * 72)
    report = "\n".join(lines)
    print("\n" + report)

    # ---- 保存 ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez_compressed(out_dir / "contact_{}.npz".format(stamp),
                        t=rec.t, raw=rec.raw,
                        base_t=base_rec.t, base_raw=base_rec.raw,
                        dC=dCs, durations=durations,
                        fps=fps, win=win, sigma_sm=sigma_sm, mu=mu)
    (out_dir / "contact_{}.txt".format(stamp)).write_text(report, encoding="utf-8")

    try:
        pf = rec.pf
        sm = moving_average(pf, win)
        t = np.arange(len(sm)) / fps
        fig, ax = plt.subplots(2, 1, figsize=(11, 7))
        ax[0].plot(t, pf - mu, lw=0.3, color="#bbb", label="原始")
        ax[0].plot(t, sm - mu, lw=0.8, color="#1f77b4", label="平滑 {}点".format(win))
        ax[0].axhline(0, color="k", lw=0.5)
        for s in (+1, -1):
            ax[0].axhline(s * args.k * sigma_sm, color="r", ls="--", lw=0.6)
        for i, e in enumerate(events, 1):
            ax[0].axvspan(e["start_idx"] / fps, e["end_idx"] / fps,
                          color="orange", alpha=0.18)
        ax[0].set_ylabel("dC (pF)")
        ax[0].set_xlabel("时间 (s)")
        ax[0].set_title("接触事件识别 — 共 {} 次,dC = {:+.4f} ± {:.4f} pF"
                        .format(len(events), mean_dC, std_dC))
        ax[0].legend(fontsize=8)
        ax[1].bar(range(1, len(dCs) + 1), dCs, color="#1f77b4")
        ax[1].axhline(mean_dC, color="r", ls="--", lw=1,
                      label="均值 {:+.4f}".format(mean_dC))
        ax[1].set_xlabel("第几次按压")
        ax[1].set_ylabel("dC (pF)")
        ax[1].set_title("重复性  CV = {:.1f}%".format(cv))
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "contact_{}.png".format(stamp), dpi=130)
        plt.close(fig)
    except Exception as e:
        print("  (画图失败: {})".format(e))

    print("\n  已保存到 {}".format(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
