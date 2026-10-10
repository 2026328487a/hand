"""ROH-LiteS001 手势库。

手势用**归一化量**定义(每路 0=张开 1=完全闭合),再由 calibration.py 换算成
指令位置。这样手势定义与标定数据解耦 —— 换一只手重新标定后,同一份手势
不用改。

为什么手势是**多步**的:六个自由度里拇指旋转会把拇指扫过掌心,如果四指已经
伸直摆在那里就会撞上。所以像握拳这类动作必须先收四指、再转拇指、最后弯
拇指。每个 Gesture 是一串 Step,而不是单个姿态。

用法::

    python gestures.py --list                 # 列出所有手势
    python gestures.py fist                   # 做一个
    python gestures.py open fist point open   # 依次做
    python gestures.py --demo                 # 跑一遍全部
    python gestures.py fist --dry-run         # 只算位置不动手

也可以当模块用::

    from gestures import GESTURES, play
    with ROHand() as h:
        play(h, cal, GESTURES["ok"])
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

from calibration import Calibration
from rohand import MOTORS, MOTOR_CN, N_MOTORS, HandError, ROHand

# 归一化姿态里的通道顺序,和 rohand.MOTORS 一致:
#   0 thumb_bend  1 index  2 middle  3 ring  4 little  5 thumb_rot
TB, IX, MD, RG, LT, TR = range(6)

# 电流阈值。标定实测正常运动 <180mA,顶到限位或夹到东西会明显高于此值。
CURRENT_WARN_MA = 400


@dataclass
class Step:
    """一个中间姿态。norms 里 None 表示"这一路保持不动"。"""
    norms: list
    speed: int = 160
    dwell: float = 0.0          # 到位后额外停留(秒),给观察留时间
    settle: bool = True         # 是否等到位再走下一步
    stop_on_current: int = 0    # >0 则分段逼近,电流超此值(mA)就停 —— 见
                                # ROHand.approach。捏合/抓握用,空载会顶死。


@dataclass
class Gesture:
    name: str
    name_cn: str
    desc: str
    steps: list = field(default_factory=list)


def _pose(**kw) -> list:
    """写姿态的糖:_pose(index=1, thumb_rot=0.5) → [None,1,None,None,None,0.5]"""
    names = {"thumb_bend": TB, "index": IX, "middle": MD, "ring": RG,
             "little": LT, "thumb_rot": TR}
    out = [None] * N_MOTORS
    for k, v in kw.items():
        if k == "fingers":                  # 四指一起给
            for i in (IX, MD, RG, LT):
                out[i] = v
        elif k == "all":
            out = [v] * N_MOTORS
        else:
            out[names[k]] = v
    return out


OPEN = _pose(all=0.0)

GESTURES = {}


def _add(g: Gesture):
    GESTURES[g.name] = g
    return g


_add(Gesture("open", "张开", "全部张开,安全姿态", [
    # 先松拇指再张四指:拇指若压在四指上,直接张四指会顶到拇指
    Step(_pose(thumb_bend=0.0), speed=140),
    Step(_pose(thumb_rot=0.0, fingers=0.0), speed=140, dwell=0.2),
]))

_add(Gesture("fist", "握拳", "四指先收,拇指再转过来压上", [
    Step(_pose(thumb_bend=0.0, thumb_rot=0.0), speed=160),
    Step(_pose(fingers=1.0), speed=160),
    Step(_pose(thumb_rot=0.85), speed=140),      # 拇指转到掌心上方
    Step(_pose(thumb_bend=0.75), speed=140, dwell=0.3),
]))

_add(Gesture("point", "指点", "只伸食指", [
    Step(_pose(thumb_rot=0.0), speed=160),
    Step(_pose(index=0.0, middle=1.0, ring=1.0, little=1.0), speed=160),
    Step(_pose(thumb_rot=0.8), speed=140),
    Step(_pose(thumb_bend=0.7), speed=140, dwell=0.3),
]))

_add(Gesture("victory", "剪刀", "食指中指伸直成 V", [
    Step(_pose(thumb_rot=0.0), speed=160),
    Step(_pose(index=0.0, middle=0.0, ring=1.0, little=1.0), speed=160),
    Step(_pose(thumb_rot=0.8), speed=140),
    Step(_pose(thumb_bend=0.7), speed=140, dwell=0.3),
]))

_add(Gesture("thumbs_up", "点赞", "四指收拢,拇指竖起", [
    Step(_pose(thumb_bend=0.0, thumb_rot=0.0), speed=160),
    Step(_pose(fingers=1.0), speed=160, dwell=0.3),
]))

_add(Gesture("pinch", "捏合", "拇指与食指对捏", [
    Step(_pose(fingers=0.0, thumb_bend=0.0), speed=160),
    Step(_pose(thumb_rot=0.62), speed=120),      # 先转到对掌位再合拢
    # 空捏时拇指会顶到食指上，实测 852mA。改成碰到就停，
    # 中间夹东西时也能自然停在物体上。
    Step(_pose(index=0.62, thumb_bend=0.72), speed=90, dwell=0.4,
         stop_on_current=350),
]))

_add(Gesture("ok", "OK", "拇指食指成环,其余三指伸直", [
    Step(_pose(all=0.0), speed=160),
    Step(_pose(thumb_rot=0.62), speed=120),
    Step(_pose(index=0.68, thumb_bend=0.75), speed=90, dwell=0.4,
         stop_on_current=350),
]))

_add(Gesture("rock", "摇滚", "食指小指伸出", [
    Step(_pose(thumb_rot=0.0), speed=160),
    Step(_pose(index=0.0, middle=1.0, ring=1.0, little=0.0), speed=160),
    Step(_pose(thumb_rot=0.8), speed=140),
    Step(_pose(thumb_bend=0.7), speed=140, dwell=0.3),
]))

_add(Gesture("grip", "抓握", "四指同步半握,拇指对掌 —— 抓圆柱物体用", [
    Step(_pose(thumb_bend=0.0), speed=160),
    Step(_pose(thumb_rot=0.55), speed=130),
    Step(_pose(fingers=0.85, thumb_bend=0.7), speed=90, dwell=0.4,
         stop_on_current=350),
]))

_add(Gesture("wave", "逐指波浪", "四指依次弯曲再依次张开,看单指独立性", [
    Step(_pose(all=0.0), speed=180),
    Step(_pose(index=0.9), speed=200, dwell=0.12),
    Step(_pose(middle=0.9), speed=200, dwell=0.12),
    Step(_pose(ring=0.9), speed=200, dwell=0.12),
    Step(_pose(little=0.9), speed=200, dwell=0.12),
    Step(_pose(index=0.0), speed=200, dwell=0.12),
    Step(_pose(middle=0.0), speed=200, dwell=0.12),
    Step(_pose(ring=0.0), speed=200, dwell=0.12),
    Step(_pose(little=0.0), speed=200, dwell=0.2),
]))

# 数数 1-4:伸出的手指张开(0.0)、其余收拢(1.0),拇指压在收拢的手指上。
# 第 n 个手势伸出前 n 根手指,所以第 i 根的目标 = 0.0 if i < n else 1.0。
for n in range(1, 5):
    _add(Gesture(f"count{n}", f"数字{n}", f"伸 {n} 根手指", [
        Step(_pose(thumb_rot=0.0), speed=170),
        Step([None,
              0.0 if 0 < n else 1.0,          # index
              0.0 if 1 < n else 1.0,          # middle
              0.0 if 2 < n else 1.0,          # ring
              0.0 if 3 < n else 1.0,          # little
              None], speed=170),
        Step(_pose(thumb_rot=0.8), speed=150),
        Step(_pose(thumb_bend=0.7), speed=150, dwell=0.3),
    ]))


def resolve(step: Step, prev_norms: list) -> list:
    """把 None(保持不动)填成上一步的值。"""
    return [prev_norms[i] if v is None else v for i, v in enumerate(step.norms)]


# 拇指旋转会把拇指扫过掌心。如果此时拇指还是**弯着**的(压在四指上),转动就是
# 蹭着四指侧面硬扫过去 —— 实测 thumb_rot 电流冲到 434mA,而空载只有 0-172mA。
# 这是机械约束不是手势设计问题,所以在 play() 里统一处理,不写进每个手势。
THUMB_BEND_SAFE = 0.15      # 低于此值认为拇指已让开
THUMB_ROT_MOVE = 0.10       # 旋转量超过此值才需要先让开


def _thumb_clearance_step(cur_norms: list, target_norms: list):
    """需要先松拇指的话,返回那一步;否则 None。"""
    if cur_norms[TB] <= THUMB_BEND_SAFE:
        return None
    if abs(target_norms[TR] - cur_norms[TR]) < THUMB_ROT_MOVE:
        return None
    return Step(_pose(thumb_bend=0.0), speed=150)


def play(hand: ROHand, cal: Calibration, gesture: Gesture,
         prev_norms=None, verbose: bool = True, check_current: bool = True,
         on_step=None):
    """执行一个手势,返回结束时的归一化姿态。

    on_step 可选回调,每完成一步调用一次:on_step(i, total, norms, pose,
    contact)。GUI 用它拿进度;命令行用 verbose 打印时传 None。
    """
    norms = list(prev_norms) if prev_norms else [0.0] * N_MOTORS
    total = len(gesture.steps)
    if verbose:
        print(f"\n▶ {gesture.name} ({gesture.name_cn}) — {gesture.desc}")

    steps = list(gesture.steps)
    if steps:
        clearance = _thumb_clearance_step(norms, resolve(steps[0], norms))
        if clearance is not None:
            steps.insert(0, clearance)
            total += 1
            if verbose:
                print("   (插入让位步:先松开拇指再转)")

    for i, step in enumerate(steps, 1):
        norms = resolve(step, norms)
        pose = cal.pose_from_norm(norms)
        contact = None
        touched = False
        if step.stop_on_current:
            touched, stopped_at, hot = hand.approach(
                pose, step.stop_on_current, speed=step.speed)
            if touched:
                contact = hot
                # 停在接触点上，后续步骤要从这里接，所以把 norm 拉回实际位置
                norms = [cal.norm_from_pos(j, p)
                         for j, p in enumerate(stopped_at)]
        else:
            hand.move(pose, speed=step.speed)
            if step.settle:
                hand.wait_settled(tol=200, timeout=3.0)
        if step.dwell:
            time.sleep(step.dwell)
        if on_step is not None:
            on_step(i, total, list(norms), list(pose),
                    ("contact" if touched else None))
        if verbose:
            shown = " ".join(f"{v:.2f}" for v in norms)
            tag = ""
            if step.stop_on_current:
                tag = (f"  [接触停住 {contact}]" if contact
                       else "  [未碰到，走完全程]")
            print(f"   步骤{i}  norm[{shown}]  →  pos{pose}{tag}")
        if check_current:
            hot = [(MOTORS[j], c) for j, c in enumerate(hand.read_currents())
                   if c > CURRENT_WARN_MA]
            if hot:
                # 不自动停 —— 捏合/抓握时高电流就是接触力本身,是想要的。
                # 出厂电流上限 1199mA/路,硬件自己会限,这里只做提示。
                print(f"   ⚠ 电流偏高: {hot}")
    return norms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="要执行的手势名,可以给多个")
    ap.add_argument("--list", action="store_true", help="列出全部手势")
    ap.add_argument("--demo", action="store_true", help="依次跑一遍全部手势")
    ap.add_argument("--hold", type=float, default=0.8,
                    help="每个手势之间停留秒数,默认 0.8")
    ap.add_argument("--dry-run", action="store_true",
                    help="只算指令位置并打印,不连手")
    ap.add_argument("--no-open-first", action="store_true",
                    help="不先回张开姿态(默认会先张开,更安全)")
    args = ap.parse_args()

    if args.list or (not args.names and not args.demo):
        print(f"{'名称':12s} {'中文':8s} 说明")
        for g in GESTURES.values():
            print(f"{g.name:12s} {g.name_cn:8s} {g.desc}")
        print(f"\n共 {len(GESTURES)} 个。用法: python gestures.py fist point open")
        return 0

    names = list(GESTURES) if args.demo else args.names
    unknown = [n for n in names if n not in GESTURES]
    if unknown:
        print(f"未知手势: {unknown}\n可用: {list(GESTURES)}")
        return 2

    cal = Calibration.load()

    if args.dry_run:
        for n in names:
            g = GESTURES[n]
            print(f"\n▶ {g.name} ({g.name_cn})")
            norms = [0.0] * N_MOTORS
            for i, s in enumerate(g.steps, 1):
                norms = resolve(s, norms)
                print(f"   步骤{i} speed={s.speed:3d} "
                      f"norm{[round(v, 2) for v in norms]} → "
                      f"pos{cal.pose_from_norm(norms)}")
        return 0

    with ROHand() as hand:
        print(f"已连接 {hand.versions()}")
        norms = [0.0] * N_MOTORS
        try:
            if not args.no_open_first:
                norms = play(hand, cal, GESTURES["open"], norms)
            for n in names:
                norms = play(hand, cal, GESTURES[n], norms)
                time.sleep(args.hold)
            # 收尾一定回张开,不要把手留在夹紧状态干等
            play(hand, cal, GESTURES["open"], norms)
        except (HandError, KeyboardInterrupt) as exc:
            print(f"\n中断: {exc}")
            try:
                hand.stop()
                hand.move(cal.pose_from_norm([0.0] * N_MOTORS), speed=140)
                hand.wait_settled()
            except Exception:                   # noqa: BLE001
                pass
            return 1
    print("\n完成,已回到张开姿态")
    return 0


if __name__ == "__main__":
    sys.exit(main())
