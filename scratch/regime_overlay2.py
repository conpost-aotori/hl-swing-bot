"""Round 2: combos of the best overlays + sized portfolio stats."""
from __future__ import annotations

import csv
import statistics
import sys

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")
from hl_swing_bot.backtest import run_backtest  # noqa: E402
from hl_swing_bot.features import HourlyBar, aggregate_to_4h, wilder_atr  # noqa: E402

CSV_PATH = r"C:\User\projects\hl-swing-bot\scratch\hist_1h.csv"
COST = 0.19
SPLIT_IDX = 2500
RISK = 0.005  # 0.5% equity risk per trade


def load_bars() -> list[HourlyBar]:
    out: list[HourlyBar] = []
    with open(CSV_PATH) as fh:
        for r in csv.DictReader(fh):
            out.append(HourlyBar(
                hour_ms=int(r["open_time_ms"]), open=float(r["open"]),
                high=float(r["high"]), low=float(r["low"]),
                close=float(r["close"]), volume=float(r["volume"]),
                trades=int(r["trades"])))
    return out


def main() -> None:
    bars = load_bars()
    span_days = (bars[-1].hour_ms - bars[0].hour_ms) / 86400000
    res = run_backtest(bars, slippage_bps=0.0, short_only=True)
    sigs = res["signals"]
    atrs = wilder_atr(bars)
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]

    feats = []
    for s in sigs:
        i = s["idx"]
        sub = bars[: i + 1]
        r30 = (closes[i] / closes[i - 720] - 1) * 100 if i >= 720 else None
        b4 = aggregate_to_4h(sub)
        red_incl = 0
        for b in reversed(b4):
            if b.close < b.open:
                red_incl += 1
            else:
                break
        bucket_age_h = (bars[i].hour_ms - b4[-1].hour_ms) // 3600000  # 0..3
        last_full = b4[:-1] if bucket_age_h < 3 else b4
        red_compl = 0
        for b in reversed(last_full):
            if b.close < b.open:
                red_compl += 1
            else:
                break
        if i >= 720:
            hi30 = max(highs[i - 719: i + 1])
            dd30 = (closes[i] / hi30 - 1) * 100
        else:
            dd30 = None
        if len(b4) >= 51:
            sma50 = statistics.mean(b.close for b in b4[-51:-1])
            depth_atr = (sma50 - closes[i]) / atrs[i] if atrs[i] > 0 else None
        else:
            depth_atr = None
        feats.append({
            "idx": i, "exit_idx": s["exit_idx"], "gross": s["realized_pct"],
            "net": s["realized_pct"] - COST, "status": s["status"],
            "r30": r30, "red_incl": red_incl, "red_compl": red_compl,
            "dd30": dd30, "depth_atr": depth_atr,
            "stop_pct": abs(s["stop"] / s["entry"] - 1) * 100,
        })

    def stats(name, keep_fn):
        kept = [f for f in feats if keep_fn(f)]
        n = len(kept)
        if n == 0:
            print(f"{name:48s} n=0")
            return
        net = statistics.mean(f["net"] for f in kept)
        hit = sum(1 for f in kept if f["status"] == "HIT_TP") / n
        rs = [f["net"] / f["stop_pct"] for f in kept]
        avg_r = statistics.mean(rs)
        h1 = [f for f in kept if f["idx"] <= SPLIT_IDX]
        h2 = [f for f in kept if f["idx"] > SPLIT_IDX]
        net1 = statistics.mean(f["net"] for f in h1) if h1 else float("nan")
        net2 = statistics.mean(f["net"] for f in h2) if h2 else float("nan")
        # sized equity curve in exit order, compounded
        eq = 1.0
        peak = 1.0
        maxdd = 0.0
        for f in sorted(kept, key=lambda x: x["exit_idx"]):
            eq *= 1 + RISK * (f["net"] / f["stop_pct"])
            peak = max(peak, eq)
            maxdd = max(maxdd, (peak - eq) / peak)
        tpy = n / (span_days / 365)
        ann = avg_r * RISK * 100 * tpy  # % equity per year (simple)
        ann_comp = (eq ** (365 / span_days) - 1) * 100
        print(f"{name:48s} n={n:3d} net={net:+.3f} hit={hit:.2f} avgR={avg_r:+.3f} "
              f"| h1 n={len(h1):3d} {net1:+.3f} | h2 n={len(h2):3d} {net2:+.3f} "
              f"| t/yr={tpy:5.1f} ann%={ann:+.2f} (comp {ann_comp:+.2f}) maxDD={maxdd*100:.2f}%")

    print(f"span_days={span_days:.1f}")
    print("\n--- singles (reference) ---")
    stats("BASELINE", lambda f: True)
    stats("red_incl>=2", lambda f: f["red_incl"] >= 2)
    stats("red_compl>=2", lambda f: f["red_compl"] >= 2)
    stats("dd30<-10", lambda f: f["dd30"] is not None and f["dd30"] < -10)
    stats("r30<-6", lambda f: f["r30"] is not None and f["r30"] < -6)
    stats("depth>=1ATR", lambda f: f["depth_atr"] is not None and f["depth_atr"] >= 1)

    print("\n--- combos ---")
    stats("red_incl>=2 AND dd30<-10", lambda f: f["red_incl"] >= 2 and f["dd30"] is not None and f["dd30"] < -10)
    stats("red_incl>=2 OR dd30<-10", lambda f: f["red_incl"] >= 2 or (f["dd30"] is not None and f["dd30"] < -10))
    stats("red_incl>=2 AND r30<-6", lambda f: f["red_incl"] >= 2 and f["r30"] is not None and f["r30"] < -6)
    stats("red_compl>=2 AND dd30<-10", lambda f: f["red_compl"] >= 2 and f["dd30"] is not None and f["dd30"] < -10)
    stats("red_compl>=1 AND dd30<-10", lambda f: f["red_compl"] >= 1 and f["dd30"] is not None and f["dd30"] < -10)
    stats("red_incl>=2 AND depth>=1", lambda f: f["red_incl"] >= 2 and f["depth_atr"] is not None and f["depth_atr"] >= 1)
    stats("dd30<-10 AND depth>=1", lambda f: f["dd30"] is not None and f["dd30"] < -10 and f["depth_atr"] is not None and f["depth_atr"] >= 1)
    stats("red_incl>=2 AND r30<0", lambda f: f["red_incl"] >= 2 and f["r30"] is not None and f["r30"] < 0)

    print("\n--- red_incl>=2 robustness: neighbors ---")
    stats("red_incl>=3", lambda f: f["red_incl"] >= 3)
    stats("red_incl==2or3", lambda f: f["red_incl"] in (2, 3))
    print("\n--- dd30 threshold neighbors ---")
    stats("dd30<-8", lambda f: f["dd30"] is not None and f["dd30"] < -8)
    stats("dd30<-12", lambda f: f["dd30"] is not None and f["dd30"] < -12)

    # quarter-by-quarter for the top 2 candidates
    print("\n--- quarters (idx 0-1250-2500-3750-5002) for top variants ---")
    for name, fn in [
        ("BASELINE", lambda f: True),
        ("red_incl>=2", lambda f: f["red_incl"] >= 2),
        ("dd30<-10", lambda f: f["dd30"] is not None and f["dd30"] < -10),
        ("red_compl>=2", lambda f: f["red_compl"] >= 2),
        ("red_incl>=2 AND dd30<-10", lambda f: f["red_incl"] >= 2 and f["dd30"] is not None and f["dd30"] < -10),
    ]:
        row = []
        for lo, hi in ((0, 1250), (1251, 2500), (2501, 3750), (3751, 6000)):
            kept = [f for f in feats if fn(f) and lo <= f["idx"] <= hi]
            if kept:
                row.append(f"n={len(kept):2d} {statistics.mean(f['net'] for f in kept):+.2f}")
            else:
                row.append("n= 0   -- ")
        print(f"{name:28s} | " + " | ".join(row))


if __name__ == "__main__":
    main()
