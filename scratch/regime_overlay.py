"""Regime-overlay lens test on the short-only baseline.

Loads 5002 1h bars from parquet, reproduces the 49-short baseline via
run_backtest (slippage_bps=0 -> gross; NET = gross - 0.19 round trip),
then post-filters the signal list with regime overlays computed
look-ahead-safe at each signal bar.
"""
from __future__ import annotations

import csv
import statistics
import sys

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")
from hl_swing_bot.backtest import run_backtest  # noqa: E402
from hl_swing_bot.features import HourlyBar, aggregate_to_4h, wilder_atr  # noqa: E402

CSV_PATH = r"C:\User\projects\hl-swing-bot\scratch\hist_1h.csv"
COST = 0.19  # round-trip pct
SPLIT_IDX = 2500
DAYS = 208.0


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
    n_bars = len(bars)
    print(f"bars: {n_bars}  span_days: {(bars[-1].hour_ms - bars[0].hour_ms) / 86400000:.1f}")

    res = run_backtest(bars, slippage_bps=0.0, short_only=True)
    sigs = res["signals"]
    print(f"baseline shorts: {len(sigs)}  gross/trade: {res['expectancy_pct_post_slippage']:.4f}"
          f"  net/trade: {res['expectancy_pct_post_slippage'] - COST:.4f}"
          f"  hit_tp: {res['hit_rate_tp']:.3f}")

    # Precompute ATR series once (full history; wilder_atr at index i uses only
    # bars <= i, so atrs[i] is look-ahead-safe).
    atrs = wilder_atr(bars)
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]

    # 1h log-ish pct returns for realized vol.
    rets = [0.0] * n_bars
    for i in range(1, n_bars):
        rets[i] = (closes[i] / closes[i - 1] - 1) * 100

    # 168h realized vol per bar (causal): stdev of rets[i-167..i].
    vol168 = [None] * n_bars
    for i in range(168, n_bars):
        window = rets[i - 167: i + 1]
        vol168[i] = statistics.pstdev(window)

    # Expanding percentile rank of vol168[i] among vol168[168..i] (causal).
    # O(n^2) but n=5000 -> fine.
    vol_rank = [None] * n_bars
    hist_sorted: list[float] = []
    import bisect
    for i in range(168, n_bars):
        v = vol168[i]
        if hist_sorted:
            pos = bisect.bisect_left(hist_sorted, v)
            vol_rank[i] = pos / len(hist_sorted)
        else:
            vol_rank[i] = 0.5
        bisect.insort(hist_sorted, v)

    # Per-signal regime features.
    feats = []
    for s in sigs:
        i = s["idx"]
        sub = bars[: i + 1]
        # (1) 30d trailing return
        r30 = (closes[i] / closes[i - 720] - 1) * 100 if i >= 720 else None
        # (2) realized vol regime
        vr = vol_rank[i]
        # (3) distance below 4h SMA50 in ATR units
        b4 = aggregate_to_4h(sub)
        if len(b4) >= 51:
            sma50 = statistics.mean(b.close for b in b4[-51:-1])
            depth_atr = (sma50 - closes[i]) / atrs[i] if atrs[i] > 0 else None
        else:
            depth_atr = None
        # (4) consecutive red 4h bars (completed buckets: exclude the bucket
        # containing bar i if it is partial; b4[-1] is the bucket bar i is in).
        # variant A: include current (possibly partial) 4h bar
        red_incl = 0
        for b in reversed(b4):
            if b.close < b.open:
                red_incl += 1
            else:
                break
        # variant B: completed 4h bars only (drop last bucket if partial)
        bucket_ms = 4 * 3600 * 1000
        last_full = b4[:-1] if (bars[i].hour_ms - b4[-1].hour_ms) < bucket_ms - 3600 * 1000 else b4
        red_compl = 0
        for b in reversed(last_full):
            if b.close < b.open:
                red_compl += 1
            else:
                break
        # (5) drawdown from 30d high
        if i >= 720:
            hi30 = max(highs[i - 719: i + 1])
            dd30 = (closes[i] / hi30 - 1) * 100
        else:
            dd30 = None
        feats.append({
            "idx": i, "gross": s["realized_pct"], "net": s["realized_pct"] - COST,
            "status": s["status"], "r30": r30, "vol_rank": vr,
            "depth_atr": depth_atr, "red_incl": red_incl, "red_compl": red_compl,
            "dd30": dd30,
            "stop_pct": abs(s["stop"] / s["entry"] - 1) * 100,
        })

    n_no30 = sum(1 for f in feats if f["r30"] is None)
    print(f"signals lacking 30d window (idx<720): {n_no30}")

    def report(name, keep_fn):
        kept = [f for f in feats if keep_fn(f)]
        n = len(kept)
        if n == 0:
            print(f"{name:55s} n=0")
            return None
        net = statistics.mean(f["net"] for f in kept)
        gross = statistics.mean(f["gross"] for f in kept)
        hit = sum(1 for f in kept if f["status"] == "HIT_TP") / n
        avg_r = statistics.mean(f["net"] / f["stop_pct"] for f in kept)
        h1 = [f for f in kept if f["idx"] <= SPLIT_IDX]
        h2 = [f for f in kept if f["idx"] > SPLIT_IDX]
        n1, n2 = len(h1), len(h2)
        net1 = statistics.mean(f["net"] for f in h1) if h1 else float("nan")
        net2 = statistics.mean(f["net"] for f in h2) if h2 else float("nan")
        print(f"{name:55s} n={n:3d} gross={gross:+.3f} net={net:+.3f} hit={hit:.2f} "
              f"avgR={avg_r:+.3f} | h1 n={n1:3d} net={net1:+.3f} | h2 n={n2:3d} net={net2:+.3f}")
        return {"name": name, "n": n, "net": net, "net1": net1, "net2": net2,
                "n1": n1, "n2": n2, "avg_r": avg_r}

    print("\n--- baseline ---")
    base = report("BASELINE short-only", lambda f: True)

    print("\n--- (1) 30d trailing return ---")
    report("r30 < 0% (loose downtrend)", lambda f: f["r30"] is not None and f["r30"] < 0)
    report("r30 < -6% (strict downtrend)", lambda f: f["r30"] is not None and f["r30"] < -6)
    report("r30 < -3%", lambda f: f["r30"] is not None and f["r30"] < -3)
    report("r30 < -10%", lambda f: f["r30"] is not None and f["r30"] < -10)

    print("\n--- (2) realized-vol regime (168h stdev, causal expanding rank) ---")
    report("vol_rank >= 0.5 (top half)", lambda f: f["vol_rank"] is not None and f["vol_rank"] >= 0.5)
    report("vol_rank < 0.5 (bottom half)", lambda f: f["vol_rank"] is not None and f["vol_rank"] < 0.5)
    report("vol_rank >= 0.75 (top quartile)", lambda f: f["vol_rank"] is not None and f["vol_rank"] >= 0.75)

    print("\n--- (3) distance below 4h SMA50 ---")
    report("depth >= 0 ATR (just below = baseline gate)", lambda f: f["depth_atr"] is not None and f["depth_atr"] > 0)
    report("depth >= 1 ATR (deep)", lambda f: f["depth_atr"] is not None and f["depth_atr"] >= 1.0)
    report("depth >= 2 ATR", lambda f: f["depth_atr"] is not None and f["depth_atr"] >= 2.0)
    report("depth >= 3 ATR", lambda f: f["depth_atr"] is not None and f["depth_atr"] >= 3.0)
    report("depth < 3 ATR (shallow only)", lambda f: f["depth_atr"] is not None and f["depth_atr"] < 3.0)

    print("\n--- (4) consecutive red 4h bars ---")
    report("red_incl >= 2 (incl current bucket)", lambda f: f["red_incl"] >= 2)
    report("red_compl >= 2 (completed only)", lambda f: f["red_compl"] >= 2)
    report("red_incl >= 1", lambda f: f["red_incl"] >= 1)
    report("red_compl >= 1", lambda f: f["red_compl"] >= 1)

    print("\n--- (5) drawdown from 30d high ---")
    report("dd30 < -10% (capitulation)", lambda f: f["dd30"] is not None and f["dd30"] < -10)
    report("dd30 < -5%", lambda f: f["dd30"] is not None and f["dd30"] < -5)
    report("dd30 < -15%", lambda f: f["dd30"] is not None and f["dd30"] < -15)
    report("dd30 >= -10% (NOT capitulation)", lambda f: f["dd30"] is not None and f["dd30"] >= -10)

    print("\n--- distributions at signals ---")
    for k in ("r30", "vol_rank", "depth_atr", "dd30"):
        vals = sorted(f[k] for f in feats if f[k] is not None)
        if vals:
            qs = [vals[0], vals[len(vals)//4], vals[len(vals)//2], vals[3*len(vals)//4], vals[-1]]
            print(f"{k:10s} min/q1/med/q3/max: " + " ".join(f"{v:+.2f}" for v in qs))
    print("red_incl counts:", {c: sum(1 for f in feats if f["red_incl"] == c) for c in range(0, 6)})
    print("red_compl counts:", {c: sum(1 for f in feats if f["red_compl"] == c) for c in range(0, 6)})


if __name__ == "__main__":
    main()
