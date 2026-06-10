"""Entry-filter lens: test filters ON TOP of the short-only baseline.

Loads hist_1h.parquet, runs run_backtest(short_only=True), then filters the
baseline signals by entry-time-computable conditions and recomputes portfolio
stats (NET of 0.19% round-trip cost) on each kept subset, split-half.
"""
import datetime
import statistics
import sys

import duckdb

sys.path.insert(0, "src")
from hl_swing_bot.backtest import run_backtest, _compute_features_at  # noqa: E402
from hl_swing_bot.features import HourlyBar, aggregate_to_4h          # noqa: E402

COST = 0.19          # round-trip cost, pct
RISK_PCT = 0.5       # sizing: 0.5% equity risk per trade
STOP_ATR_MULT = 1.5
SPLIT_IDX = 2500     # bars 0-2500 = half1, 2501+ = half2


def load_bars():
    con = duckdb.connect()
    rows = con.execute(
        "select open_time_ms, open, high, low, close, volume, trades "
        "from 'data/hist_1h.parquet' order by open_time_ms"
    ).fetchall()
    return [
        HourlyBar(hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
                  low=float(r[3]), close=float(r[4]), volume=float(r[5]),
                  trades=int(r[6]))
        for r in rows
    ]


def sma(vals):
    return sum(vals) / len(vals)


def enrich(bars, signals):
    """Attach entry-time features + slope info to each baseline signal."""
    out = []
    for s in signals:
        i = s["idx"]
        f = _compute_features_at(bars, i)
        assert f is not None
        # 4h SMA50 slope over last 10 4h bars (look-ahead-safe: bars[:i+1]).
        bars_4h = aggregate_to_4h(bars[: i + 1])
        slope_neg = None
        if len(bars_4h) >= 61:
            closes4 = [b.close for b in bars_4h]
            sma_now = sma(closes4[-51:-1])     # matches feature SMA window
            sma_prev = sma(closes4[-61:-11])   # same window 10 4h-bars earlier
            slope_neg = sma_now < sma_prev
        dt = datetime.datetime.utcfromtimestamp(s["ms"] / 1000)  # bar close = entry time
        stop_pct = abs(s["stop"] / s["entry"] - 1) * 100
        net = s["realized_pct"] - COST
        out.append({
            **s,
            "atr_pct": f["atr_pct"],
            "hour_utc": dt.hour,
            "dow": dt.weekday(),  # 0=Mon
            "slope_neg": slope_neg,
            "stop_pct": stop_pct,
            "net": net,
            "net_R": net / stop_pct,
        })
    return out


def stats(sigs, label):
    n = len(sigs)
    if n == 0:
        print(f"{label:55s} n=0")
        return {"n": 0, "net": None}
    nets = [s["net"] for s in sigs]
    hits = sum(1 for s in sigs if s["status"] == "HIT_TP")
    # sized equity curve (0.5% risk per trade, ordered by entry idx)
    eq = 0.0
    peak = 0.0
    maxdd = 0.0
    for s in sorted(sigs, key=lambda x: x["idx"]):
        eq += RISK_PCT * s["net_R"]
        peak = max(peak, eq)
        maxdd = max(maxdd, peak - eq)
    mean_net = statistics.mean(nets)
    avg_r = statistics.mean(s["net_R"] for s in sigs)
    flag = "  [n<10 UNRELIABLE]" if n < 10 else ""
    print(f"{label:55s} n={n:3d}  net={mean_net:+.3f}%/tr  hit={hits/n:.0%}  "
          f"avgR={avg_r:+.3f}  maxDD={maxdd:.2f}%  totR={sum(s['net_R'] for s in sigs):+.2f}{flag}")
    return {"n": n, "net": mean_net, "hit": hits / n, "maxdd": maxdd, "avg_r": avg_r}


def split_half(sigs, label):
    h1 = [s for s in sigs if s["idx"] <= SPLIT_IDX]
    h2 = [s for s in sigs if s["idx"] > SPLIT_IDX]
    print(f"--- split-half: {label}")
    r_full = stats(sigs, "  full")
    r1 = stats(h1, "  half1 (idx<=2500)")
    r2 = stats(h2, "  half2 (idx>2500)")
    return r_full, r1, r2


def main():
    bars = load_bars()
    print(f"bars: {len(bars)}  span: {bars[0].hour_ms} .. {bars[-1].hour_ms}")
    res = run_backtest(bars, short_only=True)
    sigs = enrich(bars, res["signals"])
    print(f"baseline short-only signals: {len(sigs)}")

    print("\n================ BASELINE ================")
    split_half(sigs, "baseline (no extra filter)")

    print("\n================ (1)+(2) ATR BANDS ================")
    stats([s for s in sigs if s["atr_pct"] >= 0.5], "(1) atr_pct >= 0.5")
    stats([s for s in sigs if s["atr_pct"] <= 1.5], "(2) atr_pct <= 1.5")
    stats([s for s in sigs if 0.5 <= s["atr_pct"] <= 1.5], "(1+2) 0.5 <= atr_pct <= 1.5")
    # distribution glance
    aps = sorted(s["atr_pct"] for s in sigs)
    print(f"    atr_pct distribution: min={aps[0]:.3f} med={aps[len(aps)//2]:.3f} max={aps[-1]:.3f}")
    for lo, hi in [(0, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 99)]:
        stats([s for s in sigs if lo <= s["atr_pct"] < hi], f"    atr band [{lo},{hi})")

    print("\n================ (3) SCORE BANDS ================")
    stats([s for s in sigs if 3 <= s["score"] < 5], "(3a) score in [3,5)")
    stats([s for s in sigs if s["score"] >= 5], "(3b) score >= 5")
    stats([s for s in sigs if 3 <= s["score"] < 4], "    score in [3,4)")
    stats([s for s in sigs if 4 <= s["score"] < 5], "    score in [4,5)")

    print("\n================ (4) UTC SESSION ================")
    stats([s for s in sigs if 13 <= s["hour_utc"] <= 22], "(4a) US 13-22 UTC")
    stats([s for s in sigs if 0 <= s["hour_utc"] <= 9], "(4b) Asia 0-9 UTC")
    stats([s for s in sigs if 7 <= s["hour_utc"] <= 16], "(4c) EU 7-16 UTC")
    stats([s for s in sigs if not (13 <= s["hour_utc"] <= 22)], "    NOT US hours")

    print("\n================ (5) DAY OF WEEK ================")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for d in range(7):
        stats([s for s in sigs if s["dow"] == d], f"    {names[d]}")
    stats([s for s in sigs if s["dow"] < 5], "(5a) weekdays only")
    stats([s for s in sigs if s["dow"] >= 5], "(5b) weekend only")

    print("\n================ (6) STEEPER TREND (SMA50 slope<0 over 10x4h) ================")
    stats([s for s in sigs if s["slope_neg"] is True], "(6) slope negative")
    stats([s for s in sigs if s["slope_neg"] is False], "    slope NOT negative (kept by baseline)")
    print(f"    slope undefined: {sum(1 for s in sigs if s['slope_neg'] is None)}")

    print("\n================ split-half for candidates ================")
    cands = {
        "(1) atr>=0.5": [s for s in sigs if s["atr_pct"] >= 0.5],
        "(2) atr<=1.5": [s for s in sigs if s["atr_pct"] <= 1.5],
        "(3a) score [3,5)": [s for s in sigs if 3 <= s["score"] < 5],
        "(3b) score >=5": [s for s in sigs if s["score"] >= 5],
        "(4a) US hours": [s for s in sigs if 13 <= s["hour_utc"] <= 22],
        "(4b) Asia hours": [s for s in sigs if 0 <= s["hour_utc"] <= 9],
        "(4c) EU hours": [s for s in sigs if 7 <= s["hour_utc"] <= 16],
        "(5a) weekdays": [s for s in sigs if s["dow"] < 5],
        "(6) slope neg": [s for s in sigs if s["slope_neg"] is True],
    }
    for label, sub in cands.items():
        split_half(sub, label)


if __name__ == "__main__":
    main()
