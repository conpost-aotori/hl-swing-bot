"""Robustness checks for the steeper-trend filter:
- vary slope lookback (4h bars): 5, 8, 10, 12, 15, 20
- combos: slope-neg + weekdays, slope-neg + US hours, slope-neg + score<5
All split-half, NET of 0.19%.
"""
import datetime
import statistics
import sys

import duckdb

sys.path.insert(0, "src")
from hl_swing_bot.backtest import run_backtest, _compute_features_at  # noqa: E402
from hl_swing_bot.features import HourlyBar, aggregate_to_4h          # noqa: E402

COST = 0.19
RISK_PCT = 0.5
SPLIT_IDX = 2500


def load_bars():
    con = duckdb.connect()
    rows = con.execute(
        "select open_time_ms, open, high, low, close, volume, trades "
        "from 'data/hist_1h.parquet' order by open_time_ms"
    ).fetchall()
    return [HourlyBar(hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
                      low=float(r[3]), close=float(r[4]), volume=float(r[5]),
                      trades=int(r[6])) for r in rows]


def sma(vals):
    return sum(vals) / len(vals)


def stats(sigs, label):
    n = len(sigs)
    if n == 0:
        print(f"{label:48s} n=0")
        return
    nets = [s["net"] for s in sigs]
    hits = sum(1 for s in sigs if s["status"] == "HIT_TP")
    eq = peak = maxdd = 0.0
    for s in sorted(sigs, key=lambda x: x["idx"]):
        eq += RISK_PCT * s["net_R"]
        peak = max(peak, eq)
        maxdd = max(maxdd, peak - eq)
    flag = "  [n<10]" if n < 10 else ""
    print(f"{label:48s} n={n:3d}  net={statistics.mean(nets):+.3f}%  hit={hits/n:.0%}  "
          f"avgR={statistics.mean(s['net_R'] for s in sigs):+.3f}  maxDD={maxdd:.2f}%{flag}")


def split_half(sigs, label):
    print(f"--- {label}")
    stats(sigs, "  full")
    stats([s for s in sigs if s["idx"] <= SPLIT_IDX], "  half1")
    stats([s for s in sigs if s["idx"] > SPLIT_IDX], "  half2")


def main():
    bars = load_bars()
    res = run_backtest(bars, short_only=True)
    base = res["signals"]

    lookbacks = [5, 8, 10, 12, 15, 20]
    enriched = []
    for s in base:
        i = s["idx"]
        bars_4h = aggregate_to_4h(bars[: i + 1])
        closes4 = [b.close for b in bars_4h]
        slopes = {}
        for lb in lookbacks:
            if len(closes4) >= 51 + lb:
                sma_now = sma(closes4[-51:-1])
                sma_prev = sma(closes4[-51 - lb:-1 - lb])
                slopes[lb] = sma_now < sma_prev
            else:
                slopes[lb] = None
        dt = datetime.datetime.utcfromtimestamp(s["ms"] / 1000)
        stop_pct = abs(s["stop"] / s["entry"] - 1) * 100
        net = s["realized_pct"] - COST
        enriched.append({**s, "slopes": slopes, "hour_utc": dt.hour,
                         "dow": dt.weekday(), "net": net, "net_R": net / stop_pct})

    print("============ slope lookback sensitivity ============")
    for lb in lookbacks:
        split_half([s for s in enriched if s["slopes"][lb] is True],
                   f"slope-neg lookback={lb} x 4h")

    print("\n============ combos (on slope-neg lb=10) ============")
    s10 = [s for s in enriched if s["slopes"][10] is True]
    split_half([s for s in s10 if s["dow"] < 5], "slope-neg10 + weekdays")
    split_half([s for s in s10 if 13 <= s["hour_utc"] <= 22], "slope-neg10 + US 13-22 UTC")
    split_half([s for s in s10 if s["score"] < 5], "slope-neg10 + score<5")
    split_half([s for s in s10 if s["dow"] < 5 and 13 <= s["hour_utc"] <= 22],
               "slope-neg10 + weekdays + US")


if __name__ == "__main__":
    main()
