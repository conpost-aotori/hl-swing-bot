"""Follow-up adversarial checks: t-stats and split-boundary sensitivity.
Uses feature caches built by verify_multicoin.py (fast)."""
from __future__ import annotations

import datetime
import json
import math
import os
import statistics

import polars as pl

from verify_multicoin import (
    HOUR_MS, load, my_backtest, portfolio, stats, fmt, CAP_SLOTS,
)


def tstat(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = statistics.mean(xs)
    sd = statistics.stdev(xs)
    return m / (sd / math.sqrt(n)), sd


def main() -> None:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    btc_bars = load("data/hist_1h.parquet")
    eth_bars = load("scratch/hist_1h_eth.parquet")
    feats = {}
    for c in ("BTC", "ETH"):
        with open(os.path.join(ROOT, "scratch", f"verify_feat_{c}.json")) as fh:
            feats[c] = json.load(fh)

    b = my_backtest(btc_bars, feats["BTC"], "BTC", 3.0, 1.0, 1.0)
    e = my_backtest(eth_bars, feats["ETH"], "ETH", 3.0, 1.0, 1.0)
    acc, _ = portfolio([b, e], CAP_SLOTS)

    for label, sigs in [("BTC", b), ("ETH", e), ("BTC+ETH cap3", acc)]:
        nets = [s["net"] for s in sigs]
        t, sd = tstat(nets)
        print(f"{label}: n={len(nets)} mean={statistics.mean(nets):+.3f}% "
              f"sd={sd:.3f}% t={t:+.2f}")

    # diff of means ETH vs BTC (is ETH actually better, or same edge?)
    nb, ne = [s["net"] for s in b], [s["net"] for s in e]
    mb, me = statistics.mean(nb), statistics.mean(ne)
    se = math.sqrt(statistics.variance(nb) / len(nb) + statistics.variance(ne) / len(ne))
    print(f"ETH-BTC mean diff: {me - mb:+.3f}%  SE={se:.3f}  t={(me - mb) / se:+.2f}")

    # split-boundary sensitivity: split at bar 1875, 2500, 3125 of BTC series
    print("\n=== boundary sensitivity (portfolio cap3) ===")
    for k in (1875, 2500, 3125):
        bd = btc_bars[k].hour_ms + HOUR_MS
        s = stats(acc, bd)
        d = datetime.datetime.utcfromtimestamp(bd / 1000).date()
        print(f"boundary bar {k} ({d}): H1 n={s['n1']} net={s['net1']:+.3f}% | "
              f"H2 n={s['n2']} net={s['net2']:+.3f}%")

    # thirds (terciles) — harsher regime slicing
    print("\n=== terciles (portfolio cap3) ===")
    b1 = btc_bars[1667].hour_ms + HOUR_MS
    b2 = btc_bars[3334].hour_ms + HOUR_MS
    t1 = [s for s in acc if s["entry_ms"] <= b1]
    t2 = [s for s in acc if b1 < s["entry_ms"] <= b2]
    t3 = [s for s in acc if s["entry_ms"] > b2]
    for i, t in enumerate([t1, t2, t3], 1):
        nets = [s["net"] for s in t]
        print(f"T{i}: n={len(t)} net={statistics.mean(nets):+.3f}%")


if __name__ == "__main__":
    main()
