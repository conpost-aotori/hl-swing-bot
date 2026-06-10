"""Run frozen short-only backtest for one coin; dump enriched signals to JSON.
Usage: run_coin.py COIN PARQUET_PATH OUT_JSON"""
from __future__ import annotations

import json
import sys

import polars as pl

from hl_swing_bot.backtest import HourlyBar, run_backtest

HOUR_MS = 3600_000
COST = 0.19


def main() -> None:
    coin, path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    df = pl.read_parquet(path).sort("open_time_ms")
    bars = [
        HourlyBar(int(r["open_time_ms"]), float(r["open"]), float(r["high"]),
                  float(r["low"]), float(r["close"]), float(r["volume"]),
                  int(r["trades"]))
        for r in df.iter_rows(named=True)
    ]
    res = run_backtest(bars, short_only=True)  # frozen params, no tuning
    sigs = []
    for s in res.get("signals", []):
        stop_dist_pct = abs(s["stop"] / s["entry"] - 1) * 100
        net = s["realized_pct"] - COST
        sigs.append({
            **s,
            "coin": coin,
            "entry_ms": s["ms"],
            "exit_ms": bars[s["exit_idx"]].hour_ms + HOUR_MS,
            "stop_dist_pct": stop_dist_pct,
            "net_pct": net,
            "net_R": net / stop_dist_pct,
        })
    with open(out_path, "w") as f:
        json.dump({"coin": coin, "n_bars": len(bars), "signals": sigs}, f)
    print(f"{coin}: {len(sigs)} signals -> {out_path}")


if __name__ == "__main__":
    main()
