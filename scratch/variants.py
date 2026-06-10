"""Portfolio-construction variants from frozen per-coin signals.
No signal-parameter tuning — only universe selection and concurrency cap."""
from __future__ import annotations

import datetime
import json
import statistics

import polars as pl

HOUR_MS = 3600_000
RISK_PCT = 0.5
SPAN_DAYS = 208.0


def equity_path(trades: list[dict]) -> tuple[float, float]:
    eq, peak, maxdd = 1.0, 1.0, 0.0
    for t in sorted(trades, key=lambda x: x["exit_ms"]):
        eq *= 1 + (RISK_PCT / 100) * t["net_R"]
        peak = max(peak, eq)
        maxdd = max(maxdd, 1 - eq / peak)
    return (eq - 1) * 100, maxdd * 100


def apply_cap(merged: list[dict], max_open: int) -> list[dict]:
    accepted, open_trades = [], []
    for s in sorted(merged, key=lambda x: (x["entry_ms"], x["coin"])):
        open_trades = [t for t in open_trades if t["exit_ms"] > s["entry_ms"]]
        if len(open_trades) >= max_open:
            continue
        accepted.append(s)
        open_trades.append(s)
    return accepted


def report(name: str, trades: list[dict], boundary_ms: int) -> None:
    n = len(trades)
    if n == 0:
        print(f"{name}: 0 trades")
        return
    net = statistics.mean(s["net_pct"] for s in trades)
    h1 = [s for s in trades if s["entry_ms"] <= boundary_ms]
    h2 = [s for s in trades if s["entry_ms"] > boundary_ms]
    n1 = statistics.mean(s["net_pct"] for s in h1) if h1 else float("nan")
    n2 = statistics.mean(s["net_pct"] for s in h2) if h2 else float("nan")
    tot, dd = equity_path(trades)
    hit = sum(1 for s in trades if s["status"] == "HIT_TP") / n
    print(f"{name:<28} n={n:>3} ({n*365.25/SPAN_DAYS:3.0f}/yr) hit={hit*100:3.0f}% "
          f"net={net:+.3f}% H1={n1:+.3f}%({len(h1)}) H2={n2:+.3f}%({len(h2)}) "
          f"sized: ret={tot:+.2f}% maxDD={dd:.2f}%")


def main() -> None:
    btc_df = pl.read_parquet("data/hist_1h.parquet").sort("open_time_ms")
    boundary_ms = int(btc_df["open_time_ms"][2500]) + HOUR_MS

    sigs = {}
    for coin in ("BTC", "ETH", "SOL"):
        with open(f"scratch/sigs_{coin.lower()}.json") as f:
            sigs[coin] = json.load(f)["signals"]

    universes = {
        "BTC only (baseline)": ["BTC"],
        "ETH only": ["ETH"],
        "SOL only": ["SOL"],
        "BTC+ETH": ["BTC", "ETH"],
        "BTC+ETH+SOL": ["BTC", "ETH", "SOL"],
    }
    for uname, coins in universes.items():
        merged = [s for c in coins for s in sigs[c]]
        for cap in (3, 2, 1):
            if len(coins) == 1 and cap != 3:
                continue
            acc = apply_cap(merged, cap)
            report(f"{uname} cap{cap}", acc, boundary_ms)
        print()


if __name__ == "__main__":
    main()
