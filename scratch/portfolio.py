"""Analyze per-coin + portfolio stats from the per-coin signal JSONs."""
from __future__ import annotations

import datetime
import json
import statistics

import polars as pl

HOUR_MS = 3600_000
RISK_PCT = 0.5
CLUSTER_CAP = 1.5
SPAN_DAYS = 208.0


def summarize(sigs: list[dict], label: str, boundary_ms: int) -> dict:
    n = len(sigs)
    if n == 0:
        print(f"{label}: 0 signals")
        return {}
    gross = statistics.mean(s["realized_pct"] for s in sigs)
    net = statistics.mean(s["net_pct"] for s in sigs)
    hit = sum(1 for s in sigs if s["status"] == "HIT_TP") / n
    h1 = [s for s in sigs if s["entry_ms"] <= boundary_ms]
    h2 = [s for s in sigs if s["entry_ms"] > boundary_ms]
    net1 = statistics.mean(s["net_pct"] for s in h1) if h1 else float("nan")
    net2 = statistics.mean(s["net_pct"] for s in h2) if h2 else float("nan")
    netR = statistics.mean(s["net_R"] for s in sigs)
    print(f"{label:<10} n={n:>3} hit={hit*100:4.0f}% gross={gross:+.3f}% "
          f"net={net:+.3f}% netR={netR:+.3f} || "
          f"H1 n={len(h1):>3} net={net1:+.3f}% | H2 n={len(h2):>3} net={net2:+.3f}%")
    return {"n": n, "net": net, "net1": net1, "net2": net2}


def equity_path(trades: list[dict]) -> tuple[float, float]:
    eq, peak, maxdd = 1.0, 1.0, 0.0
    for t in sorted(trades, key=lambda x: x["exit_ms"]):
        eq *= 1 + (RISK_PCT / 100) * t["net_R"]
        peak = max(peak, eq)
        maxdd = max(maxdd, 1 - eq / peak)
    return (eq - 1) * 100, maxdd * 100


def main() -> None:
    btc_df = pl.read_parquet("data/hist_1h.parquet").sort("open_time_ms")
    boundary_ms = int(btc_df["open_time_ms"][2500]) + HOUR_MS
    print(f"split boundary (BTC bar 2500 close): "
          f"{datetime.datetime.utcfromtimestamp(boundary_ms/1000)}")

    all_sigs: dict[str, list[dict]] = {}
    for coin in ("BTC", "ETH", "SOL"):
        with open(f"scratch/sigs_{coin.lower()}.json") as f:
            all_sigs[coin] = json.load(f)["signals"]

    print("\n=== PER-COIN (frozen short-only, net of 0.19%) ===")
    for coin in ("BTC", "ETH", "SOL"):
        summarize(all_sigs[coin], coin, boundary_ms)
        tot, dd = equity_path(all_sigs[coin])
        print(f"           standalone sized: return {tot:+.2f}%  maxDD {dd:.2f}%")

    pooled = [s for sigs in all_sigs.values() for s in sigs]
    print("\n=== POOLED (no cap, all signals) ===")
    summarize(pooled, "POOLED", boundary_ms)

    # --- portfolio with global cluster cap ---
    merged = sorted(pooled, key=lambda s: (s["entry_ms"], s["coin"]))
    accepted, open_trades = [], []
    skipped, max_conc = 0, 0
    for s in merged:
        open_trades = [t for t in open_trades if t["exit_ms"] > s["entry_ms"]]
        if len(open_trades) * RISK_PCT >= CLUSTER_CAP:
            skipped += 1
            continue
        accepted.append(s)
        open_trades.append(s)
        max_conc = max(max_conc, len(open_trades))

    print(f"\n=== PORTFOLIO (global cap {CLUSTER_CAP}% => max 3 open) ===")
    print(f"candidates {len(merged)}  accepted {len(accepted)}  "
          f"skipped-by-cap {skipped}  max concurrent {max_conc}")
    summarize(accepted, "PORTFOLIO", boundary_ms)
    by_coin = {c: sum(1 for s in accepted if s["coin"] == c) for c in all_sigs}
    print(f"accepted by coin: {by_coin}")
    per_year = len(accepted) * 365.25 / SPAN_DAYS
    tot, dd = equity_path(accepted)
    print(f"trades/year {per_year:.0f}  sized return (208d, compounded) {tot:+.2f}%  "
          f"maxDD {dd:.2f}%")
    h1 = [s for s in accepted if s["entry_ms"] <= boundary_ms]
    h2 = [s for s in accepted if s["entry_ms"] > boundary_ms]
    for label, part in (("H1", h1), ("H2", h2)):
        t, d = equity_path(part)
        print(f"  {label}: n={len(part)} sized return {t:+.2f}% maxDD {d:.2f}%")

    # --- cross-coin overlap ---
    print("\n=== CROSS-COIN OVERLAP (signal entry times) ===")
    for w_h in (6, 24):
        w = w_h * HOUR_MS
        parts = []
        for a, b in (("BTC", "ETH"), ("BTC", "SOL"), ("ETH", "SOL")):
            cnt = sum(1 for sa in all_sigs[a]
                      if any(abs(sa["entry_ms"] - sb["entry_ms"]) <= w
                             for sb in all_sigs[b]))
            parts.append(f"{a}~{b}: {cnt}/{len(all_sigs[a])}")
        print(f"  within {w_h:>2}h: " + "   ".join(parts))
    days: dict = {}
    for coin, sigs in all_sigs.items():
        for s in sigs:
            d = datetime.datetime.utcfromtimestamp(s["entry_ms"] / 1000).date()
            days.setdefault(d, set()).add(coin)
    multi = {d: c for d, c in days.items() if len(c) >= 2}
    print(f"  UTC days w/ any signal: {len(days)}  days w/ 2+ coins: {len(multi)} "
          f"  days w/ 3 coins: {sum(1 for c in multi.values() if len(c) == 3)}")
    for d in sorted(multi):
        print(f"    {d}: {','.join(sorted(multi[d]))}")


if __name__ == "__main__":
    main()
