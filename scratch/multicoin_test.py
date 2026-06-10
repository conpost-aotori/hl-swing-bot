"""Multi-coin expansion test: run the FROZEN short-only backtest on BTC/ETH/SOL,
report per-coin stats + split-half, portfolio equity with global cluster cap,
and cross-coin signal overlap. No parameter tuning anywhere."""
from __future__ import annotations

import datetime
import statistics

import polars as pl

from hl_swing_bot.backtest import HourlyBar, run_backtest

HOUR_MS = 3600_000
COST = 0.19          # round-trip cost in %
RISK_PCT = 0.5       # equity risk per trade
CLUSTER_CAP = 1.5    # max open risk -> 3 concurrent trades
SPAN_DAYS = 208.0


def load(path: str) -> list[HourlyBar]:
    df = pl.read_parquet(path).sort("open_time_ms")
    return [
        HourlyBar(int(r["open_time_ms"]), float(r["open"]), float(r["high"]),
                  float(r["low"]), float(r["close"]), float(r["volume"]),
                  int(r["trades"]))
        for r in df.iter_rows(named=True)
    ]


def enrich(signals: list[dict], bars: list[HourlyBar], coin: str) -> list[dict]:
    out = []
    for s in signals:
        stop_dist_pct = abs(s["stop"] / s["entry"] - 1) * 100
        net = s["realized_pct"] - COST
        out.append({
            **s,
            "coin": coin,
            "entry_ms": s["ms"],
            "exit_ms": bars[s["exit_idx"]].hour_ms + HOUR_MS,
            "stop_dist_pct": stop_dist_pct,
            "net_pct": net,
            "net_R": net / stop_dist_pct,
        })
    return out


def summarize(sigs: list[dict], label: str, boundary_ms: int) -> None:
    n = len(sigs)
    if n == 0:
        print(f"{label}: 0 signals")
        return
    gross = statistics.mean(s["realized_pct"] for s in sigs)
    net = statistics.mean(s["net_pct"] for s in sigs)
    hit = sum(1 for s in sigs if s["status"] == "HIT_TP") / n
    h1 = [s for s in sigs if s["entry_ms"] <= boundary_ms]
    h2 = [s for s in sigs if s["entry_ms"] > boundary_ms]
    net1 = statistics.mean(s["net_pct"] for s in h1) if h1 else float("nan")
    net2 = statistics.mean(s["net_pct"] for s in h2) if h2 else float("nan")
    netR = statistics.mean(s["net_R"] for s in sigs)
    print(f"{label}: n={n} hit={hit*100:.0f}% gross={gross:+.3f}% "
          f"net={net:+.3f}% netR={netR:+.3f} | "
          f"half1 n={len(h1)} net={net1:+.3f}% | half2 n={len(h2)} net={net2:+.3f}%")


def equity_path(trades: list[dict]) -> tuple[float, float, float]:
    """Compounded equity applying each trade's 0.5%-risk PnL at exit time.
    Returns (total_return_pct, maxDD_pct, final_equity)."""
    eq = 1.0
    peak = 1.0
    maxdd = 0.0
    for t in sorted(trades, key=lambda x: x["exit_ms"]):
        eq *= 1 + (RISK_PCT / 100) * t["net_R"]
        peak = max(peak, eq)
        maxdd = max(maxdd, 1 - eq / peak)
    return (eq - 1) * 100, maxdd * 100, eq


def main() -> None:
    btc_bars = load("data/hist_1h.parquet")
    eth_bars = load("scratch/hist_1h_eth.parquet")
    sol_bars = load("scratch/hist_1h_sol.parquet")
    boundary_ms = btc_bars[2500].hour_ms + HOUR_MS  # split-half boundary (BTC bar 2500 close)
    print(f"split boundary: {datetime.datetime.utcfromtimestamp(boundary_ms/1000)}")

    all_sigs: dict[str, list[dict]] = {}
    for coin, bars in [("BTC", btc_bars), ("ETH", eth_bars), ("SOL", sol_bars)]:
        res = run_backtest(bars, short_only=True)  # frozen params
        sigs = enrich(res.get("signals", []), bars, coin)
        all_sigs[coin] = sigs
        summarize(sigs, coin, boundary_ms)

    # --- portfolio with global cluster cap (max 3 concurrent at 0.5% risk) ---
    merged = sorted(
        (s for sigs in all_sigs.values() for s in sigs),
        key=lambda s: (s["entry_ms"], s["coin"]),
    )
    accepted: list[dict] = []
    skipped = 0
    open_trades: list[dict] = []
    max_concurrent = 0
    for s in merged:
        open_trades = [t for t in open_trades if t["exit_ms"] > s["entry_ms"]]
        if len(open_trades) * RISK_PCT >= CLUSTER_CAP:
            skipped += 1
            continue
        accepted.append(s)
        open_trades.append(s)
        max_concurrent = max(max_concurrent, len(open_trades))

    print(f"\n=== PORTFOLIO (global cap {CLUSTER_CAP}% = 3 slots) ===")
    print(f"candidate signals: {len(merged)}  accepted: {len(accepted)}  "
          f"skipped-by-cap: {skipped}  max concurrent: {max_concurrent}")
    summarize(accepted, "PORTFOLIO", boundary_ms)
    per_year = len(accepted) * 365.25 / SPAN_DAYS
    tot, maxdd, _ = equity_path(accepted)
    print(f"trades/year: {per_year:.0f}  total return (compounded, sized): {tot:+.2f}%  "
          f"maxDD: {maxdd:.2f}%")
    for coin in ("BTC", "ETH", "SOL"):
        sub = [s for s in accepted if s["coin"] == coin]
        tot_c, dd_c, _ = equity_path(all_sigs[coin])
        print(f"  {coin}: accepted {len(sub)}/{len(all_sigs[coin])}  "
              f"standalone sized return {tot_c:+.2f}% maxDD {dd_c:.2f}%")

    # --- cross-coin time overlap ---
    print("\n=== CROSS-COIN OVERLAP ===")
    for w_h in (6, 24):
        w = w_h * HOUR_MS
        pairs = {}
        for a in ("BTC", "ETH", "SOL"):
            for b in ("BTC", "ETH", "SOL"):
                if a >= b:
                    continue
                cnt = sum(
                    1 for sa in all_sigs[a]
                    if any(abs(sa["entry_ms"] - sb["entry_ms"]) <= w
                           for sb in all_sigs[b])
                )
                pairs[f"{a}~{b}"] = cnt
        print(f"  within {w_h}h: " + "  ".join(
            f"{k}: {v}/{len(all_sigs[k.split('~')[0]])}" for k, v in pairs.items()))
    days = {}
    for coin, sigs in all_sigs.items():
        for s in sigs:
            d = datetime.datetime.utcfromtimestamp(s["entry_ms"] / 1000).date()
            days.setdefault(d, set()).add(coin)
    multi = {d: c for d, c in days.items() if len(c) >= 2}
    print(f"  UTC days with any signal: {len(days)}  days with 2+ coins: {len(multi)}")
    for d in sorted(multi):
        print(f"    {d}: {sorted(multi[d])}")


if __name__ == "__main__":
    main()
