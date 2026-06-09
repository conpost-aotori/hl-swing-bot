#!/usr/bin/env python
"""Multi-regime walk-forward over long 1h history with FROZEN params.

Answers the panel's key questions WITHOUT re-tuning:
  (i)   total signals, and crucially: does the LONG branch EVER fire?
  (ii)  per-regime cells (uptrend / downtrend / chop)
  (iii) episode-level stats (signals within 72h same-dir merged into one event)
  (iv)  per-quarter expectancy

Reads data/hist_1h.parquet (fetched separately). Uses the existing frozen
backtest logic — no parameter search here, by design.
"""
from __future__ import annotations

import datetime
import statistics

import polars as pl

from hl_swing_bot.backtest import HourlyBar, run_backtest

REGIME_LOOKBACK_BARS = 720      # 30 days of 1h
REGIME_TREND_THRESH = 0.06      # +/-6% over 30d => trend, else chop
EPISODE_MERGE_HOURS = 72


def load_bars(path: str = "data/hist_1h.parquet") -> list[HourlyBar]:
    df = pl.read_parquet(path).sort("open_time_ms")
    return [
        HourlyBar(int(r["open_time_ms"]), float(r["open"]), float(r["high"]),
                  float(r["low"]), float(r["close"]), float(r["volume"]), int(r["trades"]))
        for r in df.iter_rows(named=True)
    ]


def label_regime(bars: list[HourlyBar], idx_by_ms: dict[int, int], sig_ms: int) -> str:
    """Coarse regime at signal time from trailing 30d return."""
    # signal ms is bar_close; find the bar index
    i = idx_by_ms.get(sig_ms - 3600_000)  # bar_close_ms = hour_ms + 1h
    if i is None or i < REGIME_LOOKBACK_BARS:
        return "warmup"
    past = bars[i - REGIME_LOOKBACK_BARS].close
    now = bars[i].close
    if past <= 0:
        return "warmup"
    ret = now / past - 1.0
    if ret > REGIME_TREND_THRESH:
        return "uptrend"
    if ret < -REGIME_TREND_THRESH:
        return "downtrend"
    return "chop"


def merge_episodes(signals: list[dict]) -> list[dict]:
    """Collapse same-direction signals within 72h into one episode."""
    episodes: list[dict] = []
    last_by_dir: dict[str, dict] = {}
    for s in sorted(signals, key=lambda x: x["ms"]):
        d = s["direction"]
        prev = last_by_dir.get(d)
        if prev and (s["ms"] - prev["last_ms"]) <= EPISODE_MERGE_HOURS * 3600_000:
            prev["last_ms"] = s["ms"]
            prev["legs"].append(s)
        else:
            ep = {"direction": d, "first_ms": s["ms"], "last_ms": s["ms"], "legs": [s]}
            episodes.append(ep)
            last_by_dir[d] = ep
    # episode realized = mean of its legs' realized (net) returns
    for ep in episodes:
        ep["n_legs"] = len(ep["legs"])
        ep["realized_pct"] = statistics.mean(l["realized_pct"] for l in ep["legs"])
    return episodes


def main() -> None:
    bars = load_bars()
    idx_by_ms = {b.hour_ms: i for i, b in enumerate(bars)}
    t0 = datetime.datetime.utcfromtimestamp(bars[0].hour_ms / 1000)
    t1 = datetime.datetime.utcfromtimestamp(bars[-1].hour_ms / 1000)
    print(f"history: {len(bars)} 1h bars  {t0:%Y-%m-%d} -> {t1:%Y-%m-%d}  "
          f"({(bars[-1].hour_ms-bars[0].hour_ms)/86400000:.0f}d)")

    res = run_backtest(bars)  # frozen params (score>=3.0, move>=1.0, vol>=1.0)
    sigs = res.get("signals", [])
    print(f"\n=== FROZEN-PARAM RESULT ===")
    print(f"signals: {res['n_signals']}  LONG: {res['long_count']}  SHORT: {res['short_count']}")
    print(f"  >>> DOES LONG EVER FIRE?  {'YES' if res['long_count']>0 else 'NO — short-only by construction'}")
    if res["n_signals"]:
        print(f"hit-rate: {res['hit_rate_tp']*100:.0f}%  "
              f"expectancy(net 5bps): {res['expectancy_pct_post_slippage']:+.2f}%  "
              f"worst: {res['worst_pct']:+.2f}%")

    # Regime + quarter tagging
    for s in sigs:
        s["regime"] = label_regime(bars, idx_by_ms, s["ms"])
        s["q"] = datetime.datetime.utcfromtimestamp(s["ms"]/1000).strftime("%Y-Q") + \
                 str((datetime.datetime.utcfromtimestamp(s["ms"]/1000).month - 1)//3 + 1)

    print(f"\n=== PER-REGIME CELLS (the panel's success bar: >=3 cells, >=8 episodes each) ===")
    for reg in ("uptrend", "downtrend", "chop", "warmup"):
        cell = [s for s in sigs if s["regime"] == reg]
        if not cell:
            print(f"  {reg:<10}: 0 signals")
            continue
        longs = sum(1 for s in cell if s["direction"] == "LONG")
        exp = statistics.mean(s["realized_pct"] for s in cell)
        print(f"  {reg:<10}: {len(cell):>3} signals (L:{longs} S:{len(cell)-longs})  "
              f"net-exp {exp:+.2f}%")

    print(f"\n=== PER-QUARTER ===")
    for q in sorted(set(s["q"] for s in sigs)):
        cell = [s for s in sigs if s["q"] == q]
        longs = sum(1 for s in cell if s["direction"] == "LONG")
        exp = statistics.mean(s["realized_pct"] for s in cell)
        print(f"  {q}: {len(cell):>3} signals (L:{longs} S:{len(cell)-longs})  net-exp {exp:+.2f}%")

    eps = merge_episodes(sigs)
    print(f"\n=== EPISODE-LEVEL (72h same-dir merged; the honest n) ===")
    print(f"  raw signals: {len(sigs)}  ->  independent episodes: {len(eps)}")
    longs_ep = [e for e in eps if e["direction"] == "LONG"]
    print(f"  episodes  LONG: {len(longs_ep)}  SHORT: {len(eps)-len(longs_ep)}")
    if eps:
        ep_exp = statistics.mean(e["realized_pct"] for e in eps)
        wins = sum(1 for e in eps if e["realized_pct"] > 0)
        print(f"  episode win-rate: {wins}/{len(eps)} ({wins/len(eps)*100:.0f}%)  "
              f"episode net-exp: {ep_exp:+.2f}%")


if __name__ == "__main__":
    main()
