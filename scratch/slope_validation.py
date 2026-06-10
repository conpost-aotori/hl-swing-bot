"""Final validation of slope filter:
1. Timing of the 12 removed BTC trades (clustered or spread?)
2. Cross-asset check: ETH (and SOL) 1h bars via candleSnapshot, same pipeline.
"""
import datetime
import json
import statistics
import sys
import time
import urllib.request

import duckdb

sys.path.insert(0, "src")
from hl_swing_bot.backtest import run_backtest  # noqa: E402
from hl_swing_bot.features import HourlyBar, aggregate_to_4h  # noqa: E402

COST = 0.19
RISK_PCT = 0.5


def sma(vals):
    return sum(vals) / len(vals)


def slope_neg(bars, i, lb=10):
    bars_4h = aggregate_to_4h(bars[: i + 1])
    closes4 = [b.close for b in bars_4h]
    if len(closes4) < 51 + lb:
        return None
    return sma(closes4[-51:-1]) < sma(closes4[-51 - lb:-1 - lb])


def stats(sigs, label):
    n = len(sigs)
    if n == 0:
        print(f"{label:44s} n=0")
        return
    nets = [s["realized_pct"] - COST for s in sigs]
    hits = sum(1 for s in sigs if s["status"] == "HIT_TP")
    flag = "  [n<10]" if n < 10 else ""
    print(f"{label:44s} n={n:3d}  net={statistics.mean(nets):+.3f}%  hit={hits/n:.0%}{flag}")


def load_btc():
    con = duckdb.connect()
    rows = con.execute(
        "select open_time_ms, open, high, low, close, volume, trades "
        "from 'data/hist_1h.parquet' order by open_time_ms").fetchall()
    return [HourlyBar(hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
                      low=float(r[3]), close=float(r[4]), volume=float(r[5]),
                      trades=int(r[6])) for r in rows]


def fetch_coin(coin, start_ms, end_ms):
    """Fetch 1h candles via candleSnapshot, paginating forward."""
    out = {}
    cur = start_ms
    while cur < end_ms:
        body = json.dumps({"type": "candleSnapshot", "req": {
            "coin": coin, "interval": "1h", "startTime": cur, "endTime": end_ms}}).encode()
        req = urllib.request.Request("https://api.hyperliquid.xyz/info", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        if not data:
            break
        for c in data:
            out[int(c["t"])] = HourlyBar(
                hour_ms=int(c["t"]), open=float(c["o"]), high=float(c["h"]),
                low=float(c["l"]), close=float(c["c"]), volume=float(c["v"]),
                trades=int(c["n"]))
        last = int(data[-1]["t"])
        if last <= cur and len(data) < 2:
            break
        new_cur = last + 3600_000
        if new_cur <= cur:
            break
        cur = new_cur
        time.sleep(0.3)
    return [out[k] for k in sorted(out)]


def main():
    btc = load_btc()
    res = run_backtest(btc, short_only=True)
    sigs = res["signals"]

    print("==== timing of trades REMOVED by slope filter (BTC) ====")
    removed = [s for s in sigs if slope_neg(btc, s["idx"]) is False]
    for s in removed:
        dt = datetime.datetime.utcfromtimestamp(s["ms"] / 1000)
        print(f"  {dt:%Y-%m-%d %H:%M}  idx={s['idx']:4d}  net={s['realized_pct']-COST:+.3f}%  {s['status']}")

    print("\n==== cross-asset check ====")
    start_ms, end_ms = btc[0].hour_ms, btc[-1].hour_ms + 3600_000
    for coin in ["ETH", "SOL"]:
        bars = fetch_coin(coin, start_ms, end_ms)
        print(f"\n{coin}: {len(bars)} bars  "
              f"{datetime.datetime.utcfromtimestamp(bars[0].hour_ms/1000):%Y-%m-%d} .. "
              f"{datetime.datetime.utcfromtimestamp(bars[-1].hour_ms/1000):%Y-%m-%d}")
        r = run_backtest(bars, short_only=True)
        if not r.get("n_signals"):
            print("  no signals")
            continue
        ss = r["signals"]
        half = bars[len(bars) // 2].hour_ms  # split by time midpoint
        kept = [s for s in ss if slope_neg(bars, s["idx"]) is True]
        drop = [s for s in ss if slope_neg(bars, s["idx"]) is False]
        stats(ss, f"  {coin} baseline short-only")
        stats(kept, f"  {coin} slope-neg kept")
        stats(drop, f"  {coin} slope-neg removed")
        stats([s for s in kept if s["ms"] <= half], f"  {coin} kept half1")
        stats([s for s in kept if s["ms"] > half], f"  {coin} kept half2")


if __name__ == "__main__":
    main()
