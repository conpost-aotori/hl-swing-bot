"""Follow-up: per-half maxDD, delta concentration, alternative split points,
and dates of the big SL->TP flip trades."""
import statistics
import sys
from datetime import datetime, timezone

import duckdb

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")
from hl_swing_bot.backtest import run_backtest  # noqa: E402
from hl_swing_bot.features import HourlyBar  # noqa: E402

COST, TTL, SL_MULT = 0.19, 72, 1.5

rows = duckdb.sql(
    "SELECT open_time_ms, open, high, low, close, volume, trades "
    "FROM 'C:/User/projects/hl-swing-bot/data/hist_1h.parquet' ORDER BY open_time_ms"
).fetchall()
bars = [HourlyBar(hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
                  low=float(r[3]), close=float(r[4]), volume=float(r[5]),
                  trades=int(r[6])) for r in rows]

res = run_backtest(bars, slippage_bps=0.0, short_only=True)
trades = [{"idx": s["idx"], "entry": s["entry"], "atr": (s["stop"] - s["entry"]) / SL_MULT}
          for s in res["signals"]]


def resolve(tr, tp_mult, be_mult):
    entry, atr, idx = tr["entry"], tr["atr"], tr["idx"]
    stop = entry + SL_MULT * atr
    tp = entry - tp_mult * atr if tp_mult else None
    be_trig = entry - be_mult * atr if be_mult else None
    be_armed = False
    end_idx = min(idx + TTL, len(bars) - 1)
    for j in range(idx + 1, end_idx + 1):
        b = bars[j]
        if b.high >= stop:
            return (entry / stop - 1) * 100, j
        if tp and b.low <= tp:
            return (entry / tp - 1) * 100, j
        if be_trig and not be_armed and b.low <= be_trig:
            stop = entry
            be_armed = True
    return (entry / bars[end_idx].close - 1) * 100, end_idx


def recs_for(tp, be):
    out = []
    for tr in trades:
        g, xj = resolve(tr, tp, be)
        out.append({"idx": tr["idx"], "net": g - COST, "exit_idx": xj,
                    "stop_pct": SL_MULT * tr["atr"] / tr["entry"] * 100})
    return out


def maxdd(recs):
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in sorted(recs, key=lambda r: (r["exit_idx"], r["idx"])):
        eq *= (1 + 0.005 * (r["net"] / r["stop_pct"]))
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return mdd * 100


base = recs_for(2.5, None)
best = recs_for(2.0, 1.25)
grid_max = recs_for(2.0, 1.4)

for name, rs in (("baseline", base), ("best BE1.25/TP2.0", best), ("BE1.4/TP2.0", grid_max)):
    h1 = [r for r in rs if r["idx"] <= 2500]
    h2 = [r for r in rs if r["idx"] > 2500]
    print(f"{name}: full maxDD {maxdd(rs):.2f}%  h1 maxDD {maxdd(h1):.2f}% "
          f"(net {statistics.mean(r['net'] for r in h1):+.3f})  "
          f"h2 maxDD {maxdd(h2):.2f}% (net {statistics.mean(r['net'] for r in h2):+.3f})")

print("\n--- improvement delta (variant - baseline) by half, several cells ---")
for tp, be in ((2.0, 1.25), (2.0, 1.4), (2.25, 1.4), (2.25, 1.25), (2.0, 1.0), (2.5, 1.25), (2.0, None)):
    v = recs_for(tp, be)
    d1 = statistics.mean(v[i]["net"] - base[i]["net"] for i in range(len(v)) if v[i]["idx"] <= 2500)
    d2 = statistics.mean(v[i]["net"] - base[i]["net"] for i in range(len(v)) if v[i]["idx"] > 2500)
    print(f"TP{tp} BE{be}: delta h1 {d1:+.3f}/trade, delta h2 {d2:+.3f}/trade")

print("\n--- alternative split: thirds (idx<=1667, 1668-3334, >3334) ---")
for tp, be, lbl in ((2.5, None, "baseline"), (2.0, 1.25, "best")):
    v = recs_for(tp, be)
    t1 = [r["net"] for r in v if r["idx"] <= 1667]
    t2 = [r["net"] for r in v if 1667 < r["idx"] <= 3334]
    t3 = [r["net"] for r in v if r["idx"] > 3334]
    print(f"{lbl}: T1 n={len(t1)} {statistics.mean(t1):+.3f} | "
          f"T2 n={len(t2)} {statistics.mean(t2):+.3f} | T3 n={len(t3)} {statistics.mean(t3):+.3f}")

print("\n--- big flip trades (dates, overlap) ---")
for i, tr in enumerate(trades):
    if tr["idx"] in (825, 851, 1979, 3969, 4017):
        dt = datetime.fromtimestamp(bars[tr["idx"]].hour_ms / 1000, tz=timezone.utc)
        b0, v0 = base[i], best[i]
        print(f"idx {tr['idx']} {dt:%Y-%m-%d %H:%M} entry {tr['entry']:.0f} "
              f"base net {b0['net']:+.2f} (exit_idx {b0['exit_idx']}) -> "
              f"best net {v0['net']:+.2f} (exit_idx {v0['exit_idx']})")

print("\n--- delta excluding top-3 contributors (idx 825, 851, 1979) ---")
keep = [i for i, tr in enumerate(trades) if tr["idx"] not in (825, 851, 1979)]
db = statistics.mean(base[i]["net"] for i in keep)
dv = statistics.mean(best[i]["net"] for i in keep)
print(f"excl-3: baseline NET {db:+.3f} (n={len(keep)}), best NET {dv:+.3f}, delta {dv - db:+.3f}")

print("\n--- h2-only: changed trades and their deltas (best vs baseline) ---")
ch = [(trades[i]['idx'], round(best[i]['net'] - base[i]['net'], 2))
      for i in range(len(trades)) if trades[i]["idx"] > 2500
      and abs(best[i]["net"] - base[i]["net"]) > 1e-9]
print(f"h2 changed: {len(ch)} trades: {ch}")
