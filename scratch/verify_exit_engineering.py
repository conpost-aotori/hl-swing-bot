"""Adversarial verification of the EXIT ENGINEERING claim.

Independently re-derives:
  V0 baseline  : SL 1.5 ATR / TP 2.5 ATR / 72h TTL (harness parity check)
  Best variant : SL 1.5 ATR / TP 2.0 ATR / BE stop->entry after bar low
                 trades 1.25 ATR below entry (applied from NEXT bar) / 72h TTL

Checks: (a) reproduction, (b) split-half (idx<=2500 vs >2500), (c) n per half,
(d) +/-25% parameter nudges on BE trigger and TP, (e) where the gain comes from.

NET = gross mean realized - 0.19 (round-trip cost), per the rules of evidence.
"""
import json
import statistics
import sys

import duckdb

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")
from hl_swing_bot.backtest import run_backtest  # noqa: E402
from hl_swing_bot.features import HourlyBar  # noqa: E402

COST = 0.19  # round-trip cost in %
TTL = 72
SL_MULT = 1.5

rows = duckdb.sql(
    "SELECT open_time_ms, open, high, low, close, volume, trades "
    "FROM 'C:/User/projects/hl-swing-bot/data/hist_1h.parquet' ORDER BY open_time_ms"
).fetchall()
bars = [
    HourlyBar(hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
              low=float(r[3]), close=float(r[4]), volume=float(r[5]),
              trades=int(r[6]))
    for r in rows
]
print(f"bars loaded: {len(bars)}")

# ---- Step 1: harness baseline (gross, slippage 0) --------------------------
res = run_backtest(bars, slippage_bps=0.0, short_only=True)
sigs = res["signals"]
print(f"harness short-only: n={res['n_signals']} TP={res['tp_count']} "
      f"SL={res['sl_count']} EXP={res['expired_count']} "
      f"gross_mean={res['expectancy_pct_post_slippage']:.4f} "
      f"NET={res['expectancy_pct_post_slippage'] - COST:.4f}")

# Extract per-signal primitives (entry, atr) from the harness signal list.
trades = []
for s in sigs:
    atr = (s["stop"] - s["entry"]) / SL_MULT  # SHORT: stop above entry
    assert atr > 0
    trades.append({"idx": s["idx"], "entry": s["entry"], "atr": atr})


# ---- Step 2: my own resolver ------------------------------------------------
def resolve(tr, tp_mult, be_mult):
    """Conservative resolver. Per bar: stop check first, then TP, THEN arm BE
    (so BE takes effect from the next bar). Returns (gross_pct, status, exit_idx).
    """
    entry, atr, idx = tr["entry"], tr["atr"], tr["idx"]
    stop = entry + SL_MULT * atr
    tp = entry - tp_mult * atr if tp_mult is not None else None
    be_trig = entry - be_mult * atr if be_mult is not None else None
    be_armed = False
    end_idx = min(idx + TTL, len(bars) - 1)
    for j in range(idx + 1, end_idx + 1):
        b = bars[j]
        if b.high >= stop:
            g = (entry / stop - 1) * 100
            st = "BE" if be_armed and abs(stop - entry) < 1e-9 else "SL"
            return g, st, j
        if tp is not None and b.low <= tp:
            return (entry / tp - 1) * 100, "TP", j
        if be_trig is not None and not be_armed and b.low <= be_trig:
            stop = entry
            be_armed = True
    return (entry / bars[end_idx].close - 1) * 100, "EXP", end_idx


def evaluate(tp_mult, be_mult, label, verbose=False):
    recs = []
    for tr in trades:
        g, st, xj = resolve(tr, tp_mult, be_mult)
        net = g - COST
        atr_pct = tr["atr"] / tr["entry"] * 100
        recs.append({"idx": tr["idx"], "gross": g, "net": net, "status": st,
                     "exit_idx": xj, "stop_pct": SL_MULT * atr_pct})
    n = len(recs)
    nets = [r["net"] for r in recs]
    h1 = [r["net"] for r in recs if r["idx"] <= 2500]
    h2 = [r["net"] for r in recs if r["idx"] > 2500]
    cnt = {k: sum(1 for r in recs if r["status"] == k) for k in ("TP", "SL", "BE", "EXP")}
    # sized equity curve: 0.5% risk per trade, ordered by exit time
    eq, peak, maxdd = 1.0, 1.0, 0.0
    for r in sorted(recs, key=lambda r: (r["exit_idx"], r["idx"])):
        net_r = r["net"] / r["stop_pct"]
        eq *= (1 + 0.005 * net_r)
        peak = max(peak, eq)
        maxdd = max(maxdd, 1 - eq / peak)
    avg_r = statistics.mean(r["net"] / r["stop_pct"] for r in recs)
    out = {
        "label": label, "n": n, "n_h1": len(h1), "n_h2": len(h2),
        "net_full": round(statistics.mean(nets), 4),
        "net_h1": round(statistics.mean(h1), 4) if h1 else None,
        "net_h2": round(statistics.mean(h2), 4) if h2 else None,
        "counts": cnt, "maxdd_sized_pct": round(maxdd * 100, 2),
        "avg_net_R": round(avg_r, 4),
        "median_dur_h": statistics.median(r["exit_idx"] - r["idx"] for r in recs),
    }
    print(json.dumps(out))
    return out, recs


print("\n--- replication ---")
base_out, base_recs = evaluate(2.5, None, "V0 my-resolver baseline (TP2.5, no BE)")
best_out, best_recs = evaluate(2.0, 1.25, "BEST claimed (TP2.0, BE1.25)")

print("\n--- +/-25% nudges on each key parameter ---")
evaluate(2.0, 1.25 * 0.75, "BE -25% (0.9375), TP2.0")
evaluate(2.0, 1.25 * 1.25, "BE +25% (1.5625), TP2.0")
evaluate(2.0 * 0.75, 1.25, "TP -25% (1.5), BE1.25")
evaluate(2.0 * 1.25, 1.25, "TP +25% (2.5), BE1.25")

print("\n--- claimed plateau grid (BE x TP) ---")
for be in (1.0, 1.25, 1.4):
    for tp in (1.75, 2.0, 2.25, 2.5):
        evaluate(tp, be, f"BE{be} TP{tp}")

print("\n--- spot checks from the claim list ---")
evaluate(2.5, 1.0, "V1 BE1.0 TP2.5")
evaluate(2.5, 0.75, "V1b BE0.75 TP2.5")
evaluate(2.5, 1.25, "V1c BE1.25 TP2.5")
evaluate(2.0, None, "TP2.0 alone, no BE")

print("\n--- where does the gain come from? (per-trade delta vs V0) ---")
deltas = []
for b0, b1 in zip(base_recs, best_recs):
    assert b0["idx"] == b1["idx"]
    d = b1["net"] - b0["net"]
    if abs(d) > 1e-9:
        deltas.append((b0["idx"], b0["status"], b1["status"], round(d, 3)))
n_changed = len(deltas)
tot = sum(d[3] for d in deltas)
print(f"changed trades: {n_changed}/{len(base_recs)}, total delta {tot:.2f} "
      f"=> mean delta {tot / len(base_recs):.4f}")
pos = sorted(deltas, key=lambda d: -d[3])
print("top contributors:", pos[:6])
print("worst contributors:", pos[-6:])
h1d = sum(d[3] for d in deltas if d[0] <= 2500)
h2d = sum(d[3] for d in deltas if d[0] > 2500)
n1 = sum(1 for r in base_recs if r["idx"] <= 2500)
n2 = len(base_recs) - n1
print(f"delta by half: h1 {h1d:.2f} over {n1} trades, h2 {h2d:.2f} over {n2} trades")

# simple bootstrap on per-trade net of best variant (is mean>0 fragile?)
import random
random.seed(7)
nets_best = [r["net"] for r in best_recs]
boots = []
for _ in range(10000):
    smp = [random.choice(nets_best) for _ in nets_best]
    boots.append(statistics.mean(smp))
boots.sort()
print(f"bootstrap mean NET best: p05={boots[500]:.3f} p50={boots[5000]:.3f} "
      f"p95={boots[9500]:.3f}, P(mean<=0)={sum(1 for b in boots if b <= 0) / len(boots):.3f}")
nets_base = [r["net"] for r in base_recs]
dd = [b1["net"] - b0["net"] for b0, b1 in zip(base_recs, best_recs)]
bootd = []
for _ in range(10000):
    smp = [random.choice(dd) for _ in dd]
    bootd.append(statistics.mean(smp))
bootd.sort()
print(f"bootstrap mean DELTA (best-base): p05={bootd[500]:.3f} p50={bootd[5000]:.3f} "
      f"p95={bootd[9500]:.3f}, P(delta<=0)={sum(1 for b in bootd if b <= 0) / len(bootd):.3f}")
