"""Sensitivity of the F-z winner (fund_z gate) + sized maxDD + 1h-liq exploratory.

Checks (parameter-neighborhood robustness, not new variant mining):
- threshold grid 0.0/0.25/0.5/0.75/1.0 (window fixed 168h)
- z-window grid 120/168/336 (threshold fixed 0.5)
- sized equity maxDD (0.5% risk per trade) baseline vs F-z
- exploratory: trailing-6h long-liq notional (Coinalyze 1h, only >=2026-04-03)
  Spearman vs realized on overlapping baseline trades.
"""
import bisect
import json
import statistics
import sys

import polars as pl

sys.path.insert(0, "C:/User/projects/hl-swing-bot/src")
from hl_swing_bot.backtest import (  # noqa: E402
    BTSignal, _composite_score, _compute_features_at, _resolve_outcome, HOUR_MS,
)
from hl_swing_bot.features import HourlyBar, MIN_BARS  # noqa: E402
from hl_swing_bot.signal import (  # noqa: E402
    COOLDOWN_OPP_DIR_MIN, COOLDOWN_SAME_DIR_MIN,
    FIRE_MOVE_PER_ATR_MIN, FIRE_SCORE_MIN, FIRE_VOL_Z_MIN,
    SIGNAL_TTL_HOURS, STOP_ATR_MULT, TARGET_ATR_MULT,
)

SCRATCH = "C:/User/projects/hl-swing-bot/scratch"
COST = 0.19
SPLIT_IDX = 2500

df = pl.read_parquet("C:/User/projects/hl-swing-bot/data/hist_1h.parquet")
bars = [
    HourlyBar(hour_ms=int(r["open_time_ms"]), open=float(r["open"]),
              high=float(r["high"]), low=float(r["low"]), close=float(r["close"]),
              volume=float(r["volume"]), trades=int(r["trades"]))
    for r in df.iter_rows(named=True)
]

fund = json.load(open(f"{SCRATCH}/funding_btc.json"))
f_times = [rec["time"] for rec in fund]
f_rates = [float(rec["fundingRate"]) for rec in fund]


def fund_z(at_ms, window):
    k = bisect.bisect_right(f_times, at_ms)
    if k < window + 1:
        return None
    f1 = f_rates[k - 1]
    hist = f_rates[k - window:k]
    mu = statistics.mean(hist)
    sd = statistics.pstdev(hist)
    return (f1 - mu) / sd if sd > 1e-12 else 0.0


def run(gate=None):
    signals = []
    last_dir, last_idx = None, -10_000
    for i in range(MIN_BARS, len(bars)):
        f = _compute_features_at(bars, i)
        if not f:
            continue
        if f["ret_1h"] > 0:
            continue
        elapsed_min = (i - last_idx) * 60
        if last_dir == "SHORT" and elapsed_min < COOLDOWN_SAME_DIR_MIN:
            continue
        score = _composite_score(f)
        if not (score >= FIRE_SCORE_MIN and f["move_per_atr"] >= FIRE_MOVE_PER_ATR_MIN
                and f["vol_z_168"] >= FIRE_VOL_Z_MIN and f["trend_4h"] <= -1):
            continue
        close_ms = bars[i].hour_ms + HOUR_MS
        if gate is not None and not gate(close_ms):
            continue
        atr, entry = f["atr_1h"], f["close"]
        sig = BTSignal(idx=i, bar_close_ms=close_ms, direction="SHORT",
                       entry=entry, stop=entry + STOP_ATR_MULT * atr,
                       target=entry - TARGET_ATR_MULT * atr, score=score,
                       expires_idx=i + SIGNAL_TTL_HOURS)
        _resolve_outcome(bars, sig, ttl_bars=SIGNAL_TTL_HOURS)
        signals.append(sig)
        last_dir, last_idx = "SHORT", i
    return signals


def report(name, sigs):
    def m(ss):
        if not ss:
            return "n=0", None
        nets = [s.realized_pct - COST for s in ss]
        return f"n={len(ss)} net={statistics.mean(nets):+.3f}", statistics.mean(nets)
    f_, _ = m(sigs)
    h1s, h1 = m([s for s in sigs if s.idx <= SPLIT_IDX])
    h2s, h2 = m([s for s in sigs if s.idx > SPLIT_IDX])
    print(f"{name:24s} FULL[{f_}]  H1[{h1s}]  H2[{h2s}]")


def maxdd(sigs):
    eq, peak, dd = 1.0, 1.0, 0.0
    for s in sorted(sigs, key=lambda x: x.idx):
        stop_pct = (s.stop / s.entry - 1) * 100  # positive for short
        r = (s.realized_pct - COST) / stop_pct
        eq *= 1 + 0.005 * r
        peak = max(peak, eq)
        dd = max(dd, (peak - eq) / peak)
    return eq, dd


print("--- threshold grid (window=168) ---")
for thr in [0.0, 0.25, 0.5, 0.75, 1.0]:
    report(f"fund_z>{thr}", run(lambda ms, t=thr: (fund_z(ms, 168) or -9) > t))

print("--- window grid (thr=0.5) ---")
for w in [120, 168, 336]:
    report(f"win={w} fund_z>0.5", run(lambda ms, ww=w: (fund_z(ms, ww) or -9) > 0.5))

base = run(None)
fz = run(lambda ms: (fund_z(ms, 168) or -9) > 0.5)
eb, db = maxdd(base)
ez, dz = maxdd(fz)
print(f"\nsized 0.5%-risk: baseline eq={eb:.4f} maxDD={db:.2%} | F-z eq={ez:.4f} maxDD={dz:.2%}")

# --- exploratory: 1h long-liq trailing 6h on overlap trades ---
liq1 = json.load(open(f"{SCRATCH}/liq_1h.json"))
lt = [r["t"] * 1000 for r in liq1]
ll = [float(r["l"]) for r in liq1]


def liq6(at_ms):
    k = bisect.bisect_right(lt, at_ms)
    if k < 6:
        return None
    return sum(ll[k - 6:k])


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for m_ in range(i, j + 1):
                rk[order[m_]] = (i + j) / 2 + 1
            i = j + 1
        return rk
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


pairs = [(liq6(s.bar_close_ms), s.realized_pct) for s in base if liq6(s.bar_close_ms) is not None]
print(f"\nexploratory 1h-liq overlap: n={len(pairs)} "
      f"rho={spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) > 4 else float('nan'):+.3f}")
