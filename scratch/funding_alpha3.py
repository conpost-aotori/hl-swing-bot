"""Follow-up: threshold sensitivity, combos, 1-bar-lagged funding robustness,
and funding P&L netted into the winner."""
import statistics
import sys

import duckdb

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")
from hl_swing_bot.backtest import BTSignal, _resolve_outcome, HOUR_MS
from hl_swing_bot.features import HourlyBar, MIN_BARS, robust_z
from hl_swing_bot.signal import (
    COOLDOWN_SAME_DIR_MIN, FIRE_MOVE_PER_ATR_MIN, FIRE_SCORE_MIN, FIRE_VOL_Z_MIN,
    SIGNAL_TTL_HOURS, STOP_ATR_MULT, TARGET_ATR_MULT,
)

COST = 0.19
H = 3600 * 1000

con = duckdb.connect()
bar_rows = con.execute(
    "SELECT open_time_ms, open, high, low, close, volume, trades "
    "FROM read_parquet('C:/User/projects/hl-swing-bot/data/hist_1h.parquet') ORDER BY open_time_ms"
).fetchall()
bars = [HourlyBar(hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]), low=float(r[3]),
                  close=float(r[4]), volume=float(r[5]), trades=int(r[6])) for r in bar_rows]
n = len(bars)

feat_rows = con.execute(
    "SELECT idx, close, atr, ret_1h, move_per_atr, vol_z, trend, score "
    "FROM read_parquet('C:/User/projects/hl-swing-bot/scratch/features_cache.parquet') ORDER BY idx"
).fetchall()
FEAT = {int(r[0]): {"close": r[1], "atr": r[2], "ret_1h": r[3], "move_per_atr": r[4],
                    "vol_z": r[5], "trend": int(r[6]), "score": r[7]} for r in feat_rows}

f_rows = con.execute(
    "SELECT time_ms, funding_rate FROM read_parquet('C:/User/projects/hl-swing-bot/scratch/funding_1h.parquet') ORDER BY time_ms"
).fetchall()
rate_by_hour = {int(t) // H * H: float(r) for t, r in f_rows}
hmin, hmax = min(rate_by_hour), max(rate_by_hour)
last = rate_by_hour[hmin]
h = hmin
while h <= hmax:
    if h in rate_by_hour:
        last = rate_by_hour[h]
    else:
        rate_by_hour[h] = last
    h += H


def build_arrays(lag_hours: int):
    fn = [0.0] * n
    fz = [0.0] * n
    fma = [0.0] * n
    for i, b in enumerate(bars):
        c = b.hour_ms + H - lag_hours * H
        fn[i] = rate_by_hour.get(c, 0.0)
        hist = [rate_by_hour[c - k * H] for k in range(1, 25) if (c - k * H) in rate_by_hour]
        fz[i] = robust_z(fn[i], hist) if len(hist) >= 5 else 0.0
        w = [rate_by_hour[c - k * H] for k in range(0, 24) if (c - k * H) in rate_by_hour]
        fma[i] = statistics.mean(w) if w else 0.0
    return fn, fz, fma


F0 = build_arrays(0)   # funding settled at bar close (live-faithful)
F1 = build_arrays(1)   # 1h stale (pessimistic)


def funding_pnl_pct(sig):
    c_entry = bars[sig.idx].hour_ms + H
    c_last = bars[sig.exit_idx].hour_ms + (H if sig.status == "EXPIRED" else 0)
    t = c_entry + H
    tot = 0.0
    while t <= c_last:
        tot += rate_by_hour.get(t, 0.0)
        t += H
    return tot * 100


def run(variant, arrays):
    fn, fz, fma = arrays
    signals = []
    last_idx = -10_000
    have_last = False
    for i in range(MIN_BARS, n):
        f = FEAT.get(i)
        if not f:
            continue
        if f["ret_1h"] > 0:
            continue
        if have_last and (i - last_idx) * 60 < COOLDOWN_SAME_DIR_MIN:
            continue
        ok = (f["score"] >= FIRE_SCORE_MIN and f["move_per_atr"] >= FIRE_MOVE_PER_ATR_MIN
              and f["vol_z"] >= FIRE_VOL_Z_MIN and f["trend"] <= -1)
        apr = fn[i] * 24 * 365 * 100
        ma_apr = fma[i] * 24 * 365 * 100
        if ok and variant.get("z_max") is not None and abs(fz[i]) > variant["z_max"]:
            ok = False
        if ok and variant.get("apr_min") is not None and apr < variant["apr_min"]:
            ok = False
        if ok and variant.get("ma24_apr_min") is not None and ma_apr < variant["ma24_apr_min"]:
            ok = False
        if not ok:
            continue
        atr, entry = f["atr"], f["close"]
        sig = BTSignal(idx=i, bar_close_ms=bars[i].hour_ms + HOUR_MS, direction="SHORT",
                       entry=entry, stop=entry + STOP_ATR_MULT * atr,
                       target=entry - TARGET_ATR_MULT * atr, score=f["score"],
                       expires_idx=i + SIGNAL_TTL_HOURS)
        _resolve_outcome(bars, sig, ttl_bars=SIGNAL_TTL_HOURS)
        signals.append(sig)
        last_idx = i
        have_last = True
    return signals


def stats(signals, with_f=False):
    if not signals:
        return {"n": 0}
    rets, rs = [], []
    for s in signals:
        r = s.realized_pct - COST + (funding_pnl_pct(s) if with_f else 0.0)
        rets.append(r)
        rs.append(r / (abs(s.stop - s.entry) / s.entry * 100))
    tp = sum(1 for s in signals if s.status == "HIT_TP")
    eq, peak, mdd = 1.0, 1.0, 0.0
    for rr in rs:
        eq *= (1 + 0.005 * rr)
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)
    return {"n": len(signals), "hit": tp / len(signals), "net": statistics.mean(rets),
            "netR": statistics.mean(rs), "mdd": mdd}


def show(label, signals, with_f=False):
    st = stats(signals, with_f)
    if st["n"] == 0:
        print(f"{label:52s} n=0")
        return
    h1 = stats([s for s in signals if s.idx <= 2500], with_f)
    h2 = stats([s for s in signals if s.idx > 2500], with_f)
    print(f"{label:52s} n={st['n']:3d} hit={st['hit']*100:4.1f}% net={st['net']:+.3f} "
          f"netR={st['netR']:+.3f} mdd={st['mdd']*100:4.1f}% | "
          f"H1 n={h1.get('n',0):3d} net={h1.get('net',0):+.3f} | "
          f"H2 n={h2.get('n',0):3d} net={h2.get('net',0):+.3f}")


print("=== APR threshold sensitivity (at-close funding) ===")
for a in (6.0, 7.0, 8.0, 9.0, 10.0):
    show(f"apr >= {a}%", run({"apr_min": a}, F0))

print("\n=== |z| gate sensitivity ===")
for z in (0.75, 1.0, 1.25, 1.5):
    show(f"|z| <= {z}", run({"z_max": z}, F0))

print("\n=== combos ===")
show("apr>=8 & |z|<=1.5", run({"apr_min": 8.0, "z_max": 1.5}, F0))
show("apr>=8 & |z|<=1.0", run({"apr_min": 8.0, "z_max": 1.0}, F0))
show("ma24apr>=5 & |z|<=1.5", run({"ma24_apr_min": 5.0, "z_max": 1.5}, F0))
show("ma24apr>=5 & |z|<=1.0", run({"ma24_apr_min": 5.0, "z_max": 1.0}, F0))
show("apr>=8 & ma24apr>=5", run({"apr_min": 8.0, "ma24_apr_min": 5.0}, F0))

print("\n=== 1h-stale funding (pessimistic lag robustness) ===")
show("LAG1 apr >= 8%", run({"apr_min": 8.0}, F1))
show("LAG1 apr >= 10%", run({"apr_min": 10.0}, F1))
show("LAG1 |z| <= 1.0", run({"z_max": 1.0}, F1))
show("LAG1 |z| <= 1.5", run({"z_max": 1.5}, F1))
show("LAG1 ma24apr >= 5%", run({"ma24_apr_min": 5.0}, F1))
show("LAG1 apr>=8 & |z|<=1.5", run({"apr_min": 8.0, "z_max": 1.5}, F1))

print("\n=== winners + funding P&L netted ===")
show("apr>=8 + fundingPnL", run({"apr_min": 8.0}, F0), with_f=True)
show("|z|<=1.0 + fundingPnL", run({"z_max": 1.0}, F0), with_f=True)

print("\n=== quarter-splits for top candidates (extra honesty) ===")
for label, v in (("apr>=8", {"apr_min": 8.0}), ("|z|<=1.0", {"z_max": 1.0}),
                 ("ma24apr>=5", {"ma24_apr_min": 5.0})):
    sigs = run(v, F0)
    for q, (lo, hi) in enumerate(((0, 1250), (1251, 2500), (2501, 3751), (3752, 99999)), 1):
        qs = [s for s in sigs if lo <= s.idx <= hi]
        st = stats(qs)
        if st["n"]:
            print(f"  {label:12s} Q{q}: n={st['n']:2d} net={st['net']:+.3f}")
        else:
            print(f"  {label:12s} Q{q}: n=0")
