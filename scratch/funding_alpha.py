"""Funding-as-alpha lens for the short-only BTC 1h swing strategy.

Tests (all NET of 0.19% round-trip cost, split-half validated for winners):
 1. baseline reproduction (short-only, no funding info)
 2. existing FIRE_FUNDING_Z_MAX=2.5 gate (|funding_z_24| <= 2.5)
 3. funding-in-score (live funding_bonus instead of constant 1.0)
 4. sign filter: short only when funding > 0 / funding < 0
 5. APR level bands (funding_apr thresholds)
 6. funding P&L netted into each trade's return
"""
import statistics
import sys

import duckdb

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")
from hl_swing_bot.backtest import BTSignal, _compute_features_at, _composite_score, _resolve_outcome, HOUR_MS
from hl_swing_bot.features import HourlyBar, MIN_BARS, robust_z
from hl_swing_bot.signal import (
    COOLDOWN_OPP_DIR_MIN, COOLDOWN_SAME_DIR_MIN,
    FIRE_MOVE_PER_ATR_MIN, FIRE_SCORE_MIN, FIRE_VOL_Z_MIN,
    SIGNAL_TTL_HOURS, STOP_ATR_MULT, TARGET_ATR_MULT,
)

COST = 0.19  # round-trip cost %
H = 3600 * 1000
BASELINE_HOURLY = 1.25e-05  # HL baseline funding (~10.95% APR)

con = duckdb.connect()
bar_rows = con.execute(
    "SELECT open_time_ms, open, high, low, close, volume, trades "
    "FROM read_parquet('C:/User/projects/hl-swing-bot/data/hist_1h.parquet') ORDER BY open_time_ms"
).fetchall()
bars = [HourlyBar(hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]), low=float(r[3]),
                  close=float(r[4]), volume=float(r[5]), trades=int(r[6])) for r in bar_rows]
print(f"bars: {len(bars)}")

f_rows = con.execute(
    "SELECT time_ms, funding_rate FROM read_parquet('C:/User/projects/hl-swing-bot/scratch/funding_1h.parquet') ORDER BY time_ms"
).fetchall()
# floor funding timestamps to the hour boundary
rate_by_hour: dict[int, float] = {}
for t, r in f_rows:
    rate_by_hour[int(t) // H * H] = float(r)
print(f"funding hours: {len(rate_by_hour)}")

# forward-fill any gaps over the continuous hour range
hmin, hmax = min(rate_by_hour), max(rate_by_hour)
filled = 0
last = rate_by_hour[hmin]
h = hmin
while h <= hmax:
    if h in rate_by_hour:
        last = rate_by_hour[h]
    else:
        rate_by_hour[h] = last
        filled += 1
    h += H
print(f"gap hours forward-filled: {filled}")

# per-bar funding context (look-ahead-safe: funding settled AT bar close)
n = len(bars)
f_now = [0.0] * n      # funding rate settled at bar close
f_z = [0.0] * n        # robust z vs prior 24 hourly fundings
for i, b in enumerate(bars):
    c = b.hour_ms + H
    f_now[i] = rate_by_hour.get(c, 0.0)
    hist = [rate_by_hour[c - k * H] for k in range(1, 25) if (c - k * H) in rate_by_hour]
    f_z[i] = robust_z(f_now[i], hist) if len(hist) >= 5 else 0.0

nz = sum(1 for z in f_z if z != 0.0)
pos = sum(1 for i in range(n) if f_now[i] > 0)
neg = sum(1 for i in range(n) if f_now[i] < 0)
above_base = sum(1 for i in range(n) if f_now[i] > BASELINE_HOURLY + 1e-12)
print(f"f_z nonzero: {nz}/{n} | funding>0: {pos} | funding<0: {neg} | above baseline: {above_base}")
print(f"funding rate distribution: min={min(f_now):.3e} med={statistics.median(f_now):.3e} max={max(f_now):.3e}")


def funding_pnl_pct(sig: BTSignal) -> float:
    """Funding received (+) / paid (-) by a SHORT over the hold, in %.
    Boundaries strictly after entry close; for TP/SL exclude the exit bar's
    closing boundary (exit happens intrabar), for EXPIRED include it."""
    c_entry = bars[sig.idx].hour_ms + H
    if sig.exit_idx is None:
        return 0.0
    if sig.status == "EXPIRED":
        c_last = bars[sig.exit_idx].hour_ms + H
    else:
        c_last = bars[sig.exit_idx].hour_ms  # boundary at exit-bar open
    t = c_entry + H
    tot = 0.0
    while t <= c_last:
        tot += rate_by_hour.get(t, 0.0)
        t += H
    return tot * 100  # short receives positive funding


def run(variant: dict) -> list[BTSignal]:
    """Re-run the short-only loop with a funding hook so the cooldown chain is
    simulated faithfully (a funding-blocked signal does not reset cooldown,
    matching live signal.py)."""
    signals: list[BTSignal] = []
    last_dir = None
    last_idx = -10_000
    for i in range(MIN_BARS, len(bars)):
        f = _compute_features_at(bars, i)
        if not f:
            continue
        direction = "LONG" if f["ret_1h"] > 0 else "SHORT"
        if direction == "LONG":
            continue
        elapsed_min = (i - last_idx) * 60
        if last_dir is not None:
            if last_dir == direction and elapsed_min < COOLDOWN_SAME_DIR_MIN:
                continue
            if last_dir != direction and elapsed_min < COOLDOWN_OPP_DIR_MIN:
                continue
        score = _composite_score(f)
        if variant.get("funding_in_score"):
            bonus = max(0.0, 1.0 - min(abs(f_z[i]), 3.0) / 3.0)
            score = score - 0.10 + 0.10 * bonus
        ok = (score >= FIRE_SCORE_MIN and f["move_per_atr"] >= FIRE_MOVE_PER_ATR_MIN
              and f["vol_z_168"] >= FIRE_VOL_Z_MIN and f["trend_4h"] <= -1)
        if ok and variant.get("z_max") is not None and abs(f_z[i]) > variant["z_max"]:
            ok = False
        if ok and variant.get("sign") == "pos" and not (f_now[i] > 0):
            ok = False
        if ok and variant.get("sign") == "neg" and not (f_now[i] < 0):
            ok = False
        if ok and variant.get("apr_min") is not None and f_now[i] * 24 * 365 * 100 < variant["apr_min"]:
            ok = False
        if ok and variant.get("apr_max") is not None and f_now[i] * 24 * 365 * 100 > variant["apr_max"]:
            ok = False
        if not ok:
            continue
        atr = f["atr_1h"]
        entry = f["close"]
        sig = BTSignal(idx=i, bar_close_ms=bars[i].hour_ms + HOUR_MS, direction=direction,
                       entry=entry, stop=entry + STOP_ATR_MULT * atr,
                       target=entry - TARGET_ATR_MULT * atr, score=score,
                       expires_idx=i + SIGNAL_TTL_HOURS)
        _resolve_outcome(bars, sig, ttl_bars=SIGNAL_TTL_HOURS)
        signals.append(sig)
        last_dir = direction
        last_idx = i
    return signals


def stats(signals: list[BTSignal], with_funding_pnl: bool = False, label: str = "") -> dict:
    if not signals:
        return {"n": 0}
    rets = []
    rs = []
    for s in signals:
        r = s.realized_pct - COST
        if with_funding_pnl:
            r += funding_pnl_pct(s)
        rets.append(r)
        stop_pct = abs(s.stop - s.entry) / s.entry * 100
        rs.append(r / stop_pct)
    tp = sum(1 for s in signals if s.status == "HIT_TP")
    # sized equity curve: 0.5% risk per trade
    eq, peak, mdd = 1.0, 1.0, 0.0
    for rr in rs:
        eq *= (1 + 0.005 * rr)
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)
    return {"n": len(signals), "hit": tp / len(signals), "net": statistics.mean(rets),
            "med": statistics.median(rets), "netR": statistics.mean(rs), "mdd": mdd}


def show(label: str, signals: list[BTSignal], with_f: bool = False):
    st = stats(signals, with_funding_pnl=with_f)
    if st["n"] == 0:
        print(f"{label:55s} n=0")
        return st
    h1 = stats([s for s in signals if s.idx <= 2500], with_funding_pnl=with_f)
    h2 = stats([s for s in signals if s.idx > 2500], with_funding_pnl=with_f)
    print(f"{label:55s} n={st['n']:3d} hit={st['hit']*100:4.1f}% net={st['net']:+.3f}%/tr "
          f"netR={st['netR']:+.3f} mddSized={st['mdd']*100:.1f}% | "
          f"H1 n={h1.get('n',0)} net={h1.get('net',float('nan')):+.3f} | "
          f"H2 n={h2.get('n',0)} net={h2.get('net',float('nan')):+.3f}")
    return st


print("\n=== (1) baseline (short-only, funding ignored) ===")
base = run({})
show("baseline", base)

print("\n=== (2) existing gate |funding_z_24| <= 2.5 ===")
show("gate z<=2.5", run({"z_max": 2.5}))
show("gate z<=1.5", run({"z_max": 1.5}))
show("gate z<=1.0", run({"z_max": 1.0}))

print("\n=== (3) funding in score (live funding_bonus) ===")
show("funding_bonus in score", run({"funding_in_score": True}))
show("funding_bonus + gate z<=2.5", run({"funding_in_score": True, "z_max": 2.5}))

print("\n=== (4) sign filter at signal bar ===")
show("short only when funding > 0", run({"sign": "pos"}))
show("short only when funding < 0", run({"sign": "neg"}))

print("\n=== (5) APR level bands (funding_apr = rate*24*365) ===")
for amin in (5.0, 10.0, 10.95, 15.0, 20.0, 30.0):
    show(f"short only when funding_apr >= {amin}%", run({"apr_min": amin}))
show("funding_apr <= 10.95% (at/below baseline)", run({"apr_max": 10.95}))
show("funding_apr <= 0% (negative only)", run({"apr_max": 0.0}))

print("\n=== (6) funding P&L netted into returns ===")
show("baseline + funding pnl", base, with_f=True)
fp = [funding_pnl_pct(s) for s in base]
print(f"   per-trade funding pnl: mean={statistics.mean(fp):+.4f}% med={statistics.median(fp):+.4f}% "
      f"min={min(fp):+.4f}% max={max(fp):+.4f}%")
holds = [(s.exit_idx - s.idx) for s in base]
print(f"   hold bars: mean={statistics.mean(holds):.1f} med={statistics.median(holds)}")

# diagnostics: funding state at the 49 baseline entries
print("\n--- baseline entries: funding context ---")
for s in base:
    apr = f_now[s.idx] * 24 * 365 * 100
    print(f"idx={s.idx:4d} {s.status:7s} ret={s.realized_pct:+.2f}% z={f_z[s.idx]:+.2f} "
          f"rate={f_now[s.idx]:+.2e} apr={apr:+6.1f}% fpnl={funding_pnl_pct(s):+.3f}%")
