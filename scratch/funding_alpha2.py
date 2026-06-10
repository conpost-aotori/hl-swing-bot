"""Funding-as-alpha lens, driven from the cached per-bar features.
All results NET of 0.19% round-trip cost; split-half = signal idx <=2500 vs >2500."""
import statistics
import sys

import duckdb

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")
from hl_swing_bot.backtest import BTSignal, _resolve_outcome, HOUR_MS
from hl_swing_bot.features import HourlyBar, MIN_BARS, robust_z
from hl_swing_bot.signal import (
    COOLDOWN_OPP_DIR_MIN, COOLDOWN_SAME_DIR_MIN,
    FIRE_MOVE_PER_ATR_MIN, FIRE_SCORE_MIN, FIRE_VOL_Z_MIN,
    SIGNAL_TTL_HOURS, STOP_ATR_MULT, TARGET_ATR_MULT,
)

COST = 0.19
H = 3600 * 1000
BASE_RATE = 1.25e-05  # HL baseline hourly funding (~10.95% APR)

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

f_now = [0.0] * n
f_z = [0.0] * n
f_ma24 = [0.0] * n  # trailing 24h mean funding (incl. current) — persistence measure
for i, b in enumerate(bars):
    c = b.hour_ms + H
    f_now[i] = rate_by_hour.get(c, 0.0)
    hist = [rate_by_hour[c - k * H] for k in range(1, 25) if (c - k * H) in rate_by_hour]
    f_z[i] = robust_z(f_now[i], hist) if len(hist) >= 5 else 0.0
    w = [rate_by_hour[c - k * H] for k in range(0, 24) if (c - k * H) in rate_by_hour]
    f_ma24[i] = statistics.mean(w) if w else 0.0


def funding_pnl_pct(sig: BTSignal) -> float:
    c_entry = bars[sig.idx].hour_ms + H
    if sig.exit_idx is None:
        return 0.0
    c_last = bars[sig.exit_idx].hour_ms + (H if sig.status == "EXPIRED" else 0)
    t = c_entry + H
    tot = 0.0
    while t <= c_last:
        tot += rate_by_hour.get(t, 0.0)
        t += H
    return tot * 100  # short receives positive funding


def run(variant: dict) -> list[BTSignal]:
    signals: list[BTSignal] = []
    last_dir = None
    last_idx = -10_000
    for i in range(MIN_BARS, n):
        f = FEAT.get(i)
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
        score = f["score"]
        if variant.get("funding_in_score"):
            bonus = max(0.0, 1.0 - min(abs(f_z[i]), 3.0) / 3.0)
            score = score - 0.10 + 0.10 * bonus
        ok = (score >= FIRE_SCORE_MIN and f["move_per_atr"] >= FIRE_MOVE_PER_ATR_MIN
              and f["vol_z"] >= FIRE_VOL_Z_MIN and f["trend"] <= -1)
        if ok and variant.get("z_max") is not None and abs(f_z[i]) > variant["z_max"]:
            ok = False
        if ok and variant.get("z_min") is not None and f_z[i] < variant["z_min"]:
            ok = False
        if ok and variant.get("sign") == "pos" and not (f_now[i] > 0):
            ok = False
        if ok and variant.get("sign") == "neg" and not (f_now[i] < 0):
            ok = False
        if ok and variant.get("apr_min") is not None and f_now[i] * 24 * 365 * 100 < variant["apr_min"]:
            ok = False
        if ok and variant.get("apr_max") is not None and f_now[i] * 24 * 365 * 100 > variant["apr_max"]:
            ok = False
        if ok and variant.get("ma24_apr_min") is not None and f_ma24[i] * 24 * 365 * 100 < variant["ma24_apr_min"]:
            ok = False
        if ok and variant.get("ma24_apr_max") is not None and f_ma24[i] * 24 * 365 * 100 > variant["ma24_apr_max"]:
            ok = False
        if not ok:
            continue
        atr = f["atr"]
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
        print(f"{label:48s} n=0")
        return
    h1 = stats([s for s in signals if s.idx <= 2500], with_f)
    h2 = stats([s for s in signals if s.idx > 2500], with_f)
    print(f"{label:48s} n={st['n']:3d} hit={st['hit']*100:4.1f}% net={st['net']:+.3f} "
          f"netR={st['netR']:+.3f} mdd={st['mdd']*100:4.1f}% | "
          f"H1 n={h1.get('n',0):3d} net={h1.get('net',0):+.3f} | "
          f"H2 n={h2.get('n',0):3d} net={h2.get('net',0):+.3f}")


print("=== (1) baseline ===")
base = run({})
show("baseline", base)

print("\n=== (2) |funding_z|<=Z gate (live FIRE_FUNDING_Z_MAX) ===")
for z in (2.5, 2.0, 1.5, 1.0):
    show(f"gate |z|<={z}", run({"z_max": z}))

print("\n=== (3) funding_bonus in score ===")
show("funding_bonus in score", run({"funding_in_score": True}))
show("bonus in score + |z|<=2.5", run({"funding_in_score": True, "z_max": 2.5}))

print("\n=== (4) sign filter ===")
show("funding > 0 only", run({"sign": "pos"}))
show("funding < 0 only", run({"sign": "neg"}))

print("\n=== (5) APR bands (spot funding at entry) ===")
for amin in (2.0, 5.0, 8.0, 10.0, 10.95, 15.0):
    show(f"apr >= {amin}%", run({"apr_min": amin}))
for amax in (10.95, 5.0, 0.0):
    show(f"apr <= {amax}%", run({"apr_max": amax}))

print("\n=== (5b) 24h-mean funding APR bands (persistent crowding) ===")
for amin in (2.0, 5.0, 8.0, 10.0):
    show(f"ma24 apr >= {amin}%", run({"ma24_apr_min": amin}))
for amax in (10.0, 5.0, 2.0, 0.0):
    show(f"ma24 apr <= {amax}%", run({"ma24_apr_max": amax}))

print("\n=== (5c) funding_z directional (shorts into RISING funding) ===")
for zmin in (-1.0, -0.5, 0.0, 0.5, 1.0):
    show(f"z >= {zmin}", run({"z_min": zmin}))

print("\n=== (6) funding P&L netted into trade returns ===")
show("baseline + funding pnl", base, with_f=True)
fp = [funding_pnl_pct(s) for s in base]
print(f"  per-trade funding pnl: mean={statistics.mean(fp):+.4f}% med={statistics.median(fp):+.4f}% "
      f"min={min(fp):+.4f}% max={max(fp):+.4f}%")
holds = [s.exit_idx - s.idx for s in base]
print(f"  hold bars: mean={statistics.mean(holds):.1f} med={statistics.median(holds)}")

print("\n--- baseline entries: funding context ---")
for s in base:
    apr = f_now[s.idx] * 24 * 365 * 100
    ma_apr = f_ma24[s.idx] * 24 * 365 * 100
    print(f"idx={s.idx:4d} {s.status:7s} ret={s.realized_pct - COST:+.2f}% z={f_z[s.idx]:+5.2f} "
          f"apr={apr:+6.1f}% ma24apr={ma_apr:+6.1f}% fpnl={funding_pnl_pct(s):+.3f}%")
