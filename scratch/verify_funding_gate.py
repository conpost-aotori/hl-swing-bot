"""ADVERSARIAL VERIFICATION of the 'funding APR >= 8%' gate claim.

Independent re-derivation:
- bars from data/hist_1h.parquet
- features via repo _compute_features_at (NOT the lens's features_cache)
- funding from my own re-fetch (scratch/verify_funding_btc.json)
- baseline via repo run_backtest(short_only=True) AND my own loop (cross-check)
- gate applied IN-LOOP (live-faithful: blocked signals don't set cooldown)
All nets are realized_pct - 0.19 round-trip cost.
"""
import json
import statistics
import sys

import duckdb

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")
from hl_swing_bot.backtest import (
    BTSignal, HOUR_MS, _compute_features_at, _composite_score,
    _resolve_outcome, run_backtest,
)
from hl_swing_bot.features import HourlyBar, MIN_BARS
from hl_swing_bot.signal import (
    COOLDOWN_OPP_DIR_MIN, COOLDOWN_SAME_DIR_MIN,
    FIRE_MOVE_PER_ATR_MIN, FIRE_SCORE_MIN, FIRE_VOL_Z_MIN,
    SIGNAL_TTL_HOURS, STOP_ATR_MULT, TARGET_ATR_MULT,
)

COST = 0.19
H = 3600 * 1000

con = duckdb.connect()
rows = con.execute(
    "SELECT open_time_ms, open, high, low, close, volume, trades "
    "FROM read_parquet('C:/User/projects/hl-swing-bot/data/hist_1h.parquet') "
    "ORDER BY open_time_ms"
).fetchall()
bars = [HourlyBar(hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
                  low=float(r[3]), close=float(r[4]), volume=float(r[5]),
                  trades=int(r[6])) for r in rows]
n = len(bars)
print(f"bars: {n}")

with open(r"C:\User\projects\hl-swing-bot\scratch\verify_funding_btc.json") as fh:
    fr = json.load(fh)
rate = {int(t) // H * H: float(v) for t, v in fr}

# funding APR (%) seen at bar i close, at-close join and 1h-stale join.
# Forward-fill from the past only (look-ahead-safe); no gaps were found anyway.
apr_close = [None] * n
apr_stale = [None] * n
for i, b in enumerate(bars):
    c = b.hour_ms + H
    rc = rate.get(c)
    rs = rate.get(c - H)
    apr_close[i] = None if rc is None else rc * 24 * 365 * 100
    apr_stale[i] = None if rs is None else rs * 24 * 365 * 100
print("missing at-close joins:", sum(1 for x in apr_close if x is None),
      "missing stale joins:", sum(1 for x in apr_stale if x is None))

# ---- compute features once via the repo's canonical function ----
print("computing features (O(n^2), please wait)...")
FEATS = {}
for i in range(MIN_BARS, n):
    f = _compute_features_at(bars, i)
    if f:
        FEATS[i] = f
print(f"features computed for {len(FEATS)} bars")


def my_run(apr_min=None, apr_src=None):
    """Replicates run_backtest(short_only=True) with an optional in-loop
    funding-APR gate."""
    signals = []
    last_dir, last_idx = None, -10_000
    for i in range(MIN_BARS, n):
        f = FEATS.get(i)
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
        ok = (score >= FIRE_SCORE_MIN
              and f["move_per_atr"] >= FIRE_MOVE_PER_ATR_MIN
              and f["vol_z_168"] >= FIRE_VOL_Z_MIN
              and f["trend_4h"] <= -1)
        if ok and apr_min is not None:
            a = apr_src[i]
            if a is None or a < apr_min:
                ok = False
        if not ok:
            continue
        atr = f["atr_1h"]
        entry = f["close"]
        sig = BTSignal(idx=i, bar_close_ms=bars[i].hour_ms + HOUR_MS,
                       direction=direction, entry=entry,
                       stop=entry + STOP_ATR_MULT * atr,
                       target=entry - TARGET_ATR_MULT * atr,
                       score=score, expires_idx=i + SIGNAL_TTL_HOURS)
        _resolve_outcome(bars, sig, ttl_bars=SIGNAL_TTL_HOURS)
        signals.append(sig)
        last_dir, last_idx = direction, i
    return signals


def stats(sigs):
    if not sigs:
        return {"n": 0, "net": float("nan")}
    nets = [s.realized_pct - COST for s in sigs]
    rs = [(s.realized_pct - COST) / (abs(s.stop - s.entry) / s.entry * 100)
          for s in sigs]
    tp = sum(1 for s in sigs if s.status == "HIT_TP")
    eq, peak, mdd = 1.0, 1.0, 0.0
    for rr in rs:
        eq *= (1 + 0.005 * rr)
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)
    return {"n": len(sigs), "hit": tp / len(sigs),
            "net": statistics.mean(nets), "med": statistics.median(nets),
            "netR": statistics.mean(rs), "mdd": mdd}


def show(label, sigs):
    st = stats(sigs)
    if st["n"] == 0:
        print(f"{label:34s} n=0")
        return st
    h1 = stats([s for s in sigs if s.idx <= 2500])
    h2 = stats([s for s in sigs if s.idx > 2500])
    print(f"{label:34s} n={st['n']:3d} hit={st['hit']*100:4.1f}% "
          f"net={st['net']:+.3f} med={st['med']:+.3f} netR={st['netR']:+.3f} "
          f"mddS={st['mdd']*100:4.1f}% | H1 n={h1['n']:3d} net={h1['net']:+.3f} "
          f"| H2 n={h2['n']:3d} net={h2['net']:+.3f}")
    return st


# ---- 1. baseline: repo harness vs my loop ----
print("\n=== baseline ===")
res = run_backtest(bars, slippage_bps=0.0, short_only=True)
harness = res["signals"]
print(f"repo run_backtest: n={res['n_signals']}, "
      f"gross={res['expectancy_pct_post_slippage']:+.3f}, "
      f"net={res['expectancy_pct_post_slippage'] - COST:+.3f}, "
      f"hit={res['hit_rate_tp']*100:.1f}%")
base = my_run()
show("my baseline loop", base)
same = (len(base) == len(harness)
        and all(s.idx == h["idx"] and abs(s.realized_pct - h["realized_pct"]) < 1e-9
                for s, h in zip(base, harness)))
print("my loop == repo harness:", same)

# ---- 2. the claimed winner ----
print("\n=== APR level gate (at-close join) ===")
for amin in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0):
    show(f"apr >= {amin:.0f}%", my_run(apr_min=amin, apr_src=apr_close))

print("\n=== APR level gate (1h-stale join) ===")
for amin in (6.0, 8.0, 10.0):
    show(f"stale apr >= {amin:.0f}%", my_run(apr_min=amin, apr_src=apr_stale))

# ---- 3. winner diagnostics ----
print("\n=== winner (apr>=8, at-close) diagnostics ===")
win = my_run(apr_min=8.0, apr_src=apr_close)
base_idx = {s.idx for s in base}
print("gated trades that are NOT in baseline set:",
      [s.idx for s in win if s.idx not in base_idx])
nets = sorted([(s.realized_pct - COST, s.idx) for s in win])
print("worst 3:", [(round(v, 3), i) for v, i in nets[:3]])
print("best 3:", [(round(v, 3), i) for v, i in nets[-3:]])
ex_best = [v for v, _ in nets[:-1]]
ex_best2 = [v for v, _ in nets[:-2]]
print(f"net excl. best trade: {statistics.mean(ex_best):+.3f}; "
      f"excl. best 2: {statistics.mean(ex_best2):+.3f}")

# quarter splits
qs = [(0, 1250), (1251, 2500), (2501, 3751), (3752, 6000)]
for k, (a, b) in enumerate(qs, 1):
    sub = [s for s in win if a <= s.idx <= b]
    st = stats(sub)
    print(f"Q{k} idx {a}-{b}: n={st['n']:2d} net={st['net']:+.3f}"
          if st["n"] else f"Q{k} idx {a}-{b}: n=0")

# excluded complement
excl = [s for s in base if s.idx not in {w.idx for w in win}]
show("complement (baseline minus winner)", excl)

# distribution of apr at baseline entries
aprs = sorted(apr_close[s.idx] for s in base)
print("apr at baseline entries: min={:.2f} p25={:.2f} med={:.2f} p75={:.2f} max={:.2f}".format(
    aprs[0], aprs[len(aprs)//4], aprs[len(aprs)//2], aprs[3*len(aprs)//4], aprs[-1]))
print("count at exactly baseline 10.95%:",
      sum(1 for a in aprs if abs(a - 10.95) < 0.01))

# post-hoc filter equivalence check (filter baseline list instead of in-loop)
ph = [s for s in base if apr_close[s.idx] is not None and apr_close[s.idx] >= 8.0]
show("post-hoc filtered baseline", ph)
