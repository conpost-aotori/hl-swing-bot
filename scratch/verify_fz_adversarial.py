"""ADVERSARIAL verification of the funding_z_168 > 0.5 SHORT gate claim.

Independent re-derivation. Checks:
 (a) reproduction of baseline (49 trades, net +0.093) and winner (n=20, +0.779,
     H1 +0.690 / H2 +0.913)
 (b) split-half (idx<=2500 vs >2500), plus quarter splits
 (c) trade-deletion test: n per half; overlap with baseline trades; permutation
     test (random same-size subsets of baseline trades)
 (d) parameter sensitivity +/-25%: thr {0.375, 0.5, 0.625}, window {126, 168, 210};
     plus implementation-convention nudges (exclude-latest-from-hist z, at-close
     funding alignment, 1h-stale funding)
All results NET of 0.19% round-trip cost.
"""
import bisect
import json
import random
import statistics
import sys

import polars as pl

sys.path.insert(0, "C:/User/projects/hl-swing-bot/src")
from hl_swing_bot.backtest import (
    BTSignal, _composite_score, _compute_features_at, _resolve_outcome, HOUR_MS,
)
from hl_swing_bot.features import HourlyBar, MIN_BARS
from hl_swing_bot.signal import (
    COOLDOWN_OPP_DIR_MIN, COOLDOWN_SAME_DIR_MIN,
    FIRE_MOVE_PER_ATR_MIN, FIRE_SCORE_MIN, FIRE_VOL_Z_MIN,
    SIGNAL_TTL_HOURS, STOP_ATR_MULT, TARGET_ATR_MULT,
)

COST = 0.19
SPLIT = 2500
H = 3600 * 1000

# ---------- bars ----------
df = pl.read_parquet("C:/User/projects/hl-swing-bot/data/hist_1h.parquet")
bars = [
    HourlyBar(hour_ms=int(r["open_time_ms"]), open=float(r["open"]),
              high=float(r["high"]), low=float(r["low"]), close=float(r["close"]),
              volume=float(r["volume"]), trades=int(r["trades"]))
    for r in df.iter_rows(named=True)
]
print(f"bars: {len(bars)}")

# ---------- funding (json store, verified vs parquet & live API) ----------
fund = json.load(open("C:/User/projects/hl-swing-bot/scratch/funding_btc.json"))
fund.sort(key=lambda r: r["time"])
f_times = [r["time"] for r in fund]
f_rates = [float(r["fundingRate"]) for r in fund]
print(f"funding records: {len(f_times)}")


def fund_z(at_ms: int, *, window: int = 168, lag_h: int = 0,
           exclude_latest: bool = False, at_close: bool = False) -> float | None:
    """My implementation. at_ms = bar close. Default convention = records with
    time <= at_ms (record settled AT the close has time ~at_ms+50ms -> excluded,
    i.e. effectively the rate settled 1h before close; matches lens code).
    at_close=True instead includes the record settled at the close boundary
    (time <= at_ms + 5000ms grace). lag_h shifts the cutoff back."""
    cutoff = at_ms - lag_h * H + (5000 if at_close else 0)
    k = bisect.bisect_right(f_times, cutoff)
    need = window + (1 if exclude_latest else 0)
    if k < need + 1:
        return None
    latest = f_rates[k - 1]
    if exclude_latest:
        hist = f_rates[k - 1 - window:k - 1]
    else:
        hist = f_rates[k - window:k]
    mu = statistics.mean(hist)
    sd = statistics.pstdev(hist)
    if sd <= 1e-12:
        return 0.0
    return (latest - mu) / sd


# ---------- feature cache (single O(n^2) pass, repo code) ----------
FEAT: dict[int, dict] = {}
for i in range(MIN_BARS, len(bars)):
    f = _compute_features_at(bars, i)
    if f:
        FEAT[i] = f
print(f"features cached: {len(FEAT)}")


def run(gate=None):
    """Short-only loop mirroring run_backtest; vetoed candidates set no cooldown."""
    signals = []
    last_dir, last_idx = None, -10_000
    for i in range(MIN_BARS, len(bars)):
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
        score = _composite_score(f)
        if not (score >= FIRE_SCORE_MIN and f["move_per_atr"] >= FIRE_MOVE_PER_ATR_MIN
                and f["vol_z_168"] >= FIRE_VOL_Z_MIN and f["trend_4h"] <= -1):
            continue
        close_ms = bars[i].hour_ms + HOUR_MS
        if gate is not None and not gate(close_ms):
            continue  # veto -> no cooldown
        atr, entry = f["atr_1h"], f["close"]
        sig = BTSignal(idx=i, bar_close_ms=close_ms, direction="SHORT",
                       entry=entry, stop=entry + STOP_ATR_MULT * atr,
                       target=entry - TARGET_ATR_MULT * atr, score=score,
                       expires_idx=i + SIGNAL_TTL_HOURS)
        _resolve_outcome(bars, sig, ttl_bars=SIGNAL_TTL_HOURS)
        signals.append(sig)
        last_dir, last_idx = direction, i
    return signals


def seg_stats(ss):
    if not ss:
        return {"n": 0, "net": float("nan"), "hit": float("nan")}
    nets = [s.realized_pct - COST for s in ss]
    tp = sum(1 for s in ss if s.status == "HIT_TP")
    return {"n": len(ss), "net": statistics.mean(nets), "hit": tp / len(ss),
            "med": statistics.median(nets)}


def sized(ss):
    eq, peak, mdd = 1.0, 1.0, 0.0
    for s in ss:
        r = s.realized_pct - COST
        stop_pct = abs(s.stop - s.entry) / s.entry * 100
        eq *= (1 + 0.005 * (r / stop_pct))
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)
    return (eq - 1) * 100, mdd * 100


def report(name, sigs):
    full = seg_stats(sigs)
    h1 = seg_stats([s for s in sigs if s.idx <= SPLIT])
    h2 = seg_stats([s for s in sigs if s.idx > SPLIT])
    tot, mdd = sized(sigs)
    print(f"{name:42s} n={full['n']:3d} hit={full['hit']*100 if full['n'] else 0:4.1f}% "
          f"net={full['net']:+.3f} | H1 n={h1['n']:2d} net={h1['net']:+.3f} | "
          f"H2 n={h2['n']:2d} net={h2['net']:+.3f} | sized {tot:+.2f}%/DD {mdd:.2f}%")
    return full, h1, h2


print("\n================ (a) REPRODUCTION ================")
base = run(None)
b_full, b_h1, b_h2 = report("BASELINE short-only", base)

GATE_THR = 0.5


def g(thr=GATE_THR, **kw):
    def gate(ms):
        z = fund_z(ms, **kw)
        return z is not None and z > thr
    return gate


win = run(g())
w_full, w_h1, w_h2 = report("F-z: fund_z168 > 0.5 (lens convention)", win)

print("\n================ (b) FINER SPLITS ================")
qbounds = [(0, 1250), (1251, 2500), (2501, 3751), (3752, 10**9)]
for q, (lo, hi) in enumerate(qbounds, 1):
    bq = seg_stats([s for s in base if lo <= s.idx <= hi])
    wq = seg_stats([s for s in win if lo <= s.idx <= hi])
    print(f"  Q{q}: baseline n={bq['n']:2d} net={bq['net']:+.3f} | "
          f"gated n={wq['n']:2d} net={wq['net']:+.3f}")

print("\n--- gated trade list (net, status) ---")
for s in win:
    print(f"  idx={s.idx:4d} half={'H1' if s.idx <= SPLIT else 'H2'} {s.status:7s} "
          f"net={s.realized_pct - COST:+.3f}")

print("\n================ (c) TRADE-DELETION / SUBSET TESTS ================")
base_idx = {s.idx for s in base}
win_idx = {s.idx for s in win}
print(f"gated trades also in baseline: {len(win_idx & base_idx)}/{len(win_idx)} "
      f"(new-from-cooldown-chain: {len(win_idx - base_idx)})")

# permutation: random subsets of baseline trades, same size as gated set
base_nets = [s.realized_pct - COST for s in base]
n_keep = len(win)
obs = w_full["net"]
rng = random.Random(20260610)
M = 20000
ge = 0
for _ in range(M):
    sub = rng.sample(base_nets, n_keep)
    if statistics.mean(sub) >= obs:
        ge += 1
print(f"permutation (any {n_keep} of {len(base_nets)} baseline trades): "
      f"P(mean >= {obs:+.3f}) = {ge / M:.4f}")

# half-conditional permutation (preserve per-half counts)
b1 = [s.realized_pct - COST for s in base if s.idx <= SPLIT]
b2 = [s.realized_pct - COST for s in base if s.idx > SPLIT]
n1 = len([s for s in win if s.idx <= SPLIT])
n2 = len([s for s in win if s.idx > SPLIT])
ge2 = 0
for _ in range(M):
    sub = rng.sample(b1, min(n1, len(b1))) + rng.sample(b2, min(n2, len(b2)))
    if statistics.mean(sub) >= obs:
        ge2 += 1
print(f"permutation (preserving {n1}/{n2} half counts): P = {ge2 / M:.4f}")
# joint: both halves simultaneously beat their observed gated means
gej = 0
for _ in range(M):
    s1 = rng.sample(b1, min(n1, len(b1)))
    s2 = rng.sample(b2, min(n2, len(b2)))
    if statistics.mean(s1) >= w_h1["net"] and statistics.mean(s2) >= w_h2["net"]:
        gej += 1
print(f"permutation (both halves >= observed simultaneously): P = {gej / M:.4f}")

print("\n================ (d) PARAMETER SENSITIVITY ================")
print("--- threshold grid (window 168) ---")
for thr in (0.0, 0.25, 0.375, 0.5, 0.625, 0.75):
    report(f"thr {thr:.3f}", run(g(thr=thr)))
print("--- window grid (thr 0.5), +/-25% = 126/210 ---")
for w in (126, 168, 210):
    report(f"window {w}", run(g(window=w)))
print("--- convention nudges (thr 0.5, window 168) ---")
report("exclude-latest-from-hist z", run(g(exclude_latest=True)))
report("at-close funding (incl. record at T)", run(g(at_close=True)))
report("1h-stale funding (lag 1h)", run(g(lag_h=1)))
report("2h-stale funding (lag 2h)", run(g(lag_h=2)))

print("\n--- joint nudge: thr 0.375 x window 126/210 ---")
for w in (126, 210):
    report(f"thr 0.375 window {w}", run(g(thr=0.375, window=w)))
print("--- joint nudge: thr 0.625 x window 126/210 ---")
for w in (126, 210):
    report(f"thr 0.625 window {w}", run(g(thr=0.625, window=w)))

print("\n--- robust_z (repo convention) at 168h, thr 0.25 (claimed alt) ---")
from hl_swing_bot.features import robust_z as repo_robust_z


def g_rz(thr):
    def gate(ms):
        k = bisect.bisect_right(f_times, ms)
        if k < 200:
            return False
        z = repo_robust_z(f_rates[k - 1], f_rates[k - 169:k - 1])
        return z > thr
    return gate


report("robust_z168 > 0.25", run(g_rz(0.25)))
report("robust_z168 > 0.5", run(g_rz(0.5)))

print("\n--- inverse gate sanity (funding_z <= 0.5 i.e. complement) ---")


def g_comp(ms):
    z = fund_z(ms)
    return z is not None and z <= GATE_THR


report("complement: fund_z168 <= 0.5", run(g_comp))

print("\n--- diagnostics: fund_z at the 49 baseline entries ---")
zs = []
for s in base:
    z = fund_z(s.bar_close_ms)
    zs.append((s.idx, z, s.realized_pct - COST))
print("  idx / z / net:")
for idx, z, r in zs:
    print(f"  {idx:4d} z={z:+6.2f} net={r:+.3f}")
