"""Flow-alpha gate tests for the SHORT-only baseline.

DECISION RULES (committed BEFORE looking at results):
- Baseline: run_backtest-equivalent loop, short_only, gross realized; NET = gross - 0.19.
- Split-half: entry idx <= 2500 -> H1, else H2.
- A gate variant = ROBUST only if: full NET > baseline full NET, AND H1 NET > baseline
  H1 NET, AND H2 NET > baseline H2 NET, AND both half NETs > 0, AND n >= 15.
- Feature relevance: Spearman rho of feature-at-entry vs realized_pct over the 49
  baseline trades; "has signal" iff |rho| >= 0.2 full-period AND same sign in both halves.
- Gates are applied INSIDE the loop before emit (vetoed candidates set no cooldown),
  mirroring how a flow gate would be wired in signal.py.

Variants (fixed list, no post-hoc additions):
  F-pos   : fund_1h > 0
  F-base  : fund_1h > 1.25e-5            (HL baseline hourly rate)
  F8-pos  : fund_8h > 0
  F8-base : fund_8h > 1.0e-4             (0.01%/8h)
  F-z     : fund_z(168h) > 0.5
  C1-neg  : bias C1 proxy < -10  (fund_8h > 1.25e-4)
  F-neg   : fund_1h < 0                  (inverse sanity check)
  DL-share: prior-UTC-day long-liq share > 0.5
  DL-act  : prior-day long-liq notional > 1.5x trailing-30d median
  DL+F    : DL-share AND F-pos
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
COST = 0.19  # round-trip %, per the lens brief
SPLIT_IDX = 2500

# ---------- load bars ----------
df = pl.read_parquet("C:/User/projects/hl-swing-bot/data/hist_1h.parquet")
bars = [
    HourlyBar(hour_ms=int(r["open_time_ms"]), open=float(r["open"]),
              high=float(r["high"]), low=float(r["low"]), close=float(r["close"]),
              volume=float(r["volume"]), trades=int(r["trades"]))
    for r in df.iter_rows(named=True)
]
print(f"bars: {len(bars)}")

# ---------- load funding ----------
fund = json.load(open(f"{SCRATCH}/funding_btc.json"))
f_times = [rec["time"] for rec in fund]
f_rates = [float(rec["fundingRate"]) for rec in fund]


def fund_feats(at_ms: int) -> dict | None:
    """Funding features using records with time <= at_ms (look-ahead safe)."""
    k = bisect.bisect_right(f_times, at_ms)
    if k < 200:
        return None
    f1 = f_rates[k - 1]
    f8 = sum(f_rates[k - 8:k])
    f24 = sum(f_rates[k - 24:k])
    hist = f_rates[k - 168:k]
    mu = statistics.mean(hist)
    sd = statistics.pstdev(hist)
    fz = (f1 - mu) / sd if sd > 1e-12 else 0.0
    c1 = max(-1.0, min(1.0, -f8 / 0.0005)) * 40
    return {"fund_1h": f1, "fund_8h": f8, "fund_24h": f24, "fund_z": fz, "c1": c1}


# ---------- load daily liq ----------
liq_d = json.load(open(f"{SCRATCH}/liq_daily.json"))
day_liq = {rec["t"]: (float(rec["l"]), float(rec["s"])) for rec in liq_d}
day_keys = sorted(day_liq)


def liq_feats(at_ms: int) -> dict | None:
    """Prior full UTC day's liq features (look-ahead safe)."""
    day0 = (at_ms // 1000) // 86400 * 86400  # current UTC day start
    prior = day0 - 86400
    if prior not in day_liq:
        return None
    l, s = day_liq[prior]
    tot = l + s
    share = l / tot if tot > 0 else 0.5
    j = bisect.bisect_left(day_keys, prior)
    window = [day_liq[day_keys[m]][0] for m in range(max(0, j - 30), j)]
    if len(window) < 20:
        return None
    med = statistics.median(window)
    act = l / med if med > 0 else 0.0
    return {"dl_share": share, "dl_act": act, "dl_long": l}


# ---------- backtest loop with gate ----------
def run(gate=None):
    signals = []
    last_dir, last_idx = None, -10_000
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
        if not (score >= FIRE_SCORE_MIN and f["move_per_atr"] >= FIRE_MOVE_PER_ATR_MIN
                and f["vol_z_168"] >= FIRE_VOL_Z_MIN and f["trend_4h"] <= -1):
            continue
        close_ms = bars[i].hour_ms + HOUR_MS
        if gate is not None and not gate(close_ms):
            continue  # vetoed -> no cooldown set
        atr, entry = f["atr_1h"], f["close"]
        sig = BTSignal(idx=i, bar_close_ms=close_ms, direction=direction,
                       entry=entry, stop=entry + STOP_ATR_MULT * atr,
                       target=entry - TARGET_ATR_MULT * atr, score=score,
                       expires_idx=i + SIGNAL_TTL_HOURS)
        _resolve_outcome(bars, sig, ttl_bars=SIGNAL_TTL_HOURS)
        signals.append(sig)
        last_dir, last_idx = direction, i
    return signals


def report(name, sigs):
    def stats(ss):
        if not ss:
            return "n=0"
        nets = [s.realized_pct - COST for s in ss]
        tp = sum(1 for s in ss if s.status == "HIT_TP")
        return (f"n={len(ss)} hit={tp/len(ss):.0%} net={statistics.mean(nets):+.3f} "
                f"med={statistics.median(nets):+.3f}")
    h1 = [s for s in sigs if s.idx <= SPLIT_IDX]
    h2 = [s for s in sigs if s.idx > SPLIT_IDX]
    print(f"{name:10s} FULL[{stats(sigs)}]  H1[{stats(h1)}]  H2[{stats(h2)}]")
    nets = [s.realized_pct - COST for s in sigs] or [0]
    return {"n": len(sigs), "full": statistics.mean(nets),
            "h1": statistics.mean([s.realized_pct - COST for s in h1]) if h1 else None,
            "h2": statistics.mean([s.realized_pct - COST for s in h2]) if h2 else None}


# ---------- gates ----------
def g_fund(pred):
    def g(ms):
        ff = fund_feats(ms)
        return ff is not None and pred(ff)
    return g


def g_liq(pred):
    def g(ms):
        lf = liq_feats(ms)
        return lf is not None and pred(lf)
    return g


def g_and(a, b):
    return lambda ms: a(ms) and b(ms)


base = run(None)
rb = report("BASELINE", base)

variants = {
    "F-pos":   g_fund(lambda f: f["fund_1h"] > 0),
    "F-base":  g_fund(lambda f: f["fund_1h"] > 1.25e-5),
    "F8-pos":  g_fund(lambda f: f["fund_8h"] > 0),
    "F8-base": g_fund(lambda f: f["fund_8h"] > 1.0e-4),
    "F-z":     g_fund(lambda f: f["fund_z"] > 0.5),
    "C1-neg":  g_fund(lambda f: f["c1"] < -10),
    "F-neg":   g_fund(lambda f: f["fund_1h"] < 0),
    "DL-share": g_liq(lambda l: l["dl_share"] > 0.5),
    "DL-act":  g_liq(lambda l: l["dl_act"] > 1.5),
    "DL+F":    g_and(g_liq(lambda l: l["dl_share"] > 0.5), g_fund(lambda f: f["fund_1h"] > 0)),
}
results = {}
for name, gate in variants.items():
    results[name] = report(name, run(gate))

# ---------- Spearman of features vs outcome on baseline trades ----------
def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for m in range(i, j + 1):
                rk[order[m]] = avg
            i = j + 1
        return rk
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


print("\n--- Spearman feature-at-entry vs realized_pct (baseline trades) ---")
rows = []
for s in base:
    ff = fund_feats(s.bar_close_ms) or {}
    lf = liq_feats(s.bar_close_ms) or {}
    rows.append({**ff, **lf, "y": s.realized_pct, "idx": s.idx})
for feat in ["fund_1h", "fund_8h", "fund_24h", "fund_z", "c1", "dl_share", "dl_act"]:
    sub = [r for r in rows if feat in r]
    xs = [r[feat] for r in sub]
    ys = [r["y"] for r in sub]
    s1 = [(r[feat], r["y"]) for r in sub if r["idx"] <= SPLIT_IDX]
    s2 = [(r[feat], r["y"]) for r in sub if r["idx"] > SPLIT_IDX]
    rho = spearman(xs, ys)
    rho1 = spearman([a for a, _ in s1], [b for _, b in s1]) if len(s1) > 4 else float("nan")
    rho2 = spearman([a for a, _ in s2], [b for _, b in s2]) if len(s2) > 4 else float("nan")
    print(f"{feat:9s} n={len(sub):3d} rho={rho:+.3f}  H1={rho1:+.3f}  H2={rho2:+.3f}")

json.dump({"baseline": rb, "variants": results}, open(f"{SCRATCH}/flow_results.json", "w"), indent=1)
