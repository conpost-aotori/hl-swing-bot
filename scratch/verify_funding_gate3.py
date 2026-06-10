"""Final probe: per-half nets excluding the single best squeeze episode in
each half, and exact winner-set composition (in-loop, apr>=8 at-close)."""
import json
import statistics
import sys

import duckdb

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")
from hl_swing_bot.backtest import (
    BTSignal, HOUR_MS, _compute_features_at, _composite_score, _resolve_outcome,
)
from hl_swing_bot.features import HourlyBar, MIN_BARS
from hl_swing_bot.signal import (
    COOLDOWN_OPP_DIR_MIN, COOLDOWN_SAME_DIR_MIN, FIRE_MOVE_PER_ATR_MIN,
    FIRE_SCORE_MIN, FIRE_VOL_Z_MIN, SIGNAL_TTL_HOURS, STOP_ATR_MULT,
    TARGET_ATR_MULT,
)

COST = 0.19
H = 3600 * 1000
con = duckdb.connect()
rows = con.execute(
    "SELECT open_time_ms, open, high, low, close, volume, trades "
    "FROM read_parquet('C:/User/projects/hl-swing-bot/data/hist_1h.parquet') "
    "ORDER BY open_time_ms").fetchall()
bars = [HourlyBar(hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
                  low=float(r[3]), close=float(r[4]), volume=float(r[5]),
                  trades=int(r[6])) for r in rows]
n = len(bars)
with open(r"C:\User\projects\hl-swing-bot\scratch\verify_funding_btc.json") as fh:
    rate = {int(t) // H * H: float(v) for t, v in json.load(fh)}
apr = [rate[b.hour_ms + H] * 24 * 365 * 100 for b in bars]

FEATS = {}
for i in range(MIN_BARS, n):
    f = _compute_features_at(bars, i)
    if f:
        FEATS[i] = f

def run(apr_min=None):
    signals, last_dir, last_idx = [], None, -10_000
    for i in range(MIN_BARS, n):
        f = FEATS.get(i)
        if not f:
            continue
        if f["ret_1h"] > 0:
            continue
        em = (i - last_idx) * 60
        if last_dir == "SHORT" and em < COOLDOWN_SAME_DIR_MIN:
            continue
        score = _composite_score(f)
        if not (score >= FIRE_SCORE_MIN and f["move_per_atr"] >= FIRE_MOVE_PER_ATR_MIN
                and f["vol_z_168"] >= FIRE_VOL_Z_MIN and f["trend_4h"] <= -1):
            continue
        if apr_min is not None and apr[i] < apr_min:
            continue
        atr, entry = f["atr_1h"], f["close"]
        sig = BTSignal(idx=i, bar_close_ms=bars[i].hour_ms + HOUR_MS,
                       direction="SHORT", entry=entry,
                       stop=entry + STOP_ATR_MULT * atr,
                       target=entry - TARGET_ATR_MULT * atr, score=score,
                       expires_idx=i + SIGNAL_TTL_HOURS)
        _resolve_outcome(bars, sig, ttl_bars=SIGNAL_TTL_HOURS)
        signals.append(sig)
        last_dir, last_idx = "SHORT", i
    return signals

win = run(apr_min=8.0)
print("in-loop winner set (n={}):".format(len(win)))
for s in win:
    print(f"  idx={s.idx:4d} {'H1' if s.idx<=2500 else 'H2'} {s.status:7s} "
          f"net={s.realized_pct-COST:+.3f} apr={apr[s.idx]:.2f}")

h1 = [s for s in win if s.idx <= 2500]
h2 = [s for s in win if s.idx > 2500]
def m(x): return statistics.mean([s.realized_pct - COST for s in x]) if x else float('nan')
print(f"\nH1 n={len(h1)} net={m(h1):+.3f} | H2 n={len(h2)} net={m(h2):+.3f}")

# exclude the single best contiguous episode (trades within 72 bars of each
# other containing the max trade) from each half
def best_episode(sigs):
    best = max(sigs, key=lambda s: s.realized_pct)
    ep = [s for s in sigs if abs(s.idx - best.idx) <= 72]
    return ep
ep1, ep2 = best_episode(h1), best_episode(h2)
print(f"H1 best episode: idx {[s.idx for s in ep1]} sum={sum(s.realized_pct-COST for s in ep1):+.3f}")
print(f"H2 best episode: idx {[s.idx for s in ep2]} sum={sum(s.realized_pct-COST for s in ep2):+.3f}")
r1 = [s for s in h1 if s not in ep1]
r2 = [s for s in h2 if s not in ep2]
print(f"H1 excl best episode: n={len(r1)} net={m(r1):+.3f}")
print(f"H2 excl best episode: n={len(r2)} net={m(r2):+.3f}")

# same treatment for baseline halves (fair comparison)
base = run()
b1 = [s for s in base if s.idx <= 2500]
b2 = [s for s in base if s.idx > 2500]
be1, be2 = best_episode(b1), best_episode(b2)
rb1 = [s for s in b1 if s not in be1]
rb2 = [s for s in b2 if s not in be2]
print(f"\nbaseline H1 excl best episode: n={len(rb1)} net={m(rb1):+.3f} "
      f"(full {m(b1):+.3f})")
print(f"baseline H2 excl best episode: n={len(rb2)} net={m(rb2):+.3f} "
      f"(full {m(b2):+.3f})")

# TP-rate comparison gated vs complement (Fisher-style counts)
comp = [s for s in base if s.idx not in {w.idx for w in win}]
tp_w = sum(1 for s in win if s.status == "HIT_TP")
tp_c = sum(1 for s in comp if s.status == "HIT_TP")
print(f"\nTP gated {tp_w}/{len(win)} vs complement {tp_c}/{len(comp)}")
