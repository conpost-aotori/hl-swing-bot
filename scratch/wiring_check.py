"""Wiring-oriented checks of the F-z winner: 24h window (matches existing
funding_z_24 plumbing) and robust_z (median/MAD, repo convention) at 168h."""
import bisect
import json
import statistics
import sys

import polars as pl

sys.path.insert(0, "C:/User/projects/hl-swing-bot/src")
from hl_swing_bot.backtest import (  # noqa: E402
    BTSignal, _composite_score, _compute_features_at, _resolve_outcome, HOUR_MS,
)
from hl_swing_bot.features import HourlyBar, MIN_BARS, robust_z  # noqa: E402
from hl_swing_bot.signal import (  # noqa: E402
    COOLDOWN_SAME_DIR_MIN, FIRE_MOVE_PER_ATR_MIN, FIRE_SCORE_MIN, FIRE_VOL_Z_MIN,
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


def z_mean(at_ms, window):
    k = bisect.bisect_right(f_times, at_ms)
    if k < window + 1:
        return None
    hist = f_rates[k - window:k]
    sd = statistics.pstdev(hist)
    return (f_rates[k - 1] - statistics.mean(hist)) / sd if sd > 1e-12 else 0.0


def z_rob(at_ms, window):
    k = bisect.bisect_right(f_times, at_ms)
    if k < window + 1:
        return None
    return robust_z(f_rates[k - 1], f_rates[k - window:k])


def run(gate=None):
    signals = []
    last_idx = -10_000
    for i in range(MIN_BARS, len(bars)):
        f = _compute_features_at(bars, i)
        if not f or f["ret_1h"] > 0:
            continue
        if (i - last_idx) * 60 < COOLDOWN_SAME_DIR_MIN:
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
        last_idx = i
    return signals


def report(name, sigs):
    def m(ss):
        if not ss:
            return "n=0"
        nets = [s.realized_pct - COST for s in ss]
        return f"n={len(ss)} net={statistics.mean(nets):+.3f}"
    h1 = [s for s in sigs if s.idx <= SPLIT_IDX]
    h2 = [s for s in sigs if s.idx > SPLIT_IDX]
    print(f"{name:26s} FULL[{m(sigs)}]  H1[{m(h1)}]  H2[{m(h2)}]")


for w in [24, 48, 168]:
    report(f"mean-z win={w} thr=0.5", run(lambda ms, ww=w: (z_mean(ms, ww) or -9) > 0.5))
for w in [24, 168]:
    report(f"robust-z win={w} thr=0.5", run(lambda ms, ww=w: (z_rob(ms, ww) or -9) > 0.5))
report("robust-z win=168 thr=0.25", run(lambda ms: (z_rob(ms, 168) or -9) > 0.25))
