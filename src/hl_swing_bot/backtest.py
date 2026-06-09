"""Backtest harness for the Phase 1 composite-score strategy.

Walks through historical 1h bars in DuckDB, simulates signal generation and
outcome tracking *as if* the strategy had been running. Used to:
- Verify the score=3.0 / move_per_atr=1.0 / vol_z=1.0 thresholds make sense
  against accumulated data.
- Estimate signal frequency, hit-rate, and expectancy before going live.

Caveats:
- Funding-rate filter is disabled by default (we don't have rich enough
  funding history yet). Re-enable once Phase 0 has accumulated 2+ weeks.
- Slippage is modeled as a fixed bps subtracted from realized return.
- Outcomes evaluate intrabar via subsequent bars' high/low — TP/SL are
  assumed to fill at the trigger price (no slippage on the trigger).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from dataclasses import dataclass
from typing import Iterable

from .config import load_settings
from .features import (
    ATR_PERIOD, HIST_LOOKBACK_BARS, HourlyBar, MIN_BARS,
    aggregate_to_4h, pct_return, robust_z, wilder_atr,
)
from .signal import (
    COOLDOWN_OPP_DIR_MIN, COOLDOWN_SAME_DIR_MIN,
    FIRE_MOVE_PER_ATR_MIN, FIRE_SCORE_MIN, FIRE_VOL_Z_MIN,
    SIGNAL_TTL_HOURS, STOP_ATR_MULT, TARGET_ATR_MULT,
)
from .storage import Storage

log = logging.getLogger(__name__)

HOUR_MS = 60 * 60 * 1000


@dataclass
class BTSignal:
    idx: int
    bar_close_ms: int
    direction: str
    entry: float
    stop: float
    target: float
    score: float
    expires_idx: int
    status: str = "OPEN"
    exit_idx: int | None = None
    exit_price: float | None = None
    realized_pct: float | None = None


def _compute_features_at(bars: list[HourlyBar], i: int) -> dict | None:
    """Recompute Phase 1 features as if bar[i] just closed (look-ahead-safe).

    Mirrors features.compute_features() but slices the bar list rather than
    querying live storage.
    """
    sub = bars[: i + 1]
    if len(sub) < MIN_BARS:
        return None

    closes = [b.close for b in sub]
    atrs = wilder_atr(sub)
    bar_now = sub[-1]
    atr_now = atrs[-1]
    atr_pct = (atr_now / bar_now.close * 100) if bar_now.close > 0 else 0.0
    if atr_pct <= 0:
        return None

    ret_1h = pct_return(closes, 1)
    ret_4h = pct_return(closes, 4)
    move_per_atr = abs(ret_1h) / atr_pct

    hist = sub[-(HIST_LOOKBACK_BARS + 1):-1]
    if len(hist) < 30:
        return None
    robust_z_close = robust_z(bar_now.close, [b.close for b in hist])
    vol_z = robust_z(bar_now.volume, [b.volume for b in hist])

    # Move-per-ATR z over history.
    hist_moves: list[float] = []
    for j in range(len(sub) - len(hist) - 1, len(sub) - 1):
        if j < 1 or atrs[j] <= 0 or sub[j].close <= 0 or sub[j - 1].close <= 0:
            continue
        m_atr = (atrs[j] / sub[j].close * 100)
        if m_atr <= 0:
            continue
        m = abs((sub[j].close / sub[j - 1].close - 1) * 100) / m_atr
        hist_moves.append(m)
    move_per_atr_z = robust_z(move_per_atr, hist_moves) if hist_moves else 0.0

    bars_4h = aggregate_to_4h(sub)
    if len(bars_4h) >= 51:
        sma50 = statistics.mean(b.close for b in bars_4h[-51:-1])
        trend_4h = 1 if bars_4h[-1].close > sma50 else (-1 if bars_4h[-1].close < sma50 else 0)
    else:
        trend_4h = 0

    return {
        "close": bar_now.close,
        "atr_1h": atr_now,
        "atr_pct": atr_pct,
        "ret_1h": ret_1h,
        "ret_4h": ret_4h,
        "move_per_atr": move_per_atr,
        "move_per_atr_z": move_per_atr_z,
        "robust_z_168": robust_z_close,
        "vol_z_168": vol_z,
        "trend_4h": trend_4h,
        "funding_z_24": 0.0,  # disabled in backtest v1
    }


def _composite_score(f: dict) -> float:
    # Mirrors signal.composite_score (funding bonus = 1.0 since z=0).
    return (
        0.30 * abs(f["move_per_atr_z"])
        + 0.25 * abs(f["robust_z_168"])
        + 0.20 * f["vol_z_168"]
        + 0.15 * abs(f["ret_4h"]) / max(f["atr_pct"], 1e-9)
        + 0.10 * 1.0
    )


def _resolve_outcome(
    bars: list[HourlyBar], sig: BTSignal, *, ttl_bars: int
) -> None:
    """Walk forward from sig.idx+1 through ttl_bars or end of data; set
    status/exit fields in place."""
    end_idx = min(sig.idx + ttl_bars, len(bars) - 1)
    for j in range(sig.idx + 1, end_idx + 1):
        b = bars[j]
        if sig.direction == "LONG":
            if b.low <= sig.stop:
                sig.status = "HIT_SL"
                sig.exit_idx = j
                sig.exit_price = sig.stop
                sig.realized_pct = (sig.stop / sig.entry - 1) * 100
                return
            if b.high >= sig.target:
                sig.status = "HIT_TP"
                sig.exit_idx = j
                sig.exit_price = sig.target
                sig.realized_pct = (sig.target / sig.entry - 1) * 100
                return
        else:  # SHORT
            if b.high >= sig.stop:
                sig.status = "HIT_SL"
                sig.exit_idx = j
                sig.exit_price = sig.stop
                sig.realized_pct = (sig.entry / sig.stop - 1) * 100
                return
            if b.low <= sig.target:
                sig.status = "HIT_TP"
                sig.exit_idx = j
                sig.exit_price = sig.target
                sig.realized_pct = (sig.entry / sig.target - 1) * 100
                return
    # Expired.
    last_close = bars[end_idx].close
    sig.status = "EXPIRED"
    sig.exit_idx = end_idx
    sig.exit_price = last_close
    if sig.direction == "LONG":
        sig.realized_pct = (last_close / sig.entry - 1) * 100
    else:
        sig.realized_pct = (sig.entry / last_close - 1) * 100


def run_backtest(
    bars: list[HourlyBar],
    *,
    score_min: float = FIRE_SCORE_MIN,
    move_min: float = FIRE_MOVE_PER_ATR_MIN,
    vol_min: float = FIRE_VOL_Z_MIN,
    slippage_bps: float = 5.0,
    ttl_hours: int = SIGNAL_TTL_HOURS,
    short_only: bool = False,
) -> dict:
    """Returns a summary dict with all signals and aggregate metrics.

    short_only=True drops LONG signals (Path-A specialization).
    """
    signals: list[BTSignal] = []
    last_dir: str | None = None
    last_idx: int = -10_000

    for i in range(MIN_BARS, len(bars)):
        # Cooldown: same direction 4h, opposite direction 1h (in bar units).
        f = _compute_features_at(bars, i)
        if not f:
            continue
        direction = "LONG" if f["ret_1h"] > 0 else "SHORT"
        if short_only and direction == "LONG":
            continue
        elapsed_min = (i - last_idx) * 60
        if last_dir is not None:
            if last_dir == direction and elapsed_min < COOLDOWN_SAME_DIR_MIN:
                continue
            if last_dir != direction and elapsed_min < COOLDOWN_OPP_DIR_MIN:
                continue

        score = _composite_score(f)
        passes_score = score >= score_min
        passes_move = f["move_per_atr"] >= move_min
        passes_vol = f["vol_z_168"] >= vol_min
        trend_aligned = (
            (direction == "LONG" and f["trend_4h"] >= 1)
            or (direction == "SHORT" and f["trend_4h"] <= -1)
        )
        if not (passes_score and passes_move and passes_vol and trend_aligned):
            continue

        atr = f["atr_1h"]
        entry = f["close"]
        if direction == "LONG":
            stop = entry - STOP_ATR_MULT * atr
            target = entry + TARGET_ATR_MULT * atr
        else:
            stop = entry + STOP_ATR_MULT * atr
            target = entry - TARGET_ATR_MULT * atr

        sig = BTSignal(
            idx=i, bar_close_ms=bars[i].hour_ms + HOUR_MS,
            direction=direction, entry=entry, stop=stop, target=target,
            score=score, expires_idx=i + ttl_hours,
        )
        _resolve_outcome(bars, sig, ttl_bars=ttl_hours)
        signals.append(sig)
        last_dir = direction
        last_idx = i

    # Aggregate.
    n = len(signals)
    if n == 0:
        return {"n_signals": 0, "bars_evaluated": len(bars) - MIN_BARS}

    slip = slippage_bps / 100  # bps → percent
    realized = [s.realized_pct - slip * 2 for s in signals if s.realized_pct is not None]
    hits_tp = sum(1 for s in signals if s.status == "HIT_TP")
    hits_sl = sum(1 for s in signals if s.status == "HIT_SL")
    expired = sum(1 for s in signals if s.status == "EXPIRED")
    longs = sum(1 for s in signals if s.direction == "LONG")
    shorts = n - longs

    return {
        "n_signals": n,
        "bars_evaluated": len(bars) - MIN_BARS,
        "signals_per_week": n / max((len(bars) - MIN_BARS) / 168, 1e-9),
        "long_count": longs,
        "short_count": shorts,
        "tp_count": hits_tp,
        "sl_count": hits_sl,
        "expired_count": expired,
        "hit_rate_tp": hits_tp / n,
        "expectancy_pct_post_slippage": statistics.mean(realized),
        "expectancy_median_pct": statistics.median(realized),
        "best_pct": max(realized),
        "worst_pct": min(realized),
        "signals": [
            {
                "idx": s.idx, "ms": s.bar_close_ms, "direction": s.direction,
                "entry": s.entry, "stop": s.stop, "exit": s.exit_price,
                "exit_idx": s.exit_idx, "status": s.status,
                "realized_pct": s.realized_pct, "score": s.score,
            }
            for s in signals
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="hl-swing-bot backtest harness")
    parser.add_argument("--score-min", type=float, default=FIRE_SCORE_MIN)
    parser.add_argument("--move-min", type=float, default=FIRE_MOVE_PER_ATR_MIN)
    parser.add_argument("--vol-min", type=float, default=FIRE_VOL_Z_MIN)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    settings = load_settings()
    with Storage(settings.duckdb_path) as store:
        rows = store.fetch_hourly_ohlcv(
            settings.hl_coin, end_exclusive_ms=2**62
        )
    bars = [
        HourlyBar(
            hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
            low=float(r[3]), close=float(r[4]), volume=float(r[5]),
            trades=int(r[6]),
        )
        for r in rows
    ]
    print(f"loaded {len(bars)} hourly bars")

    result = run_backtest(
        bars,
        score_min=args.score_min,
        move_min=args.move_min,
        vol_min=args.vol_min,
        slippage_bps=args.slippage_bps,
    )
    print(json.dumps(
        {k: v for k, v in result.items() if k != "signals"},
        indent=2, default=float,
    ))
    if args.verbose and result.get("n_signals"):
        for s in result["signals"]:
            print(s)


if __name__ == "__main__":
    main()
