"""Composite-score signal generation and outcome tracking.

Strategy per SPEC_PHASE1.md:
- Generate a LONG/SHORT signal when the composite score and gates fire on a
  newly closed 1h bar that agrees with the 4h trend, with non-extreme funding.
- Track outcomes: TP / SL / EXPIRED.
- Cooldown: same direction 4h, opposite direction 1h.

This module reads features from features.compute_features(), inserts and
updates rows in storage.signals, and returns dicts that publisher.py renders
into Discord embeds.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .features import compute_features
from .storage import Storage

log = logging.getLogger(__name__)

# --- Composite score weights & gates --------------------------------------
SCORE_WEIGHTS = {
    "move_per_atr": 0.30,
    "robust_z_168": 0.25,
    "vol_z_168":    0.20,
    "ret_4h":       0.15,
    "funding_bonus":0.10,
}

# Hard fire conditions.
FIRE_SCORE_MIN = 3.0
FIRE_MOVE_PER_ATR_MIN = 1.0
FIRE_VOL_Z_MIN = 1.0
FIRE_FUNDING_Z_MAX = 2.5  # |funding_z_24| must be <= this

# Path A (2026-06-10): SHORT-ONLY specialization. The 208-day walk-forward showed
# the LONG branch loses in every regime (uptrend -0.83% net, chop bleeds) while
# short-side downtrends are the only profitable cell (+0.47% net). Crypto cascades
# are also structurally asymmetric (forced long-liquidations cascade harder than
# short squeezes). So we disable LONG emission and forward-test the short edge.
# LONGs are still logged as would-be signals for out-of-sample confirmation and
# for the Path-B (liquidation-bias squeeze) R&D. Set True to re-enable longs.
ENABLE_LONG = False

# Risk/reward.
STOP_ATR_MULT = 1.5
TARGET_ATR_MULT = 2.5  # R:R = 1:1.67
SIGNAL_TTL_HOURS = 72

# Cooldown windows (minutes).
COOLDOWN_SAME_DIR_MIN = 240   # 4h
COOLDOWN_OPP_DIR_MIN = 60     # 1h

# Position sizing (fixed-fractional RISK off the ATR stop). Each stop-out loses a
# constant fraction of equity regardless of the 4.6x atr_pct swing — so per-trade
# equity risk is normalized, which keeps drawdown bounded (208d sim: 9.8% -> 4.7%).
EQUITY_JPY = 100_000          # paper account size
RISK_FRAC = 0.005             # 0.5% equity risked per trade (survive 20 losses = -10%)
# Cap AGGREGATE open risk across overlapping same-coin signals. The 240min cooldown
# + 72h TTL allow 3-4 simultaneous shorts during a crash — this bounds the cluster
# bet so one waterfall can't compound into ruin.
CLUSTER_RISK_CAP = 0.015      # 1.5% total open risk

# Cost model for honest paper-trade accounting. realized_return is stored NET
# of these so the track record reflects what a real account would keep.
# HL taker 0.045%/side x2 = 0.09% round-trip; +0.10% round-trip slippage
# assumption (matches the backtest's 5bps/side). NOTE: funding over the hold is
# NOT yet netted (can be material on 72h holds, and favourable for shorts in a
# dump) — that's a known TODO requiring per-hour funding integration.
TAKER_FEE_RT_PCT = 0.09
SLIPPAGE_RT_PCT = 0.10
COST_RT_PCT = TAKER_FEE_RT_PCT + SLIPPAGE_RT_PCT  # 0.19% round-trip

HOUR_MS = 60 * 60 * 1000


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def composite_score(f: dict) -> float:
    """Weighted sum of feature z-scores and structural terms. Higher = more
    statistically anomalous + structurally aligned. See SPEC_PHASE1.md."""
    atr_pct = max(f["atr_pct"], 1e-9)
    ret_4h_per_atr = abs(f["ret_4h"]) / atr_pct
    funding_bonus = max(0.0, 1.0 - min(abs(f["funding_z_24"]), 3.0) / 3.0)
    # move_per_atr_z is always populated by features.compute_features(). The
    # previous `... or f["move_per_atr"]` fell back to the raw (un-normalized)
    # ratio whenever the z-score was exactly 0.0 (a legitimate neutral reading),
    # silently mixing two scales in this 0.30-weighted lead term. Fixed.
    return (
        SCORE_WEIGHTS["move_per_atr"] * abs(f["move_per_atr_z"])
        + SCORE_WEIGHTS["robust_z_168"] * abs(f["robust_z_168"])
        + SCORE_WEIGHTS["vol_z_168"]    * f["vol_z_168"]
        + SCORE_WEIGHTS["ret_4h"]       * ret_4h_per_atr
        + SCORE_WEIGHTS["funding_bonus"]* funding_bonus
    )


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

def _in_cooldown(storage: Storage, coin: str, *, direction: str, now_ms: int) -> tuple[bool, str | None]:
    """Returns (suppressed, reason). Looks at the most recent signal for this coin."""
    last = storage.latest_signal(coin)
    if not last:
        return False, None
    elapsed_min = (now_ms - last["generated_at_ms"]) / 60_000
    if last["direction"] == direction and elapsed_min < COOLDOWN_SAME_DIR_MIN:
        return True, f"same-direction cooldown ({elapsed_min:.0f}/{COOLDOWN_SAME_DIR_MIN}min)"
    if last["direction"] != direction and elapsed_min < COOLDOWN_OPP_DIR_MIN:
        return True, f"reversal cooldown ({elapsed_min:.0f}/{COOLDOWN_OPP_DIR_MIN}min)"
    return False, None


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def evaluate_and_emit(storage: Storage, coin: str, *, now_ms: int,
                      dry_run: bool = False) -> dict | None:
    """Compute features, check fire conditions, emit a signal if all gates pass.

    Returns the new signal dict (publisher-friendly) or None.
    """
    features = compute_features(storage, coin, now_ms=now_ms)
    if not features:
        return None

    score = composite_score(features)
    direction = "LONG" if features["ret_1h"] > 0 else "SHORT"

    # Gate evaluation (record all failures for logging).
    reasons: list[str] = []
    passes_score = score >= FIRE_SCORE_MIN
    passes_move = features["move_per_atr"] >= FIRE_MOVE_PER_ATR_MIN
    passes_vol = features["vol_z_168"] >= FIRE_VOL_Z_MIN
    trend_aligned = (
        (direction == "LONG" and features["trend_4h"] >= 1)
        or (direction == "SHORT" and features["trend_4h"] <= -1)
    )
    funding_ok = abs(features["funding_z_24"]) <= FIRE_FUNDING_Z_MAX
    long_allowed = ENABLE_LONG or direction == "SHORT"
    in_cd, cd_reason = _in_cooldown(storage, coin, direction=direction, now_ms=now_ms)
    # Aggregate-risk cap: each open signal already risks RISK_FRAC; adding one more
    # must not push total open risk over CLUSTER_RISK_CAP.
    open_risk = len(storage.open_signals(coin)) * RISK_FRAC
    cluster_ok = (open_risk + RISK_FRAC) <= CLUSTER_RISK_CAP + 1e-9

    if not passes_score: reasons.append(f"score {score:.2f} < {FIRE_SCORE_MIN}")
    if not passes_move:  reasons.append(f"move/ATR {features['move_per_atr']:.2f} < {FIRE_MOVE_PER_ATR_MIN}")
    if not passes_vol:   reasons.append(f"vol_z {features['vol_z_168']:.2f} < {FIRE_VOL_Z_MIN}")
    if not trend_aligned: reasons.append(f"trend_4h={features['trend_4h']} vs {direction}")
    if not funding_ok:   reasons.append(f"|funding_z| {abs(features['funding_z_24']):.2f} > {FIRE_FUNDING_Z_MAX}")
    if not long_allowed: reasons.append("LONG disabled (short-only mode)")
    if not cluster_ok:   reasons.append(f"cluster risk cap (open {open_risk*100:.1f}% + {RISK_FRAC*100:.1f}% > {CLUSTER_RISK_CAP*100:.1f}%)")
    if in_cd and cd_reason: reasons.append(cd_reason)

    # would_fire = all signal gates pass; fired = also passes direction + risk policy.
    would_fire = (passes_score and passes_move and passes_vol and trend_aligned
                  and funding_ok and not in_cd)
    fired = would_fire and long_allowed and cluster_ok

    if not fired:
        # Log suppressed-but-otherwise-valid LONGs explicitly so we can confirm
        # out-of-sample that the long side keeps losing (Path-A validation) and
        # accumulate them for Path-B (bias-driven long rebuild).
        if would_fire and not long_allowed:
            log.info("WOULD-BE LONG suppressed (short-only): score=%.2f move/ATR=%.2f",
                     score, features["move_per_atr"])
        else:
            log.info("no fire (%s): score=%.2f reasons=%s", direction, score, "; ".join(reasons))
        return None

    # Stop and target from ATR.
    atr = features["atr_1h"]
    entry = features["close"]
    if direction == "LONG":
        stop = entry - STOP_ATR_MULT * atr
        target = entry + TARGET_ATR_MULT * atr
    else:
        stop = entry + STOP_ATR_MULT * atr
        target = entry - TARGET_ATR_MULT * atr

    # Fixed-fractional-risk size: a stop-out loses exactly RISK_FRAC of equity,
    # regardless of atr_pct. size_coin = risk_jpy / stop_distance_jpy.
    stop_distance = abs(entry - stop)
    risk_jpy = RISK_FRAC * EQUITY_JPY
    size_coin = risk_jpy / stop_distance if stop_distance > 0 else 0.0
    notional_jpy = size_coin * entry

    expires_at_ms = now_ms + SIGNAL_TTL_HOURS * HOUR_MS
    features_json = json.dumps(features, default=str)

    if dry_run:
        signal_id = -1
    else:
        signal_id = storage.insert_signal(
            generated_at_ms=now_ms,
            coin=coin,
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            expires_at_ms=expires_at_ms,
            composite_score=score,
            features_json=features_json,
        )

    return {
        "signal_id": signal_id,
        "generated_at_ms": now_ms,
        "coin": coin,
        "direction": direction,
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "expires_at_ms": expires_at_ms,
        "composite_score": score,
        "features": features,
        "rr_ratio": TARGET_ATR_MULT / STOP_ATR_MULT,
        "size_coin": size_coin,
        "notional_jpy": notional_jpy,
        "risk_jpy": risk_jpy,
    }


# ---------------------------------------------------------------------------
# Outcome tracking
# ---------------------------------------------------------------------------

def update_outcomes(storage: Storage, coin: str, *, mark_price: float,
                    now_ms: int, dry_run: bool = False) -> list[dict]:
    """For each open signal, mark hit/expired and return notifications."""
    notifications: list[dict] = []
    for sig in storage.open_signals(coin):
        new_status: str | None = None
        if sig["direction"] == "LONG":
            if mark_price >= sig["target_price"]:
                new_status = "HIT_TP"
            elif mark_price <= sig["stop_price"]:
                new_status = "HIT_SL"
        else:
            if mark_price <= sig["target_price"]:
                new_status = "HIT_TP"
            elif mark_price >= sig["stop_price"]:
                new_status = "HIT_SL"
        if new_status is None and now_ms >= sig["expires_at_ms"]:
            new_status = "EXPIRED"
        if new_status is None:
            continue

        # Realized return at the trigger price (TP/SL) or mark (EXPIRED).
        close_price = (
            sig["target_price"] if new_status == "HIT_TP"
            else sig["stop_price"] if new_status == "HIT_SL"
            else mark_price
        )
        if sig["direction"] == "LONG":
            gross = (close_price / sig["entry_price"] - 1.0) * 100
        else:
            gross = (sig["entry_price"] / close_price - 1.0) * 100

        # Net of round-trip fees + slippage (NOT funding yet — see COST_RT_PCT).
        # This is what we store, so the paper track record is honest about costs.
        realized = gross - COST_RT_PCT

        if not dry_run:
            storage.close_signal(
                signal_id=sig["signal_id"],
                status=new_status,
                closed_at_ms=now_ms,
                realized_return=realized,
            )
        notifications.append({
            **sig,
            "status": new_status,
            "close_price": close_price,
            "realized_return": realized,
            "gross_return": gross,
        })
    return notifications
