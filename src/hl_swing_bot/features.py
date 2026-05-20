"""Technical features for the 1h swing strategy.

Indicators ported from Btc_alert_bot/features.py but rebuilt around DuckDB
1h-aggregated bars (not 5m). All inputs come from Storage; outputs are a
plain dict suitable for JSON serialization into the signals.features_json
column.
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from typing import Iterable

from .storage import Storage

log = logging.getLogger(__name__)

# Wilder ATR period.
ATR_PERIOD = 14

# Robust-z lookback. 168 bars = 1 week of 1h candles.
HIST_LOOKBACK_BARS = 168

# Minimum bars required before features are considered well-defined.
MIN_BARS = max(ATR_PERIOD + 5, 60)

# Trend filter: simple MA period applied to 4h closes.
TREND_MA_PERIOD = 50

# Funding z-score window (last N hourly fundings).
FUNDING_Z_WINDOW = 24


@dataclass
class HourlyBar:
    hour_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int


# ---------------------------------------------------------------------------
# Stats helpers (ported)
# ---------------------------------------------------------------------------

def robust_z(value: float, history: Iterable[float]) -> float:
    """Median + 1.4826·MAD z-score. Returns 0.0 on degenerate input."""
    hist = [float(x) for x in history if x is not None and not math.isnan(x)]
    if len(hist) < 10:
        return 0.0
    med = statistics.median(hist)
    mad = statistics.median(abs(x - med) for x in hist) or 0.0
    if mad <= 0:
        return 0.0
    return (value - med) / (1.4826 * mad)


def clipped_z(value: float, history: Iterable[float], lo: float, hi: float) -> float:
    return max(lo, min(hi, robust_z(value, history)))


def true_range(bars: list[HourlyBar]) -> list[float]:
    out: list[float] = []
    prev_close: float | None = None
    for b in bars:
        if prev_close is None:
            out.append(b.high - b.low)
        else:
            out.append(max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close)))
        prev_close = b.close
    return out


def wilder_atr(bars: list[HourlyBar], period: int = ATR_PERIOD) -> list[float]:
    """One ATR per bar, 0.0 during warmup."""
    trs = true_range(bars)
    out: list[float] = []
    rma: float | None = None
    for i, tr in enumerate(trs):
        if i < period:
            out.append(0.0)
            if i == period - 1:
                rma = sum(trs[:period]) / period
                out[-1] = rma
        else:
            assert rma is not None
            rma = (rma * (period - 1) + tr) / period
            out.append(rma)
    return out


def pct_return(closes: list[float], lag: int) -> float:
    if len(closes) <= lag:
        return 0.0
    base = closes[-1 - lag]
    if base <= 0:
        return 0.0
    return (closes[-1] / base - 1.0) * 100


# ---------------------------------------------------------------------------
# Bar loaders
# ---------------------------------------------------------------------------

def _truncate_to_hour_ms(now_ms: int) -> int:
    """Floor a millisecond timestamp to the start of its hour."""
    return now_ms - (now_ms % (60 * 60 * 1000))


def load_hourly_bars(storage: Storage, coin: str, *, now_ms: int) -> list[HourlyBar]:
    """All hourly bars STRICTLY BEFORE the current (forming) hour."""
    end_excl = _truncate_to_hour_ms(now_ms)
    rows = storage.fetch_hourly_ohlcv(coin, end_exclusive_ms=end_excl)
    return [
        HourlyBar(
            hour_ms=int(r[0]),
            open=float(r[1]),
            high=float(r[2]),
            low=float(r[3]),
            close=float(r[4]),
            volume=float(r[5]),
            trades=int(r[6]),
        )
        for r in rows
    ]


def aggregate_to_4h(bars: list[HourlyBar]) -> list[HourlyBar]:
    """Group hourly bars into 4h buckets aligned on UTC 00/04/08/12/16/20."""
    out: list[HourlyBar] = []
    bucket: list[HourlyBar] = []
    bucket_start: int | None = None
    BUCKET_MS = 4 * 60 * 60 * 1000
    for b in bars:
        start = b.hour_ms - (b.hour_ms % BUCKET_MS)
        if bucket_start is None or start != bucket_start:
            if bucket:
                out.append(_collapse_bucket(bucket, bucket_start))  # type: ignore[arg-type]
            bucket = [b]
            bucket_start = start
        else:
            bucket.append(b)
    if bucket and bucket_start is not None:
        out.append(_collapse_bucket(bucket, bucket_start))
    return out


def _collapse_bucket(bucket: list[HourlyBar], start_ms: int) -> HourlyBar:
    return HourlyBar(
        hour_ms=start_ms,
        open=bucket[0].open,
        high=max(b.high for b in bucket),
        low=min(b.low for b in bucket),
        close=bucket[-1].close,
        volume=sum(b.volume for b in bucket),
        trades=sum(b.trades for b in bucket),
    )


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def compute_features(storage: Storage, coin: str, *, now_ms: int) -> dict | None:
    """Compute headline features for the latest CLOSED 1h bar.

    Returns None if there isn't enough history. The output dict is what the
    composite score consumes and what gets persisted as features_json.
    """
    bars = load_hourly_bars(storage, coin, now_ms=now_ms)
    if len(bars) < MIN_BARS:
        log.info("Not enough hourly bars: %d / %d", len(bars), MIN_BARS)
        return None

    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]
    atrs = wilder_atr(bars)

    bar_now = bars[-1]
    atr_now = atrs[-1]
    close_now = bar_now.close
    atr_pct = (atr_now / close_now * 100) if close_now > 0 else 0.0

    ret_1h = pct_return(closes, 1)
    ret_4h = pct_return(closes, 4)
    ret_24h = pct_return(closes, 24) if len(closes) >= 25 else 0.0
    move_per_atr = abs(ret_1h) / atr_pct if atr_pct > 0 else 0.0

    # 1-week history slice for robust z-scores (excludes the latest bar itself
    # so the "anomaly" is measured against prior context, not self).
    hist_slice = bars[-(HIST_LOOKBACK_BARS + 1):-1]
    hist_atr_pct = [
        (atrs[i] / bars[i].close * 100)
        for i in range(len(bars) - len(hist_slice) - 1, len(bars) - 1)
        if bars[i].close > 0 and atrs[i] > 0
    ]
    robust_z_close_168 = robust_z(close_now, [b.close for b in hist_slice])
    vol_z_168 = robust_z(bar_now.volume, [b.volume for b in hist_slice])
    move_per_atr_z = robust_z(
        move_per_atr,
        [
            abs((bars[i].close / bars[i - 1].close - 1) * 100) / max(atrs[i] / bars[i].close * 100, 1e-9)
            for i in range(len(bars) - len(hist_slice) - 1, len(bars) - 1)
            if i >= 1 and bars[i - 1].close > 0 and atrs[i] > 0 and bars[i].close > 0
        ],
    )

    # 4h trend filter — sign of (close - SMA50 of 4h closes).
    bars_4h = aggregate_to_4h(bars)
    if len(bars_4h) >= TREND_MA_PERIOD + 1:
        sma50_4h = statistics.mean(b.close for b in bars_4h[-TREND_MA_PERIOD - 1:-1])
        trend_4h = 1 if bars_4h[-1].close > sma50_4h else (-1 if bars_4h[-1].close < sma50_4h else 0)
    else:
        trend_4h = 0

    # Funding context — z-score of latest funding vs last N hourly fundings.
    fundings = storage.recent_funding_rates(coin, n=FUNDING_Z_WINDOW + 1)
    if len(fundings) >= 5:
        funding_now = fundings[-1]
        funding_hist = fundings[:-1] if len(fundings) > 1 else fundings
        funding_z_24 = robust_z(funding_now, funding_hist)
    else:
        funding_now = fundings[-1] if fundings else 0.0
        funding_z_24 = 0.0

    return {
        "bar_close_ms": bar_now.hour_ms + 60 * 60 * 1000,
        "close": close_now,
        "atr_1h": atr_now,
        "atr_pct": atr_pct,
        "ret_1h": ret_1h,
        "ret_4h": ret_4h,
        "ret_24h": ret_24h,
        "move_per_atr": move_per_atr,
        "robust_z_168": robust_z_close_168,
        "vol_z_168": vol_z_168,
        "move_per_atr_z": move_per_atr_z,
        "trend_4h": trend_4h,
        "funding_rate_hourly": funding_now,
        "funding_z_24": funding_z_24,
        "hist_bars": len(bars),
        "hist_atr_median_pct": statistics.median(hist_atr_pct) if hist_atr_pct else 0.0,
    }
