"""DuckDB storage for candles + funding/OI snapshots.

Schema is intentionally tall (one row per bar per coin) so we can add coins
without altering schema. Primary keys prevent duplicate inserts when polls
overlap.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import duckdb

from .hyperliquid_client import Candle, HyperliquidPerp

log = logging.getLogger(__name__)


SCHEMA_CANDLES = """
CREATE TABLE IF NOT EXISTS candles (
    coin           VARCHAR NOT NULL,
    interval       VARCHAR NOT NULL,
    open_time_ms   BIGINT  NOT NULL,
    close_time_ms  BIGINT  NOT NULL,
    open           DOUBLE  NOT NULL,
    high           DOUBLE  NOT NULL,
    low            DOUBLE  NOT NULL,
    close          DOUBLE  NOT NULL,
    volume         DOUBLE  NOT NULL,
    trades         INTEGER NOT NULL,
    PRIMARY KEY (coin, interval, open_time_ms)
);
"""

SCHEMA_FUNDING = """
CREATE TABLE IF NOT EXISTS perp_snapshots (
    coin                    VARCHAR NOT NULL,
    snapshot_time_ms        BIGINT  NOT NULL,
    mark_price_usd          DOUBLE  NOT NULL,
    prev_day_price_usd      DOUBLE  NOT NULL,
    day_volume_usd          DOUBLE  NOT NULL,
    open_interest_coin      DOUBLE  NOT NULL,
    open_interest_usd       DOUBLE  NOT NULL,
    funding_rate_hourly     DOUBLE  NOT NULL,
    PRIMARY KEY (coin, snapshot_time_ms)
);
"""

SCHEMA_SIGNALS = """
CREATE SEQUENCE IF NOT EXISTS signal_id_seq;
CREATE TABLE IF NOT EXISTS signals (
    signal_id          BIGINT PRIMARY KEY DEFAULT nextval('signal_id_seq'),
    generated_at_ms    BIGINT  NOT NULL,
    coin               VARCHAR NOT NULL,
    direction          VARCHAR NOT NULL,   -- LONG / SHORT
    entry_price        DOUBLE  NOT NULL,
    stop_price         DOUBLE  NOT NULL,
    target_price       DOUBLE  NOT NULL,
    expires_at_ms      BIGINT  NOT NULL,
    composite_score    DOUBLE  NOT NULL,
    features_json      VARCHAR NOT NULL,
    status             VARCHAR NOT NULL,   -- NEW / HIT_TP / HIT_SL / EXPIRED / CANCELLED
    closed_at_ms       BIGINT,
    realized_return    DOUBLE
);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status, coin);
"""

# Phase 1.5: tall feature store. One row per (coin, time, feature_name) so
# sibling-project features at any cadence (1h / 6h / daily) coexist without
# schema changes. See SPEC_PHASE1_5.md.
SCHEMA_FEATURES = """
CREATE TABLE IF NOT EXISTS features (
    coin              VARCHAR NOT NULL,
    feature_time_ms   BIGINT  NOT NULL,
    feature_name      VARCHAR NOT NULL,
    feature_value     DOUBLE  NOT NULL,
    source            VARCHAR NOT NULL,
    ingested_at_ms    BIGINT  NOT NULL,
    PRIMARY KEY (coin, feature_time_ms, feature_name)
);
CREATE INDEX IF NOT EXISTS idx_features_name_time ON features (feature_name, feature_time_ms);
"""

# Aggregates 1m candles into 1h bars. Only emits CLOSED hours (we filter the
# in-progress bar out at the call site by capping end_hour at now-1h).
SQL_HOURLY_OHLCV = """
SELECT
    epoch_ms(date_trunc('hour', to_timestamp(open_time_ms / 1000))) AS hour_ms,
    arg_min(open, open_time_ms)  AS open,
    MAX(high)                    AS high,
    MIN(low)                     AS low,
    arg_max(close, open_time_ms) AS close,
    SUM(volume)                  AS volume,
    SUM(trades)                  AS trades
FROM candles
WHERE coin = ? AND interval = '1m' AND open_time_ms < ?
GROUP BY hour_ms
ORDER BY hour_ms
"""


class Storage:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._conn = duckdb.connect(str(db_path))
        self._conn.execute(SCHEMA_CANDLES)
        self._conn.execute(SCHEMA_FUNDING)
        self._conn.execute(SCHEMA_SIGNALS)
        self._conn.execute(SCHEMA_FEATURES)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def upsert_candles(self, candles: Iterable[Candle]) -> int:
        rows = [
            (
                c.coin,
                c.interval,
                c.open_time_ms,
                c.close_time_ms,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.trades,
            )
            for c in candles
        ]
        if not rows:
            return 0
        # INSERT OR IGNORE preserves the earliest row; replace it if the close
        # has changed (the current/forming bar updates on every poll).
        self._conn.executemany(
            """
            INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (coin, interval, open_time_ms) DO UPDATE SET
                close_time_ms = excluded.close_time_ms,
                open  = excluded.open,
                high  = excluded.high,
                low   = excluded.low,
                close = excluded.close,
                volume= excluded.volume,
                trades= excluded.trades
            """,
            rows,
        )
        return len(rows)

    def insert_perp_snapshot(self, perp: HyperliquidPerp, *, snapshot_time_ms: int) -> None:
        self._conn.execute(
            """
            INSERT INTO perp_snapshots VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT (coin, snapshot_time_ms) DO NOTHING
            """,
            (
                perp.coin,
                snapshot_time_ms,
                perp.mark_price_usd,
                perp.prev_day_price_usd,
                perp.day_volume_usd,
                perp.open_interest_coin,
                perp.open_interest_usd,
                perp.funding_rate_hourly,
            ),
        )

    def candle_count(self, coin: str, interval: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM candles WHERE coin = ? AND interval = ?",
            (coin, interval),
        ).fetchone()
        return int(row[0]) if row else 0

    def latest_candle_time_ms(self, coin: str, interval: str) -> int | None:
        row = self._conn.execute(
            "SELECT MAX(open_time_ms) FROM candles WHERE coin = ? AND interval = ?",
            (coin, interval),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    # -- Hourly aggregation --------------------------------------------------

    def fetch_hourly_ohlcv(
        self, coin: str, *, end_exclusive_ms: int
    ) -> list[tuple]:
        """Return rows (hour_ms, open, high, low, close, volume, trades) for
        hours fully contained in [-infinity, end_exclusive_ms). Pass
        ``end_exclusive_ms = now_ms_truncated_to_hour`` to exclude the bar
        currently forming.
        """
        return self._conn.execute(
            SQL_HOURLY_OHLCV, (coin, end_exclusive_ms)
        ).fetchall()

    def latest_perp_snapshot(self, coin: str) -> tuple | None:
        """(snapshot_time_ms, mark, funding_hourly) for the most recent snapshot."""
        row = self._conn.execute(
            """
            SELECT snapshot_time_ms, mark_price_usd, funding_rate_hourly
            FROM perp_snapshots
            WHERE coin = ?
            ORDER BY snapshot_time_ms DESC
            LIMIT 1
            """,
            (coin,),
        ).fetchone()
        return tuple(row) if row else None

    def recent_funding_rates(self, coin: str, *, n: int = 24) -> list[float]:
        """Last ``n`` funding rates, oldest first."""
        rows = self._conn.execute(
            """
            SELECT funding_rate_hourly FROM (
                SELECT funding_rate_hourly, snapshot_time_ms
                FROM perp_snapshots
                WHERE coin = ?
                ORDER BY snapshot_time_ms DESC
                LIMIT ?
            ) ORDER BY snapshot_time_ms ASC
            """,
            (coin, n),
        ).fetchall()
        return [float(r[0]) for r in rows]

    # -- Signal CRUD ---------------------------------------------------------

    def insert_signal(self, *, generated_at_ms: int, coin: str, direction: str,
                      entry_price: float, stop_price: float, target_price: float,
                      expires_at_ms: int, composite_score: float,
                      features_json: str) -> int:
        row = self._conn.execute(
            """
            INSERT INTO signals (
                generated_at_ms, coin, direction, entry_price, stop_price,
                target_price, expires_at_ms, composite_score, features_json, status
            ) VALUES (?,?,?,?,?,?,?,?,?, 'NEW')
            RETURNING signal_id
            """,
            (generated_at_ms, coin, direction, entry_price, stop_price,
             target_price, expires_at_ms, composite_score, features_json),
        ).fetchone()
        return int(row[0])

    def open_signals(self, coin: str) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT signal_id, generated_at_ms, direction, entry_price,
                   stop_price, target_price, expires_at_ms, composite_score
            FROM signals
            WHERE coin = ? AND status = 'NEW'
            ORDER BY signal_id
            """,
            (coin,),
        ).fetchall()
        keys = ("signal_id", "generated_at_ms", "direction", "entry_price",
                "stop_price", "target_price", "expires_at_ms", "composite_score")
        return [dict(zip(keys, r)) for r in rows]

    def close_signal(self, *, signal_id: int, status: str,
                     closed_at_ms: int, realized_return: float) -> None:
        self._conn.execute(
            """
            UPDATE signals
            SET status = ?, closed_at_ms = ?, realized_return = ?
            WHERE signal_id = ?
            """,
            (status, closed_at_ms, realized_return, signal_id),
        )

    def latest_signal(self, coin: str) -> dict | None:
        row = self._conn.execute(
            """
            SELECT signal_id, generated_at_ms, direction, status
            FROM signals
            WHERE coin = ?
            ORDER BY signal_id DESC
            LIMIT 1
            """,
            (coin,),
        ).fetchone()
        if not row:
            return None
        return {
            "signal_id": int(row[0]),
            "generated_at_ms": int(row[1]),
            "direction": row[2],
            "status": row[3],
        }

    # -- Feature store (Phase 1.5) -------------------------------------------

    def upsert_features(self, rows: list[tuple], *, source: str,
                        ingested_at_ms: int) -> int:
        """Insert feature rows. Each row is (coin, feature_time_ms,
        feature_name, feature_value). Idempotent on the composite PK."""
        if not rows:
            return 0
        payload = [
            (coin, ftime, fname, fval, source, ingested_at_ms)
            for (coin, ftime, fname, fval) in rows
        ]
        self._conn.executemany(
            """
            INSERT INTO features VALUES (?,?,?,?,?,?)
            ON CONFLICT (coin, feature_time_ms, feature_name) DO UPDATE SET
                feature_value = excluded.feature_value,
                source        = excluded.source,
                ingested_at_ms= excluded.ingested_at_ms
            """,
            payload,
        )
        return len(payload)

    def latest_feature_time_ms(self, source: str) -> int | None:
        row = self._conn.execute(
            "SELECT MAX(feature_time_ms) FROM features WHERE source = ?",
            (source,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def feature_series(self, coin: str, feature_name: str) -> list[tuple]:
        """All (feature_time_ms, feature_value) for a feature, oldest first."""
        return self._conn.execute(
            """
            SELECT feature_time_ms, feature_value
            FROM features
            WHERE coin = ? AND feature_name = ?
            ORDER BY feature_time_ms
            """,
            (coin, feature_name),
        ).fetchall()

    def latest_feature_value(self, coin: str, feature_name: str,
                             *, as_of_ms: int | None = None) -> float | None:
        """Most recent feature value at or before ``as_of_ms`` (forward-fill).
        Without as_of_ms, returns the latest known value."""
        if as_of_ms is None:
            row = self._conn.execute(
                """
                SELECT feature_value FROM features
                WHERE coin = ? AND feature_name = ?
                ORDER BY feature_time_ms DESC LIMIT 1
                """,
                (coin, feature_name),
            ).fetchone()
        else:
            row = self._conn.execute(
                """
                SELECT feature_value FROM features
                WHERE coin = ? AND feature_name = ? AND feature_time_ms <= ?
                ORDER BY feature_time_ms DESC LIMIT 1
                """,
                (coin, feature_name, as_of_ms),
            ).fetchone()
        return float(row[0]) if row else None

    def feature_count(self, source: str | None = None) -> int:
        if source is None:
            row = self._conn.execute("SELECT COUNT(*) FROM features").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM features WHERE source = ?", (source,)
            ).fetchone()
        return int(row[0]) if row else 0
