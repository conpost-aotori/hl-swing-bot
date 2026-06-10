"""Fetch ETH and SOL 1h candles from Hyperliquid matching BTC parquet range."""
from __future__ import annotations

import sys
import time

import httpx
import polars as pl

API = "https://api.hyperliquid.xyz/info"
HOUR_MS = 3600_000


def fetch_coin(coin: str, start_ms: int, end_ms: int) -> pl.DataFrame:
    rows: dict[int, dict] = {}
    cursor = start_ms
    with httpx.Client(timeout=30) as client:
        while cursor < end_ms:
            chunk_end = min(cursor + 2000 * HOUR_MS, end_ms)
            body = {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": "1h",
                    "startTime": cursor,
                    "endTime": chunk_end,
                },
            }
            for attempt in range(5):
                try:
                    r = client.post(API, json=body)
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  retry {attempt}: {e}", flush=True)
                    time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"failed to fetch {coin} chunk at {cursor}")
            got = 0
            for c in data or []:
                t = int(c["t"])
                rows[t] = {
                    "open_time_ms": t,
                    "open": float(c["o"]),
                    "high": float(c["h"]),
                    "low": float(c["l"]),
                    "close": float(c["c"]),
                    "volume": float(c["v"]),
                    "trades": int(c.get("n", 0)),
                }
                got += 1
            print(f"  {coin} chunk {cursor} -> {chunk_end}: {got} candles "
                  f"(total {len(rows)})", flush=True)
            if got == 0:
                cursor = chunk_end + HOUR_MS
            else:
                cursor = max(int(max(int(c["t"]) for c in data)) + HOUR_MS,
                             cursor + HOUR_MS)
            time.sleep(0.3)
    df = pl.DataFrame(sorted(rows.values(), key=lambda r: r["open_time_ms"]))
    return df


def main() -> None:
    btc = pl.read_parquet("data/hist_1h.parquet").sort("open_time_ms")
    start_ms = int(btc["open_time_ms"][0])
    end_ms = int(btc["open_time_ms"][-1]) + HOUR_MS
    print(f"BTC range: {start_ms} -> {end_ms}  n={btc.height}")
    for coin in ("ETH", "SOL"):
        df = fetch_coin(coin, start_ms, end_ms)
        # gap check
        ts = df["open_time_ms"].to_list()
        gaps = sum(1 for a, b in zip(ts, ts[1:]) if b - a != HOUR_MS)
        out = f"scratch/hist_1h_{coin.lower()}.parquet"
        df.write_parquet(out)
        print(f"{coin}: {df.height} bars, {gaps} gaps -> {out}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
