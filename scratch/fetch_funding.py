"""Fetch full BTC hourly funding history from Hyperliquid covering the 1h bar
range (2025-11-13..now) and save to scratch/funding_1h.parquet via duckdb."""
import json
import time

import duckdb
import httpx

API = "https://api.hyperliquid.xyz/info"
OUT = r"C:\User\projects\hl-swing-bot\scratch\funding_1h.parquet"

# Bar range from hist_1h.parquet
con = duckdb.connect()
row = con.execute(
    "SELECT min(open_time_ms), max(open_time_ms) FROM read_parquet('C:/User/projects/hl-swing-bot/data/hist_1h.parquet')"
).fetchone()
bar_min_ms, bar_max_ms = int(row[0]), int(row[1])
print("bars:", bar_min_ms, "->", bar_max_ms)

# Fetch from 48h before first bar so funding_z_24 has history at bar 0.
start = bar_min_ms - 48 * 3600 * 1000
end = bar_max_ms + 2 * 3600 * 1000

records: dict[int, dict] = {}
cursor = start
client = httpx.Client(timeout=30)
calls = 0
while cursor < end:
    body = {"type": "fundingHistory", "coin": "BTC", "startTime": cursor, "endTime": end}
    r = client.post(API, json=body)
    r.raise_for_status()
    batch = r.json()
    calls += 1
    if not batch:
        break
    for rec in batch:
        t = int(rec["time"])
        records[t] = {
            "time_ms": t,
            "funding_rate": float(rec["fundingRate"]),
            "premium": float(rec.get("premium") or 0.0),
        }
    last_t = max(int(rec["time"]) for rec in batch)
    if last_t <= cursor:
        break
    cursor = last_t + 1
    if len(batch) < 400:  # final partial page
        if last_t >= end - 3600 * 1000:
            break
    time.sleep(0.25)

client.close()
rows = sorted(records.values(), key=lambda x: x["time_ms"])
print(f"fetched {len(rows)} funding records in {calls} calls")
print("first:", rows[0], "last:", rows[-1])

con.execute("CREATE TABLE f (time_ms BIGINT, funding_rate DOUBLE, premium DOUBLE)")
con.executemany("INSERT INTO f VALUES (?,?,?)", [(r["time_ms"], r["funding_rate"], r["premium"]) for r in rows])
con.execute(f"COPY f TO '{OUT}' (FORMAT PARQUET)")
print("saved to", OUT)
