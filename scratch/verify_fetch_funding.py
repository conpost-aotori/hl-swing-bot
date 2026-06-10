"""ADVERSARIAL VERIFICATION: independent re-fetch of BTC hourly funding from
Hyperliquid, compared against the lens's scratch/funding_1h.parquet."""
import json
import time

import duckdb
import httpx

API = "https://api.hyperliquid.xyz/info"
OUT = r"C:\User\projects\hl-swing-bot\scratch\verify_funding_btc.json"
H = 3600 * 1000

con = duckdb.connect()
row = con.execute(
    "SELECT min(open_time_ms), max(open_time_ms) FROM read_parquet('C:/User/projects/hl-swing-bot/data/hist_1h.parquet')"
).fetchone()
bar_min, bar_max = int(row[0]), int(row[1])

start = bar_min - 48 * H  # need 24h+ history before bar 0 for z; match lens window
end = bar_max + 2 * H

recs = {}
client = httpx.Client(timeout=30)
cursor = start
calls = 0
while True:
    r = client.post(API, json={"type": "fundingHistory", "coin": "BTC",
                               "startTime": cursor, "endTime": end})
    r.raise_for_status()
    batch = r.json()
    calls += 1
    if not batch:
        break
    new = 0
    for rec in batch:
        t = int(rec["time"])
        if t not in recs:
            new += 1
        recs[t] = float(rec["fundingRate"])
    mx = max(int(rec["time"]) for rec in batch)
    if new == 0 or mx >= end:
        break
    cursor = mx + 1
    time.sleep(0.3)
client.close()

rows = sorted(recs.items())
print(f"independent fetch: {len(rows)} records in {calls} calls; "
      f"range {rows[0][0]} .. {rows[-1][0]}")

with open(OUT, "w") as fh:
    json.dump(rows, fh)

# Compare to lens's parquet (floor both to hour).
lens = {int(t) // H * H: float(fr) for t, fr in con.execute(
    "SELECT time_ms, funding_rate FROM read_parquet('C:/User/projects/hl-swing-bot/scratch/funding_1h.parquet')"
).fetchall()}
mine = {t // H * H: fr for t, fr in rows}

only_lens = set(lens) - set(mine)
only_mine = set(mine) - set(lens)
both = set(lens) & set(mine)
diffs = [(h, lens[h], mine[h]) for h in both if abs(lens[h] - mine[h]) > 1e-12]
print(f"lens-only hours: {len(only_lens)}, mine-only: {len(only_mine)}, "
      f"shared: {len(both)}, value-mismatches: {len(diffs)}")
for d in diffs[:5]:
    print("  mismatch:", d)
for h in sorted(only_mine)[:5]:
    print("  mine-only:", h, mine[h])

# Check hourly continuity over bar range
expected = set(range((bar_min // H) * H, bar_max + H + 1, H))
missing = sorted(h for h in expected if h not in mine)
print(f"missing hours within bar range (mine): {len(missing)}")
print("baseline-rate share:", sum(1 for h in both if abs(mine[h] - 1.25e-05) < 1e-12) / len(both))
