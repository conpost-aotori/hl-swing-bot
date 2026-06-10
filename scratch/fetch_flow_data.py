"""Backfill flow data to scratch/:
- funding_btc.json : HL hourly funding 2025-11-06 -> now (paginated)
- liq_daily.json   : Coinalyze daily long/short liq, BTCUSDT_PERP.A, full window
- liq_1h.json      : Coinalyze 1h liq (only ~67d available)
- oi_1h.json       : Coinalyze 1h OI (only ~64d available)
"""
import json
import time
import httpx

OUT = "C:/User/projects/hl-swing-bot/scratch"
COINALYZE_KEY = "34d80bab-4b7e-4d8a-8927-2404eb6c0c27"
CBASE = "https://api.coinalyze.net/v1"
NOW = int(time.time())
START = 1762387200  # 2025-11-06 00:00 UTC (7d lookback before parquet start 11-13)

# --- HL funding, paginated ---
fund = []
cursor = START * 1000
while True:
    r = httpx.post("https://api.hyperliquid.xyz/info",
                   json={"type": "fundingHistory", "coin": "BTC",
                         "startTime": cursor, "endTime": NOW * 1000},
                   timeout=30)
    r.raise_for_status()
    page = r.json()
    if not page:
        break
    fund.extend(page)
    last = page[-1]["time"]
    print(f"funding page n={len(page)} last={time.strftime('%Y-%m-%d %H:%M', time.gmtime(last/1000))}")
    if len(page) < 500:
        break
    cursor = last + 1
    time.sleep(0.3)

# dedupe by time
seen = {}
for rec in fund:
    seen[rec["time"]] = rec
fund = [seen[k] for k in sorted(seen)]
json.dump(fund, open(f"{OUT}/funding_btc.json", "w"))
print(f"funding total n={len(fund)} "
      f"first={time.strftime('%Y-%m-%d %H:%M', time.gmtime(fund[0]['time']/1000))} "
      f"last={time.strftime('%Y-%m-%d %H:%M', time.gmtime(fund[-1]['time']/1000))}")


def cget(path, params):
    r = httpx.get(f"{CBASE}{path}", params=params,
                  headers={"api_key": COINALYZE_KEY}, timeout=30)
    r.raise_for_status()
    return r.json()


# --- Coinalyze daily liq (full window incl. 30d pre-window for z baselines) ---
d = cget("/liquidation-history", {"symbols": "BTCUSDT_PERP.A", "interval": "daily",
                                  "from": START - 35 * 86400, "to": NOW,
                                  "convert_to_usd": "true"})
hist = d[0]["history"]
json.dump(hist, open(f"{OUT}/liq_daily.json", "w"))
print(f"liq_daily n={len(hist)}")
time.sleep(2)

# --- Coinalyze 1h liq + OI (as deep as it goes) ---
d = cget("/liquidation-history", {"symbols": "BTCUSDT_PERP.A", "interval": "1hour",
                                  "from": START, "to": NOW, "convert_to_usd": "true"})
hist = d[0]["history"]
json.dump(hist, open(f"{OUT}/liq_1h.json", "w"))
print(f"liq_1h n={len(hist)} first={time.strftime('%Y-%m-%d', time.gmtime(hist[0]['t']))}")
time.sleep(2)

d = cget("/open-interest-history", {"symbols": "BTCUSDT_PERP.A", "interval": "1hour",
                                    "from": START, "to": NOW, "convert_to_usd": "true"})
hist = d[0]["history"]
json.dump(hist, open(f"{OUT}/oi_1h.json", "w"))
print(f"oi_1h n={len(hist)} first={time.strftime('%Y-%m-%d', time.gmtime(hist[0]['t']))}")
