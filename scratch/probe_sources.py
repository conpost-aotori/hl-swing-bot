"""Probe backfillable depth of flow data sources (no analysis yet).

1. Coinalyze liquidation-history / open-interest-history: how far back at 1hour?
2. HL fundingHistory: confirm depth (expected 300+ days hourly).
"""
import json
import time
import httpx

COINALYZE_KEY = "34d80bab-4b7e-4d8a-8927-2404eb6c0c27"  # from sibling Perp-oi-chart/.env
CBASE = "https://api.coinalyze.net/v1"

NOW = int(time.time())
START_2025_11_01 = 1761955200  # 2025-11-01 00:00 UTC


def cget(path, params):
    p = dict(params)
    r = httpx.get(f"{CBASE}{path}", params=p, headers={"api_key": COINALYZE_KEY}, timeout=30)
    print(f"GET {path} {params.get('symbols')} {params.get('interval')} -> {r.status_code}")
    if r.status_code != 200:
        print("  body:", r.text[:300])
        return None
    return r.json()


def summarize(name, data):
    if not data:
        print(f"  {name}: NO DATA")
        return
    for item in data:
        hist = item.get("history", [])
        if not hist:
            print(f"  {name} {item.get('symbol')}: empty history")
            continue
        ts = [h["t"] for h in hist]
        print(f"  {name} {item.get('symbol')}: n={len(hist)} "
              f"first={time.strftime('%Y-%m-%d %H:%M', time.gmtime(min(ts)))} "
              f"last={time.strftime('%Y-%m-%d %H:%M', time.gmtime(max(ts)))}")
        print(f"    sample={hist[len(hist)//2]}")


# --- Coinalyze probes ---
for sym in ["BTCUSDT_PERP.A", "BTC.H"]:
    d = cget("/liquidation-history", {"symbols": sym, "interval": "1hour",
                                      "from": START_2025_11_01, "to": NOW,
                                      "convert_to_usd": "true"})
    summarize("liq-1h", d)
    time.sleep(2)

d = cget("/open-interest-history", {"symbols": "BTCUSDT_PERP.A", "interval": "1hour",
                                    "from": START_2025_11_01, "to": NOW,
                                    "convert_to_usd": "true"})
summarize("oi-1h", d)
time.sleep(2)

d = cget("/liquidation-history", {"symbols": "BTCUSDT_PERP.A", "interval": "daily",
                                  "from": START_2025_11_01 - 200 * 86400, "to": NOW,
                                  "convert_to_usd": "true"})
summarize("liq-daily", d)
time.sleep(2)

# --- HL funding probe: one page from 2025-11-01 ---
r = httpx.post("https://api.hyperliquid.xyz/info",
               json={"type": "fundingHistory", "coin": "BTC",
                     "startTime": START_2025_11_01 * 1000,
                     "endTime": (START_2025_11_01 + 30 * 86400) * 1000},
               timeout=30)
print("HL fundingHistory ->", r.status_code)
fh = r.json()
print(f"  n={len(fh)} first={time.strftime('%Y-%m-%d %H:%M', time.gmtime(fh[0]['time']/1000))} "
      f"last={time.strftime('%Y-%m-%d %H:%M', time.gmtime(fh[-1]['time']/1000))}")
print("  sample:", json.dumps(fh[0]))
