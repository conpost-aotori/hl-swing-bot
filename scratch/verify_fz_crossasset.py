"""Part 2 of adversarial verification:
 1. BTC trade clustering: count effective independent episodes (trades whose
    entries are within 72 bars of the previous gated entry = overlapping holds).
 2. Cross-asset out-of-sample: fetch ETH + SOL hourly funding from HL, run the
    identical short-only baseline and funding_z_168>0.5 gate on their 1h bars.
All NET of 0.19%. Split = idx<=2500 / >2500 (same convention).
"""
import bisect
import json
import statistics
import sys
import time
import urllib.request

import polars as pl

sys.path.insert(0, "C:/User/projects/hl-swing-bot/src")
from hl_swing_bot.backtest import (
    BTSignal, _composite_score, _compute_features_at, _resolve_outcome, HOUR_MS,
)
from hl_swing_bot.features import HourlyBar, MIN_BARS
from hl_swing_bot.signal import (
    COOLDOWN_OPP_DIR_MIN, COOLDOWN_SAME_DIR_MIN,
    FIRE_MOVE_PER_ATR_MIN, FIRE_SCORE_MIN, FIRE_VOL_Z_MIN,
    SIGNAL_TTL_HOURS, STOP_ATR_MULT, TARGET_ATR_MULT,
)

COST = 0.19
SPLIT = 2500
H = 3600 * 1000
SCRATCH = "C:/User/projects/hl-swing-bot/scratch"


def fetch_funding(coin: str, start_ms: int, end_ms: int) -> list[dict]:
    import os
    cache = f"{SCRATCH}/funding_{coin.lower()}_verify.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    out = []
    cur = start_ms
    while cur < end_ms:
        body = json.dumps({"type": "fundingHistory", "coin": coin,
                           "startTime": cur, "endTime": end_ms}).encode()
        req = urllib.request.Request("https://api.hyperliquid.xyz/info", data=body,
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        if not resp:
            break
        out.extend(resp)
        nxt = resp[-1]["time"] + 1
        if nxt <= cur:
            break
        cur = nxt
        if len(resp) < 2:
            break
        time.sleep(0.3)
    json.dump(out, open(cache, "w"))
    return out


def load_bars(path: str) -> list[HourlyBar]:
    df = pl.read_parquet(path)
    return [
        HourlyBar(hour_ms=int(r["open_time_ms"]), open=float(r["open"]),
                  high=float(r["high"]), low=float(r["low"]), close=float(r["close"]),
                  volume=float(r["volume"]), trades=int(r["trades"]))
        for r in df.iter_rows(named=True)
    ]


def make_fund_z(f_times, f_rates):
    def fund_z(at_ms, window=168):
        k = bisect.bisect_right(f_times, at_ms)
        if k < window + 1:
            return None
        latest = f_rates[k - 1]
        hist = f_rates[k - window:k]
        sd = statistics.pstdev(hist)
        if sd <= 1e-12:
            return 0.0
        return (latest - statistics.mean(hist)) / sd
    return fund_z


def run(bars, FEAT, gate=None):
    signals = []
    last_dir, last_idx = None, -10_000
    for i in range(MIN_BARS, len(bars)):
        f = FEAT.get(i)
        if not f:
            continue
        if f["ret_1h"] > 0:
            continue
        elapsed_min = (i - last_idx) * 60
        if last_dir is not None and elapsed_min < COOLDOWN_SAME_DIR_MIN:
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
        last_dir, last_idx = "SHORT", i
    return signals


def seg(ss):
    if not ss:
        return "n=0"
    nets = [s.realized_pct - COST for s in ss]
    tp = sum(1 for s in ss if s.status == "HIT_TP")
    return f"n={len(ss)} hit={tp/len(ss):.0%} net={statistics.mean(nets):+.3f}"


def report(name, sigs):
    h1 = [s for s in sigs if s.idx <= SPLIT]
    h2 = [s for s in sigs if s.idx > SPLIT]
    print(f"{name:34s} FULL[{seg(sigs)}]  H1[{seg(h1)}]  H2[{seg(h2)}]")


def clusters(sigs, gap=SIGNAL_TTL_HOURS):
    """Group trades whose entry is within `gap` bars of the previous entry."""
    if not sigs:
        return []
    out = [[sigs[0]]]
    for s in sigs[1:]:
        if s.idx - out[-1][-1].idx <= gap:
            out[-1].append(s)
        else:
            out.append([s])
    return out


# ================= 1. BTC clustering =================
print("=== BTC cluster analysis of the 20 gated trades ===")
btc_bars = load_bars("C:/User/projects/hl-swing-bot/data/hist_1h.parquet")
fund_btc = json.load(open(f"{SCRATCH}/funding_btc.json"))
fund_btc.sort(key=lambda r: r["time"])
fz_btc = make_fund_z([r["time"] for r in fund_btc],
                     [float(r["fundingRate"]) for r in fund_btc])

FEAT_BTC = {}
for i in range(MIN_BARS, len(btc_bars)):
    f = _compute_features_at(btc_bars, i)
    if f:
        FEAT_BTC[i] = f

gate = lambda ms: (z := fz_btc(ms)) is not None and z > 0.5
win = run(btc_bars, FEAT_BTC, gate)
cl = clusters(win)
print(f"gated trades: {len(win)} -> {len(cl)} non-overlapping episodes (72-bar gap)")
for c in cl:
    nets = [s.realized_pct - COST for s in c]
    half = "H1" if c[0].idx <= SPLIT else "H2"
    print(f"  {half} idx {c[0].idx}-{c[-1].idx}: {len(c)} trades, "
          f"sum={sum(nets):+.2f}, mean={statistics.mean(nets):+.2f}")
ep_means = [statistics.mean([s.realized_pct - COST for s in c]) for c in cl]
print(f"episode-level mean-of-means: {statistics.mean(ep_means):+.3f} "
      f"({sum(1 for m in ep_means if m > 0)}/{len(ep_means)} episodes positive)")
base = run(btc_bars, FEAT_BTC, None)
cl_b = clusters(base)
ep_b = [statistics.mean([s.realized_pct - COST for s in c]) for c in cl_b]
print(f"baseline: {len(base)} trades -> {len(cl_b)} episodes, "
      f"episode mean {statistics.mean(ep_b):+.3f} "
      f"({sum(1 for m in ep_b if m > 0)}/{len(ep_b)} positive)")

# ================= 2. ETH / SOL out-of-sample =================
for coin in ("ETH", "SOL"):
    print(f"\n=== {coin} cross-asset out-of-sample ===")
    bars = load_bars(f"{SCRATCH}/hist_1h_{coin.lower()}.parquet")
    start = bars[0].hour_ms - 200 * H
    end = bars[-1].hour_ms + 2 * H
    fr = fetch_funding(coin, start, end)
    fr.sort(key=lambda r: r["time"])
    print(f"bars: {len(bars)}, funding records: {len(fr)}")
    fz = make_fund_z([r["time"] for r in fr], [float(r["fundingRate"]) for r in fr])
    FEAT = {}
    for i in range(MIN_BARS, len(bars)):
        f = _compute_features_at(bars, i)
        if f:
            FEAT[i] = f
    b = run(bars, FEAT, None)
    report(f"{coin} baseline", b)
    for thr in (0.375, 0.5, 0.625):
        g = lambda ms, t=thr: (z := fz(ms)) is not None and z > t
        report(f"{coin} fund_z168 > {thr}", run(bars, FEAT, g))
    # feature relevance on baseline trades
    pairs = [(fz(s.bar_close_ms), s.realized_pct) for s in b]
    pairs = [(a, y) for a, y in pairs if a is not None]
    if len(pairs) > 5:
        xs, ys = zip(*pairs)
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0.0] * len(v)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                for m in range(i, j + 1):
                    rk[order[m]] = (i + j) / 2 + 1
                i = j + 1
            return rk
        rx, ry = rank(list(xs)), rank(list(ys))
        mx, my = statistics.mean(rx), statistics.mean(ry)
        num = sum((a - mx) * (b2 - my) for a, b2 in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b2 - my) ** 2 for b2 in ry)) ** 0.5
        print(f"  Spearman(fund_z, realized) on {len(pairs)} baseline trades: "
              f"{num / den if den else 0.0:+.3f}")
