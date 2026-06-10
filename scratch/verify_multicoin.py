"""ADVERSARIAL VERIFICATION of the multi-coin (BTC+ETH) expansion claim.

Independent re-derivation:
1. Reproduce BTC baseline + ETH/SOL standalone via repo run_backtest (frozen params).
2. Reimplement the signal loop on top of a per-bar feature cache; verify my loop
   produces IDENTICAL signals to run_backtest at frozen params (cross-check).
3. My own portfolio simulator (global cap), split-half with n per half.
4. Parameter sensitivity: one-at-a-time +/-25% nudges of score_min/move_min/vol_min.
"""
from __future__ import annotations

import datetime
import json
import os
import statistics
import sys

import polars as pl

from hl_swing_bot.backtest import HourlyBar, run_backtest, _compute_features_at, _composite_score
from hl_swing_bot.signal import (
    COOLDOWN_OPP_DIR_MIN, COOLDOWN_SAME_DIR_MIN,
    STOP_ATR_MULT, TARGET_ATR_MULT, SIGNAL_TTL_HOURS,
)

HOUR_MS = 3600_000
COST = 0.19
RISK = 0.5
CAP_SLOTS = 3
MIN_BARS = 60

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path: str) -> list[HourlyBar]:
    df = pl.read_parquet(os.path.join(ROOT, path)).sort("open_time_ms")
    return [HourlyBar(int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                      float(r[4]), float(r[5]), int(r[6]))
            for r in df.iter_rows()]


def build_feature_cache(coin: str, bars: list[HourlyBar]) -> list[dict | None]:
    cache_path = os.path.join(ROOT, "scratch", f"verify_feat_{coin}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            return json.load(fh)
    feats: list[dict | None] = []
    for i in range(len(bars)):
        if i < MIN_BARS:
            feats.append(None)
            continue
        f = _compute_features_at(bars, i)
        feats.append(f)
    with open(cache_path, "w") as fh:
        json.dump(feats, fh)
    return feats


def resolve_short(bars: list[HourlyBar], i: int, entry: float, stop: float,
                  target: float, ttl: int) -> tuple[str, int, float]:
    end_idx = min(i + ttl, len(bars) - 1)
    for j in range(i + 1, end_idx + 1):
        b = bars[j]
        if b.high >= stop:
            return "HIT_SL", j, (entry / stop - 1) * 100
        if b.low <= target:
            return "HIT_TP", j, (entry / target - 1) * 100
    return "EXPIRED", end_idx, (entry / bars[end_idx].close - 1) * 100


def my_backtest(bars: list[HourlyBar], feats: list[dict | None], coin: str,
                score_min: float, move_min: float, vol_min: float) -> list[dict]:
    """My own loop, short-only, mirrors the live pipeline rules."""
    out: list[dict] = []
    last_dir: str | None = None
    last_idx = -10_000
    for i in range(MIN_BARS, len(bars)):
        f = feats[i]
        if not f:
            continue
        direction = "LONG" if f["ret_1h"] > 0 else "SHORT"
        if direction == "LONG":
            continue
        elapsed_min = (i - last_idx) * 60
        if last_dir is not None:
            if last_dir == direction and elapsed_min < COOLDOWN_SAME_DIR_MIN:
                continue
            if last_dir != direction and elapsed_min < COOLDOWN_OPP_DIR_MIN:
                continue
        score = _composite_score(f)
        trend_ok = f["trend_4h"] <= -1
        if not (score >= score_min and f["move_per_atr"] >= move_min
                and f["vol_z_168"] >= vol_min and trend_ok):
            continue
        atr = f["atr_1h"]
        entry = f["close"]
        stop = entry + STOP_ATR_MULT * atr
        target = entry - TARGET_ATR_MULT * atr
        status, exit_idx, realized = resolve_short(bars, i, entry, stop, target, SIGNAL_TTL_HOURS)
        stop_dist = abs(stop / entry - 1) * 100
        net = realized - COST
        out.append({
            "coin": coin, "idx": i, "entry_ms": bars[i].hour_ms + HOUR_MS,
            "exit_ms": bars[exit_idx].hour_ms + HOUR_MS, "status": status,
            "gross": realized, "net": net, "net_R": net / stop_dist,
            "entry": entry, "stop": stop,
        })
        last_dir = direction
        last_idx = i
    return out


def stats(sigs: list[dict], boundary_ms: int) -> dict:
    n = len(sigs)
    if n == 0:
        return {"n": 0}
    h1 = [s for s in sigs if s["entry_ms"] <= boundary_ms]
    h2 = [s for s in sigs if s["entry_ms"] > boundary_ms]
    d = {
        "n": n,
        "hit": sum(1 for s in sigs if s["status"] == "HIT_TP") / n,
        "gross": statistics.mean(s["gross"] for s in sigs),
        "net": statistics.mean(s["net"] for s in sigs),
        "net_R": statistics.mean(s["net_R"] for s in sigs),
        "n1": len(h1), "n2": len(h2),
        "net1": statistics.mean(s["net"] for s in h1) if h1 else float("nan"),
        "net2": statistics.mean(s["net"] for s in h2) if h2 else float("nan"),
    }
    return d


def fmt(label: str, d: dict) -> str:
    if d["n"] == 0:
        return f"{label}: n=0"
    return (f"{label}: n={d['n']} hit={d['hit']*100:.0f}% gross={d['gross']:+.3f}% "
            f"net={d['net']:+.3f}% netR={d['net_R']:+.3f} | "
            f"H1 n={d['n1']} net={d['net1']:+.3f}% | H2 n={d['n2']} net={d['net2']:+.3f}%")


def portfolio(sig_lists: list[list[dict]], cap_slots: int) -> tuple[list[dict], list[dict]]:
    merged = sorted((s for sl in sig_lists for s in sl),
                    key=lambda s: (s["entry_ms"], s["coin"]))
    accepted: list[dict] = []
    skipped: list[dict] = []
    open_tr: list[dict] = []
    for s in merged:
        open_tr = [t for t in open_tr if t["exit_ms"] > s["entry_ms"]]
        if len(open_tr) >= cap_slots:
            skipped.append(s)
            continue
        accepted.append(s)
        open_tr.append(s)
    return accepted, skipped


def equity(trades: list[dict]) -> tuple[float, float]:
    eq, peak, mdd = 1.0, 1.0, 0.0
    for t in sorted(trades, key=lambda x: x["exit_ms"]):
        eq *= 1 + (RISK / 100) * t["net_R"]
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return (eq - 1) * 100, mdd * 100


def main() -> None:
    btc_bars = load("data/hist_1h.parquet")
    eth_bars = load("scratch/hist_1h_eth.parquet")
    sol_bars = load("scratch/hist_1h_sol.parquet")
    boundary_ms = btc_bars[2500].hour_ms + HOUR_MS
    print(f"boundary: {datetime.datetime.utcfromtimestamp(boundary_ms/1000)} UTC")

    # ---- step 1: repo harness reproduction at frozen params ----
    print("\n=== STEP 1: repo run_backtest, frozen params, short_only ===")
    repo_sigs = {}
    for coin, bars in [("BTC", btc_bars), ("ETH", eth_bars), ("SOL", sol_bars)]:
        res = run_backtest(bars, short_only=True)
        sigs = res.get("signals", [])
        repo_sigs[coin] = sigs
        nets = [s["realized_pct"] - COST for s in sigs]
        n = len(sigs)
        hit = sum(1 for s in sigs if s["status"] == "HIT_TP") / max(n, 1)
        print(f"{coin}: n={n} hit={hit*100:.0f}% net={statistics.mean(nets):+.3f}%")

    # ---- step 2: my own loop + identity cross-check ----
    print("\n=== STEP 2: my loop vs repo harness (frozen) ===")
    coins = {"BTC": btc_bars, "ETH": eth_bars, "SOL": sol_bars}
    feats = {c: build_feature_cache(c, b) for c, b in coins.items()}
    mine = {}
    for c, b in coins.items():
        m = my_backtest(b, feats[c], c, 3.0, 1.0, 1.0)
        mine[c] = m
        repo_set = [(s["idx"], round(s["realized_pct"], 9)) for s in repo_sigs[c]]
        my_set = [(s["idx"], round(s["gross"], 9)) for s in m]
        print(f"{c}: my n={len(m)}, repo n={len(repo_sigs[c])}, identical={repo_set == my_set}")
        print("  " + fmt(c, stats(m, boundary_ms)))

    # BTC baseline sized equity
    tot, mdd = equity(mine["BTC"])
    print(f"BTC baseline sized: total {tot:+.2f}% maxDD {mdd:.2f}%")

    # ---- step 3: portfolio BTC+ETH ----
    print("\n=== STEP 3: BTC+ETH portfolio, global cap=3 slots ===")
    acc, skp = portfolio([mine["BTC"], mine["ETH"]], CAP_SLOTS)
    print(fmt("BTC+ETH cap3", stats(acc, boundary_ms)))
    tot, mdd = equity(acc)
    print(f"sized: total {tot:+.2f}% maxDD {mdd:.2f}%  skipped={len(skp)}")
    for s in skp:
        print(f"  SKIPPED: {s['coin']} idx={s['idx']} net={s['net']:+.3f}% "
              f"{datetime.datetime.utcfromtimestamp(s['entry_ms']/1000)}")
    pooled = mine["BTC"] + mine["ETH"]
    print(fmt("BTC+ETH pooled (no cap)", stats(pooled, boundary_ms)))

    # also cap=2, cap=1 variants claimed
    for cap in (2, 1):
        a2, _ = portfolio([mine["BTC"], mine["ETH"]], cap)
        print(fmt(f"BTC+ETH cap{cap}", stats(a2, boundary_ms)))

    # 3-coin portfolio reproduction (claimed REJECTED)
    acc3, _ = portfolio([mine["BTC"], mine["ETH"], mine["SOL"]], 3)
    print(fmt("BTC+ETH+SOL cap3", stats(acc3, boundary_ms)))
    tot3, mdd3 = equity(acc3)
    print(f"sized: total {tot3:+.2f}% maxDD {mdd3:.2f}%")

    # ---- step 4: parameter sensitivity +/-25% one-at-a-time ----
    print("\n=== STEP 4: sensitivity, one-at-a-time +/-25% nudges ===")
    base = (3.0, 1.0, 1.0)
    variants = [
        ("frozen        ", base),
        ("score 2.25    ", (2.25, 1.0, 1.0)),
        ("score 3.75    ", (3.75, 1.0, 1.0)),
        ("move  0.75    ", (3.0, 0.75, 1.0)),
        ("move  1.25    ", (3.0, 1.25, 1.0)),
        ("vol   0.75    ", (3.0, 1.0, 0.75)),
        ("vol   1.25    ", (3.0, 1.0, 1.25)),
    ]
    for label, (sm, mm, vm) in variants:
        b = my_backtest(btc_bars, feats["BTC"], "BTC", sm, mm, vm)
        e = my_backtest(eth_bars, feats["ETH"], "ETH", sm, mm, vm)
        acc_v, _ = portfolio([b, e], CAP_SLOTS)
        sb, sp = stats(b, boundary_ms), stats(acc_v, boundary_ms)
        print(f"{label} BTC-only:  " + fmt("", sb))
        print(f"{label} BTC+ETH :  " + fmt("", sp))


if __name__ == "__main__":
    main()
