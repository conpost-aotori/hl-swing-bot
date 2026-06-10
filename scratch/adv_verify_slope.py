"""Adversarial verification of the 'steeper trend' (4h SMA50 slope) gate.

Independently re-derives:
  (a) baseline short-only backtest numbers (n, gross, net of 0.19%)
  (b) filter-on-top variant: keep baseline signals where SMA50(4h) is declining
  (c) integrated-gate variant: slope condition inside the fire gate (cooldown
      only consumed by fired signals) -- matches the proposed implementation
  (d) split-half (signal idx <= 2500 vs >= 2501), n per half
  (e) parameter sensitivity: lookback in {5,7,8,10,12,13,15,20}
  (f) sized maxDD at 0.5% risk per trade, avg net R
  (g) cross-asset check on ETH / SOL
"""
import statistics
import sys

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")

import polars as pl

from hl_swing_bot.backtest import (
    HOUR_MS, BTSignal, _composite_score, _compute_features_at,
    _resolve_outcome, run_backtest,
)
from hl_swing_bot.features import MIN_BARS, HourlyBar, aggregate_to_4h
from hl_swing_bot.signal import (
    COOLDOWN_OPP_DIR_MIN, COOLDOWN_SAME_DIR_MIN, SIGNAL_TTL_HOURS,
    STOP_ATR_MULT, TARGET_ATR_MULT,
)

COST = 0.19  # round-trip cost in % per trade
SPLIT_IDX = 2500  # half1: idx <= 2500, half2: idx >= 2501


def load_bars(path: str) -> list[HourlyBar]:
    df = pl.read_parquet(path)
    return [
        HourlyBar(
            hour_ms=int(r["open_time_ms"]), open=float(r["open"]),
            high=float(r["high"]), low=float(r["low"]),
            close=float(r["close"]), volume=float(r["volume"]),
            trades=int(r["trades"]),
        )
        for r in df.iter_rows(named=True)
    ]


# ---------------------------------------------------------------------------
# slope feature, computed exactly as the implementation notes specify
# ---------------------------------------------------------------------------

_b4_cache: dict[tuple[int, int], list] = {}


def bars4h_at(bars, i, key):
    ck = (key, i)
    if ck not in _b4_cache:
        _b4_cache[ck] = aggregate_to_4h(bars[: i + 1])
    return _b4_cache[ck]


def slope_diff(bars, i, key, lookback=10, sma=50):
    """sma50_now - sma50_{lookback bars ago}; None if insufficient history."""
    b4 = bars4h_at(bars, i, key)
    need = sma + lookback + 1
    if len(b4) < need:
        return None
    now = statistics.mean(b.close for b in b4[-(sma + 1):-1])
    prev = statistics.mean(
        b.close for b in b4[-(sma + lookback + 1):-(lookback + 1)]
    )
    return now - prev


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def metrics(sigs, label):
    if not sigs:
        print(f"  {label}: n=0")
        return None
    nets = [s["realized_pct"] - COST for s in sigs]
    tp = sum(1 for s in sigs if s["status"] == "HIT_TP")
    # sized PnL: 0.5% equity risk per trade, R = net_pct / risk_pct
    rs, equity_curve, cum, peak, maxdd = [], [], 0.0, 0.0, 0.0
    for s in sorted(sigs, key=lambda x: x["idx"]):
        risk_pct = abs(s["stop"] - s["entry"]) / s["entry"] * 100
        r = (s["realized_pct"] - COST) / risk_pct
        rs.append(r)
        cum += 0.5 * r
        peak = max(peak, cum)
        maxdd = max(maxdd, peak - cum)
    h1 = [s["realized_pct"] - COST for s in sigs if s["idx"] <= SPLIT_IDX]
    h2 = [s["realized_pct"] - COST for s in sigs if s["idx"] > SPLIT_IDX]
    print(
        f"  {label}: n={len(sigs)} hit={tp/len(sigs):.0%} "
        f"net={statistics.mean(nets):+.3f}%/tr "
        f"h1(n={len(h1)})={statistics.mean(h1):+.3f} "
        f"h2(n={len(h2)})={statistics.mean(h2):+.3f} "
        f"avgR={statistics.mean(rs):+.3f} maxDD(sized)={maxdd:.2f}%"
    )
    return {
        "n": len(sigs), "net": statistics.mean(nets),
        "h1n": len(h1), "h1": statistics.mean(h1) if h1 else None,
        "h2n": len(h2), "h2": statistics.mean(h2) if h2 else None,
        "maxdd": maxdd, "avgR": statistics.mean(rs),
    }


# ---------------------------------------------------------------------------
# integrated-gate backtest (slope inside the fire gate)
# ---------------------------------------------------------------------------

def run_backtest_sloped(bars, key, lookback=10):
    signals = []
    last_dir, last_idx = None, -10_000
    for i in range(MIN_BARS, len(bars)):
        f = _compute_features_at(bars, i)
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
        if not (
            score >= 3.0 and f["move_per_atr"] >= 1.0
            and f["vol_z_168"] >= 1.0 and f["trend_4h"] <= -1
        ):
            continue
        sd = slope_diff(bars, i, key, lookback=lookback)
        if sd is None or sd >= 0:  # undefined slope blocks; require declining
            continue
        atr = f["atr_1h"]
        entry = f["close"]
        sig = BTSignal(
            idx=i, bar_close_ms=bars[i].hour_ms + HOUR_MS, direction="SHORT",
            entry=entry, stop=entry + STOP_ATR_MULT * atr,
            target=entry - TARGET_ATR_MULT * atr, score=score,
            expires_idx=i + SIGNAL_TTL_HOURS,
        )
        _resolve_outcome(bars, sig, ttl_bars=SIGNAL_TTL_HOURS)
        signals.append({
            "idx": sig.idx, "direction": "SHORT", "entry": sig.entry,
            "stop": sig.stop, "status": sig.status,
            "realized_pct": sig.realized_pct, "score": sig.score,
        })
        last_dir, last_idx = "SHORT", i
    return signals


def analyze(path, key):
    print(f"\n=== {key} ({path}) ===")
    bars = load_bars(path)
    print(f"bars: {len(bars)}")

    base = run_backtest(bars, short_only=True)
    bsigs = base["signals"]
    m_base = metrics(bsigs, "BASELINE        ")

    # filter-on-top
    kept, removed, undef = [], [], []
    for s in bsigs:
        sd = slope_diff(bars, s["idx"], key)
        if sd is None:
            undef.append(s)
        elif sd < 0:
            kept.append(s)
        else:
            removed.append(s)
    m_kept = metrics(kept, "SLOPE filter-top")
    if removed:
        metrics(removed, "  removed(sd>=0)")
    if undef:
        metrics(undef, "  removed(undef)")

    # integrated gate
    integ = run_backtest_sloped(bars, key)
    m_int = metrics(integ, "SLOPE integrated")
    extra = sorted(set(s["idx"] for s in integ) - set(s["idx"] for s in kept))
    missing = sorted(set(s["idx"] for s in kept) - set(s["idx"] for s in integ))
    print(f"  integrated-vs-filtertop: extra idx={extra} missing idx={missing}")

    # parameter sensitivity (BTC only is enough, but cheap everywhere)
    print("  lookback sensitivity (filter-on-top):")
    sel10 = None
    for lb in [5, 7, 8, 10, 12, 13, 15, 20]:
        k2 = [s for s in bsigs if (sd := slope_diff(bars, s["idx"], key, lookback=lb)) is not None and sd < 0]
        nets = [s["realized_pct"] - COST for s in k2]
        ids = tuple(s["idx"] for s in k2)
        if lb == 10:
            sel10 = ids
        same = "SAME-SELECTION" if ids == sel10 else f"diff({len(set(ids) ^ set(sel10 or ()))})" if sel10 else ""
        print(
            f"    lb={lb:>2}: n={len(k2)} net={statistics.mean(nets):+.3f} {same}"
        )

    # SMA window sensitivity (the other hidden parameter)
    print("  sma-window sensitivity (filter-on-top, lb=10):")
    for sw in [40, 50, 60]:
        k2 = [s for s in bsigs if (sd := slope_diff(bars, s["idx"], key, sma=sw)) is not None and sd < 0]
        nets = [s["realized_pct"] - COST for s in k2]
        h1 = [s["realized_pct"] - COST for s in k2 if s["idx"] <= SPLIT_IDX]
        h2 = [s["realized_pct"] - COST for s in k2 if s["idx"] > SPLIT_IDX]
        print(
            f"    sma={sw}: n={len(k2)} net={statistics.mean(nets):+.3f} "
            f"h1(n={len(h1)})={statistics.mean(h1):+.3f} h2(n={len(h2)})={statistics.mean(h2):+.3f}"
        )
    return m_base, m_kept, m_int


if __name__ == "__main__":
    analyze(r"C:\User\projects\hl-swing-bot\data\hist_1h.parquet", "BTC")
    analyze(r"C:\User\projects\hl-swing-bot\scratch\hist_1h_eth.parquet", "ETH")
    analyze(r"C:\User\projects\hl-swing-bot\scratch\hist_1h_sol.parquet", "SOL")
