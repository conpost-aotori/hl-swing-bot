"""ADVERSARIAL VERIFICATION of the 'consecutive red 4h >= 2' regime overlay.

Independent re-derivation: own red-streak computation (O(1) rolling, not
aggregate_to_4h slicing), own split-half / sensitivity / DD math, plus an
IN-LOOP gating variant that mirrors how the rule would actually run live
(blocked signals no longer set the cooldown -> different trade set than
post-hoc filtering).
"""
from __future__ import annotations

import csv
import statistics
import sys

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")
from hl_swing_bot.backtest import (  # noqa: E402
    BTSignal, HOUR_MS, _composite_score, _compute_features_at, _resolve_outcome,
    run_backtest,
)
from hl_swing_bot.features import HourlyBar, MIN_BARS, aggregate_to_4h  # noqa: E402
from hl_swing_bot.signal import (  # noqa: E402
    COOLDOWN_OPP_DIR_MIN, COOLDOWN_SAME_DIR_MIN, FIRE_MOVE_PER_ATR_MIN,
    FIRE_SCORE_MIN, FIRE_VOL_Z_MIN, SIGNAL_TTL_HOURS, STOP_ATR_MULT,
    TARGET_ATR_MULT,
)

CSV_PATH = r"C:\User\projects\hl-swing-bot\scratch\hist_1h.csv"
COST = 0.19
SPLIT = 2500
BUCKET_MS = 4 * 3600 * 1000


def load_bars() -> list[HourlyBar]:
    out = []
    with open(CSV_PATH) as fh:
        for r in csv.DictReader(fh):
            out.append(HourlyBar(
                hour_ms=int(r["open_time_ms"]), open=float(r["open"]),
                high=float(r["high"]), low=float(r["low"]),
                close=float(r["close"]), volume=float(r["volume"]),
                trades=int(r["trades"])))
    return out


def precompute_red_streaks(bars: list[HourlyBar]) -> tuple[list[int], list[int]]:
    """For each bar index i: (streak_incl, streak_compl).

    streak_incl: consecutive red 4h buckets scanning back from the bucket
    containing bar i, where the current bucket uses close[i] vs bucket open
    (red-so-far). streak_compl: completed buckets only (current excluded).
    Independent implementation: rolling bucket bookkeeping, no aggregate_to_4h.
    """
    n = len(bars)
    streak_incl = [0] * n
    streak_compl = [0] * n
    completed_red_run = 0          # consecutive red completed buckets ending at prev bucket
    cur_bucket_start = None
    cur_bucket_open = None
    prev_close_in_bucket = None    # last close of the bucket being finalized
    for i, b in enumerate(bars):
        bstart = b.hour_ms - (b.hour_ms % BUCKET_MS)
        if bstart != cur_bucket_start:
            # finalize previous bucket
            if cur_bucket_start is not None:
                was_red = prev_close_in_bucket < cur_bucket_open
                completed_red_run = completed_red_run + 1 if was_red else 0
            cur_bucket_start = bstart
            cur_bucket_open = b.open
        prev_close_in_bucket = b.close
        cur_red = b.close < cur_bucket_open
        streak_compl[i] = completed_red_run
        streak_incl[i] = (1 + completed_red_run) if cur_red else 0
    return streak_incl, streak_compl


def cross_check_streaks(bars, streak_incl, idxs):
    """Sanity: recompute streak at given indices via aggregate_to_4h (the
    repo convention the claim references) and compare."""
    bad = 0
    for i in idxs:
        b4 = aggregate_to_4h(bars[: i + 1])
        k = 0
        for bb in reversed(b4):
            if bb.close < bb.open:
                k += 1
            else:
                break
        if k != streak_incl[i]:
            bad += 1
            print(f"  MISMATCH at idx {i}: rolling={streak_incl[i]} agg={k}")
    print(f"streak cross-check on {len(idxs)} signal bars: {'OK' if bad == 0 else f'{bad} mismatches'}")


def stats_block(trades, label):
    """trades: list of dicts with idx, net, status, stop_pct, exit_idx."""
    n = len(trades)
    if n == 0:
        print(f"{label:48s} n=0")
        return None
    net = statistics.mean(t["net"] for t in trades)
    med = statistics.median(t["net"] for t in trades)
    hit = sum(1 for t in trades if t["status"] == "HIT_TP") / n
    rs = [t["net"] / t["stop_pct"] for t in trades]
    avg_r = statistics.mean(rs)
    h1 = [t for t in trades if t["idx"] <= SPLIT]
    h2 = [t for t in trades if t["idx"] > SPLIT]
    net1 = statistics.mean(t["net"] for t in h1) if h1 else float("nan")
    net2 = statistics.mean(t["net"] for t in h2) if h2 else float("nan")
    # sized equity curve: 0.5% risk, additive R accounting ordered by exit
    eq, peak, maxdd = 0.0, 0.0, 0.0
    for t in sorted(trades, key=lambda t: (t["exit_idx"] if t["exit_idx"] is not None else t["idx"])):
        eq += 0.5 * (t["net"] / t["stop_pct"])
        peak = max(peak, eq)
        maxdd = max(maxdd, peak - eq)
    print(f"{label:48s} n={n:3d} net={net:+.3f} med={med:+.3f} hit={hit:.2f} avgR={avg_r:+.3f} "
          f"maxDD={maxdd:.2f}% | h1 n={len(h1):3d} net={net1:+.3f} | h2 n={len(h2):3d} net={net2:+.3f}")
    return {"n": n, "net": net, "net1": net1, "net2": net2, "n1": len(h1),
            "n2": len(h2), "avg_r": avg_r, "maxdd": maxdd, "trades": trades}


def to_trades(sigs):
    return [{
        "idx": s["idx"], "net": s["realized_pct"] - COST, "status": s["status"],
        "stop_pct": abs(s["stop"] / s["entry"] - 1) * 100, "exit_idx": s["exit_idx"],
    } for s in sigs]


def inloop_backtest(bars, streaks, red_min):
    """Replicates run_backtest's loop but with the red-4h gate applied at fire
    time, so blocked signals do NOT set the cooldown (live behavior)."""
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
        if not (score >= FIRE_SCORE_MIN and f["move_per_atr"] >= FIRE_MOVE_PER_ATR_MIN
                and f["vol_z_168"] >= FIRE_VOL_Z_MIN and f["trend_4h"] <= -1):
            continue
        if streaks[i] < red_min:
            continue  # gate: blocked signal does not update cooldown
        atr, entry = f["atr_1h"], f["close"]
        sig = BTSignal(
            idx=i, bar_close_ms=bars[i].hour_ms + HOUR_MS, direction="SHORT",
            entry=entry, stop=entry + STOP_ATR_MULT * atr,
            target=entry - TARGET_ATR_MULT * atr, score=score,
            expires_idx=i + SIGNAL_TTL_HOURS)
        _resolve_outcome(bars, sig, ttl_bars=SIGNAL_TTL_HOURS)
        signals.append(sig)
        last_dir, last_idx = "SHORT", i
    return [{
        "idx": s.idx, "net": s.realized_pct - COST, "status": s.status,
        "stop_pct": abs(s.stop / s.entry - 1) * 100, "exit_idx": s.exit_idx,
    } for s in signals]


def main():
    bars = load_bars()
    n = len(bars)
    print(f"bars={n} span_days={(bars[-1].hour_ms - bars[0].hour_ms) / 86400000:.1f}")
    streak_incl, streak_compl = precompute_red_streaks(bars)

    # ---- 1. reproduce baseline ----
    res = run_backtest(bars, slippage_bps=0.0, short_only=True)
    sigs = res["signals"]
    base_trades = to_trades(sigs)
    print(f"\nbaseline: n={len(sigs)} gross={res['expectancy_pct_post_slippage']:+.3f} "
          f"net={res['expectancy_pct_post_slippage'] - COST:+.3f} hit_tp={res['hit_rate_tp']:.3f}")
    stats_block(base_trades, "BASELINE short-only")

    cross_check_streaks(bars, streak_incl, [s["idx"] for s in sigs])

    # ---- 2. post-hoc filter sensitivity ----
    print("\n--- post-hoc filter: red_incl >= k ---")
    results = {}
    for k in (1, 2, 3, 4):
        kept = [t for t, s in zip(base_trades, sigs) if streak_incl[s["idx"]] >= k]
        results[k] = stats_block(kept, f"red_incl >= {k}")
    print("\n--- post-hoc: completed-buckets-only definition ---")
    for k in (1, 2, 3):
        kept = [t for t, s in zip(base_trades, sigs) if streak_compl[s["idx"]] >= k]
        stats_block(kept, f"red_compl >= {k}")
    print("\n--- post-hoc: REMOVED trades by winner (red_incl < 2) ---")
    removed = [t for t, s in zip(base_trades, sigs) if streak_incl[s["idx"]] < 2]
    stats_block(removed, "removed (red_incl < 2)")

    # ---- 3. outlier sensitivity on winner ----
    win = results[2]["trades"]
    nets = sorted(t["net"] for t in win)
    print(f"\nwinner trade nets sorted: " + " ".join(f"{v:+.2f}" for v in nets))
    print(f"winner net excl best:  {statistics.mean(nets[:-1]):+.3f}")
    print(f"winner net excl worst: {statistics.mean(nets[1:]):+.3f}")
    print(f"winner net excl best+worst: {statistics.mean(nets[1:-1]):+.3f}")

    # ---- 4. alternative split points + quarters ----
    print("\n--- winner by alternative splits ---")
    for sp in (2000, 2500, 3000):
        a = [t for t in win if t["idx"] <= sp]
        b = [t for t in win if t["idx"] > sp]
        ma = statistics.mean(t["net"] for t in a) if a else float("nan")
        mb = statistics.mean(t["net"] for t in b) if b else float("nan")
        print(f"split@{sp}: h1 n={len(a)} net={ma:+.3f} | h2 n={len(b)} net={mb:+.3f}")
    q = (n - MIN_BARS) / 4
    print("--- winner by quarters (idx) ---")
    for qi in range(4):
        lo, hi = MIN_BARS + qi * q, MIN_BARS + (qi + 1) * q
        seg = [t for t in win if lo <= t["idx"] < hi]
        m = statistics.mean(t["net"] for t in seg) if seg else float("nan")
        print(f"Q{qi+1}: n={len(seg)} net={m:+.3f}")
    print("--- baseline by quarters ---")
    for qi in range(4):
        lo, hi = MIN_BARS + qi * q, MIN_BARS + (qi + 1) * q
        seg = [t for t in base_trades if lo <= t["idx"] < hi]
        m = statistics.mean(t["net"] for t in seg) if seg else float("nan")
        print(f"Q{qi+1}: n={len(seg)} net={m:+.3f}")

    # ---- 5. in-loop gating (live implementation) ----
    print("\n--- IN-LOOP gating (cooldown only on fired signals) ---")
    for k in (1, 2, 3):
        tl = inloop_backtest(bars, streak_incl, k)
        stats_block(tl, f"in-loop red_incl >= {k}")

    # ---- 6. cost robustness on winner ----
    print("\n--- winner at higher cost ---")
    for cost in (0.19, 0.25, 0.30):
        m = statistics.mean(t["net"] + COST - cost for t in win)
        print(f"cost={cost:.2f}: net={m:+.3f}")


if __name__ == "__main__":
    main()
