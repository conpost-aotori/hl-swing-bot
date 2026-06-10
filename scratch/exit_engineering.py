"""Exit-engineering experiments on the 208d short-only baseline.

Reuses run_backtest() for the 49 short entries, then re-resolves outcomes
with alternative exit rules over the same hist_1h.parquet bars.

All exits are resolved conservatively:
- Within a bar, stop is checked BEFORE target (same as repo resolver).
- Stop adjustments (breakeven / trailing) computed from a bar's extremes
  only take effect from the NEXT bar (no intrabar trail-then-stop).
Cost model: NET = gross - 0.19 (% round trip). Partial exits pay the same
0.19% on full notional (each half pays its own round trip).
"""
from __future__ import annotations

import statistics
import sys

import duckdb

sys.path.insert(0, "src")
from hl_swing_bot.backtest import run_backtest  # noqa: E402
from hl_swing_bot.features import HourlyBar  # noqa: E402

COST = 0.19  # round-trip %
SL_MULT = 1.5
SPLIT_IDX = 2500  # entries idx <= 2500 -> half1, else half2


def load_bars() -> list[HourlyBar]:
    rows = duckdb.sql(
        "SELECT open_time_ms, open, high, low, close, volume, trades "
        "FROM 'data/hist_1h.parquet' ORDER BY open_time_ms"
    ).fetchall()
    return [
        HourlyBar(
            hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
            low=float(r[3]), close=float(r[4]), volume=float(r[5]),
            trades=int(r[6]),
        )
        for r in rows
    ]


def resolve_short(
    bars: list[HourlyBar],
    idx: int,
    entry: float,
    atr: float,
    *,
    ttl: int = 72,
    sl_mult: float = SL_MULT,
    tp_mult: float | None = 2.5,
    be_trigger: float | None = None,   # move stop to entry after this ATR move in favor
    trail_mult: float | None = None,   # trail stop at this ATR from best favorable low
    partial: tuple[float, float] | None = None,  # (fraction, tp1_atr_mult); BE on rest after tp1
) -> dict:
    """Re-resolve a SHORT trade. Returns dict with realized_pct (full position,
    weighted if partial), status, exit_idx."""
    stop = entry + sl_mult * atr
    target = entry - tp_mult * atr if tp_mult is not None else None
    best = entry  # best (lowest) favorable price seen so far
    end_idx = min(idx + ttl, len(bars) - 1)

    leg1_pct: float | None = None  # realized pct of the partial leg
    frac = partial[0] if partial else 0.0
    tp1 = entry - partial[1] * atr if partial else None
    tp1_done = False

    def short_pct(px: float) -> float:
        return (entry / px - 1) * 100

    def blend(rest_pct: float) -> float:
        if partial and tp1_done:
            return frac * leg1_pct + (1 - frac) * rest_pct
        return rest_pct

    for j in range(idx + 1, end_idx + 1):
        b = bars[j]
        # 1) stop first (conservative)
        if b.high >= stop:
            return {
                "realized_pct": blend(short_pct(stop)),
                "status": "HIT_SL" if stop > entry else ("BE" if stop == entry else "TRAIL"),
                "exit_idx": j,
            }
        # 2) partial TP1
        if partial and not tp1_done and b.low <= tp1:
            leg1_pct = short_pct(tp1)
            tp1_done = True
            # breakeven on the rest takes effect from NEXT bar
            # (recorded below after target check via pending flag)
        # 3) full target
        if target is not None and b.low <= target:
            return {
                "realized_pct": blend(short_pct(target)),
                "status": "HIT_TP",
                "exit_idx": j,
            }
        # 4) update stop for NEXT bar
        best = min(best, b.low)
        new_stop = stop
        if be_trigger is not None and best <= entry - be_trigger * atr:
            new_stop = min(new_stop, entry)
        if trail_mult is not None:
            new_stop = min(new_stop, best + trail_mult * atr)
        if partial and tp1_done:
            new_stop = min(new_stop, entry)
        stop = new_stop

    # expired
    exit_px = bars[end_idx].close
    return {
        "realized_pct": blend(short_pct(exit_px)),
        "status": "EXPIRED",
        "exit_idx": end_idx,
    }


def summarize(name: str, results: list[dict], entries: list[dict]) -> dict:
    gross = [r["realized_pct"] for r in results]
    n = len(gross)
    net = [g - COST for g in gross]
    h1 = [g - COST for g, e in zip(gross, entries) if e["idx"] <= SPLIT_IDX]
    h2 = [g - COST for g, e in zip(gross, entries) if e["idx"] > SPLIT_IDX]
    tp = sum(1 for r in results if r["status"] == "HIT_TP")
    sl = sum(1 for r in results if r["status"] == "HIT_SL")
    be = sum(1 for r in results if r["status"] in ("BE", "TRAIL"))
    exp = sum(1 for r in results if r["status"] == "EXPIRED")
    # sized equity / maxDD: 0.5% risk per trade against initial 1.5-ATR stop
    eq, peak, maxdd = 1.0, 1.0, 0.0
    for g, e in zip(gross, entries):
        risk_pct = SL_MULT * e["atr"] / e["entry"] * 100  # initial stop distance %
        r_mult = (g - COST) / risk_pct
        eq *= 1 + 0.005 * r_mult
        peak = max(peak, eq)
        maxdd = max(maxdd, (peak - eq) / peak)
    out = {
        "name": name, "n": n,
        "gross": statistics.mean(gross), "net": statistics.mean(net),
        "net_h1": statistics.mean(h1) if h1 else float("nan"),
        "n_h1": len(h1),
        "net_h2": statistics.mean(h2) if h2 else float("nan"),
        "n_h2": len(h2),
        "tp": tp, "sl": sl, "be_or_trail": be, "exp": exp,
        "maxdd_sized": maxdd * 100,
    }
    print(
        f"{name:<42} n={n:>2} gross={out['gross']:+.3f} NET={out['net']:+.3f} "
        f"| h1({out['n_h1']})={out['net_h1']:+.3f} h2({out['n_h2']})={out['net_h2']:+.3f} "
        f"| TP={tp} SL={sl} BE/TR={be} EXP={exp} | maxDD={out['maxdd_sized']:.1f}%"
    )
    return out


def main() -> None:
    bars = load_bars()
    print(f"bars: {len(bars)}  ({bars[0].hour_ms} .. {bars[-1].hour_ms})")

    res = run_backtest(bars, slippage_bps=0.0, short_only=True)
    sigs = res["signals"]
    print(f"baseline entries: {res['n_signals']} shorts, "
          f"harness gross={res['expectancy_pct_post_slippage']:+.4f} "
          f"TP={res['tp_count']} SL={res['sl_count']} EXP={res['expired_count']}")

    entries = []
    for s in sigs:
        atr = (s["stop"] - s["entry"]) / SL_MULT  # short: stop above entry
        entries.append({"idx": s["idx"], "entry": s["entry"], "atr": atr})

    def run_variant(name, **kw):
        results = [
            resolve_short(bars, e["idx"], e["entry"], e["atr"], **kw)
            for e in entries
        ]
        return summarize(name, results, entries)

    print()
    run_variant("V0 baseline replicate (1.5SL/2.5TP/72h)")
    run_variant("V1 breakeven after 1.0 ATR", be_trigger=1.0)
    run_variant("V1b breakeven after 0.75 ATR", be_trigger=0.75)
    run_variant("V1c breakeven after 1.25 ATR", be_trigger=1.25)
    run_variant("V2 trail 1.5 ATR (TP 2.5 kept)", trail_mult=1.5)
    run_variant("V2b trail 1.5 ATR, no TP", trail_mult=1.5, tp_mult=None)
    run_variant("V2c trail 2.0 ATR (TP 2.5 kept)", trail_mult=2.0)
    run_variant("V2d trail 1.0 ATR (TP 2.5 kept)", trail_mult=1.0)
    run_variant("V3 partial 50% @1.25, rest 2.5 + BE", partial=(0.5, 1.25))
    run_variant("V4a time-stop 24h", ttl=24)
    run_variant("V4b time-stop 48h", ttl=48)
    run_variant("V5 TP 3.5 + BE after 1.0 ATR", tp_mult=3.5, be_trigger=1.0)
    print()
    # combos
    run_variant("C1 BE 1.0 + 48h TTL", be_trigger=1.0, ttl=48)
    run_variant("C2 BE 1.0 + 24h TTL", be_trigger=1.0, ttl=24)
    run_variant("C3 partial(0.5,1.25) + 48h TTL", partial=(0.5, 1.25), ttl=48)
    run_variant("C4 trail 2.0, no TP", trail_mult=2.0, tp_mult=None)
    run_variant("C5 TP 3.5 + trail 1.5", tp_mult=3.5, trail_mult=1.5)
    run_variant("C6 partial(0.5,1.0) rest 2.5 + BE", partial=(0.5, 1.0))
    run_variant("C7 TP 3.5 + BE 1.0 + 48h", tp_mult=3.5, be_trigger=1.0, ttl=48)
    run_variant("C8 24h TTL alone (no BE)", ttl=24)


if __name__ == "__main__":
    main()
