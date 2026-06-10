"""Final grid: BE trigger x TP mult, to verify plateau vs spike."""
from __future__ import annotations

import statistics
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scratch")

from exit_engineering import load_bars, resolve_short, summarize  # noqa: E402
from hl_swing_bot.backtest import run_backtest  # noqa: E402

SL_MULT = 1.5


def main() -> None:
    bars = load_bars()
    res = run_backtest(bars, slippage_bps=0.0, short_only=True)
    entries = [
        {"idx": s["idx"], "entry": s["entry"], "atr": (s["stop"] - s["entry"]) / SL_MULT}
        for s in res["signals"]
    ]

    def run_variant(name, **kw):
        results = [
            resolve_short(bars, e["idx"], e["entry"], e["atr"], **kw)
            for e in entries
        ]
        return summarize(name, results, entries), results

    print("--- BE x TP grid ---")
    for be in (1.0, 1.25, 1.4):
        for tp in (1.75, 2.0, 2.25, 2.5):
            run_variant(f"BE {be} + TP {tp}", be_trigger=be, tp_mult=tp)

    print("--- TP-only sweep, no BE (sanity: is TP 2.0 alone the driver?) ---")
    for tp in (1.75, 2.0, 2.25, 2.5):
        run_variant(f"TP {tp}, no BE", tp_mult=tp)

    print("--- detail: BE 1.25 + TP 2.0 ---")
    _, results = run_variant("BE 1.25 + TP 2.0 (detail)", be_trigger=1.25, tp_mult=2.0)
    gross = [r["realized_pct"] for r in results]
    net = [g - 0.19 for g in gross]
    wins = sum(1 for x in net if x > 0)
    durs = [r["exit_idx"] - e["idx"] for r, e in zip(results, entries)]
    rs = []
    for g, e in zip(gross, entries):
        risk = SL_MULT * e["atr"] / e["entry"] * 100
        rs.append((g - 0.19) / risk)
    print(f"  win_rate(net>0)={wins}/{len(net)}  median_net={statistics.median(net):+.3f}")
    print(f"  avg_net_R={statistics.mean(rs):+.3f}  median_dur_h={statistics.median(durs):.0f}  mean_dur_h={statistics.mean(durs):.1f}")
    print("--- detail: BE 1.25 + TP 2.5 (no new TP param) ---")
    _, results = run_variant("BE 1.25 + TP 2.5 (detail)", be_trigger=1.25, tp_mult=2.5)
    gross = [r["realized_pct"] for r in results]
    net = [g - 0.19 for g in gross]
    wins = sum(1 for x in net if x > 0)
    rs = []
    for g, e in zip(gross, entries):
        risk = SL_MULT * e["atr"] / e["entry"] * 100
        rs.append((g - 0.19) / risk)
    print(f"  win_rate(net>0)={wins}/{len(net)}  median_net={statistics.median(net):+.3f}  avg_net_R={statistics.mean(rs):+.3f}")


if __name__ == "__main__":
    main()
