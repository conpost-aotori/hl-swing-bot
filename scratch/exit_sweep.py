"""Parameter-sensitivity sweep around the winning exit variants."""
from __future__ import annotations

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
        return summarize(name, results, entries)

    print("--- trail (no TP) sweep ---")
    for t in (1.75, 2.0, 2.25, 2.5, 3.0):
        run_variant(f"trail {t} ATR, no TP", trail_mult=t, tp_mult=None)
    print("--- trail (no TP) + BE sweep ---")
    for t in (2.0, 2.5):
        for be in (1.0, 1.25):
            run_variant(f"trail {t} + BE {be}, no TP", trail_mult=t, tp_mult=None, be_trigger=be)
    print("--- BE trigger sweep (TP 2.5 kept) ---")
    for be in (0.9, 1.0, 1.1, 1.25, 1.4, 1.5):
        run_variant(f"BE after {be} ATR", be_trigger=be)
    print("--- BE 1.25 with TP sweep ---")
    for tp in (2.0, 2.5, 3.0):
        run_variant(f"BE 1.25 + TP {tp}", be_trigger=1.25, tp_mult=tp)
    print("--- trail with TP sweep ---")
    for t in (1.75, 2.0, 2.25):
        for tp in (2.5, 3.0):
            run_variant(f"trail {t} + TP {tp}", trail_mult=t, tp_mult=tp)


if __name__ == "__main__":
    main()
