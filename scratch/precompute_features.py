"""Precompute per-bar features once (the O(n^2) part) and cache to parquet,
so variant runs don't re-trigger the CPython 3.13 crash-prone hot loop."""
import sys

import duckdb

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")
from hl_swing_bot.backtest import _compute_features_at, _composite_score
from hl_swing_bot.features import HourlyBar, MIN_BARS

con = duckdb.connect()
bar_rows = con.execute(
    "SELECT open_time_ms, open, high, low, close, volume, trades "
    "FROM read_parquet('C:/User/projects/hl-swing-bot/data/hist_1h.parquet') ORDER BY open_time_ms"
).fetchall()
bars = [HourlyBar(hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]), low=float(r[3]),
                  close=float(r[4]), volume=float(r[5]), trades=int(r[6])) for r in bar_rows]
print(f"bars: {len(bars)}")

rows = []
for i in range(MIN_BARS, len(bars)):
    f = _compute_features_at(bars, i)
    if not f:
        continue
    rows.append((i, f["close"], f["atr_1h"], f["ret_1h"], f["move_per_atr"],
                 f["vol_z_168"], int(f["trend_4h"]), _composite_score(f)))
    if i % 500 == 0:
        print(f"  {i}/{len(bars)}", flush=True)

con.execute("CREATE TABLE feat (idx INT, close DOUBLE, atr DOUBLE, ret_1h DOUBLE, "
            "move_per_atr DOUBLE, vol_z DOUBLE, trend INT, score DOUBLE)")
con.executemany("INSERT INTO feat VALUES (?,?,?,?,?,?,?,?)", rows)
con.execute("COPY feat TO 'C:/User/projects/hl-swing-bot/scratch/features_cache.parquet' (FORMAT PARQUET)")
print(f"saved {len(rows)} feature rows")
