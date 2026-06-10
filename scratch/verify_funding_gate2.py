"""Follow-up adversarial probes: trade clustering, baseline quarter coverage,
H2 dispersion, event-deduped stats, bootstrap of gate effect."""
import json
import random
import statistics
import sys

import duckdb

sys.path.insert(0, r"C:\User\projects\hl-swing-bot\src")
from hl_swing_bot.backtest import run_backtest
from hl_swing_bot.features import HourlyBar

COST = 0.19
H = 3600 * 1000

con = duckdb.connect()
rows = con.execute(
    "SELECT open_time_ms, open, high, low, close, volume, trades "
    "FROM read_parquet('C:/User/projects/hl-swing-bot/data/hist_1h.parquet') "
    "ORDER BY open_time_ms"
).fetchall()
bars = [HourlyBar(hour_ms=int(r[0]), open=float(r[1]), high=float(r[2]),
                  low=float(r[3]), close=float(r[4]), volume=float(r[5]),
                  trades=int(r[6])) for r in rows]

with open(r"C:\User\projects\hl-swing-bot\scratch\verify_funding_btc.json") as fh:
    rate = {int(t) // H * H: float(v) for t, v in json.load(fh)}

res = run_backtest(bars, slippage_bps=0.0, short_only=True)
base = res["signals"]
for s in base:
    s["apr"] = rate[bars[s["idx"]].hour_ms + H] * 24 * 365 * 100
    s["net"] = s["realized_pct"] - COST

print("baseline per-quarter:")
for k, (a, b) in enumerate([(0, 1250), (1251, 2500), (2501, 3751), (3752, 9999)], 1):
    sub = [s for s in base if a <= s["idx"] <= b]
    gated = [s for s in sub if s["apr"] >= 8.0]
    print(f"  Q{k}: baseline n={len(sub):2d} net={statistics.mean([s['net'] for s in sub]) if sub else 0:+.3f}"
          f" | gated n={len(gated):2d}"
          + (f" net={statistics.mean([s['net'] for s in gated]):+.3f}" if gated else ""))

# H2 gated trades detail (post-hoc on baseline; in-loop adds idx 4020)
h2g = [s for s in base if s["idx"] > 2500 and s["apr"] >= 8.0]
print("\nH2 gated trades (post-hoc):")
for s in h2g:
    print(f"  idx={s['idx']:4d} status={s['status']:7s} net={s['net']:+.3f} apr={s['apr']:.2f}")

# event clustering in the in-loop winner set (from run 1: 25 trades)
# rebuild winner set quickly via post-hoc + known extras [777,1906,4020]
# -> approximate with post-hoc here for clustering only
wins = sorted([s for s in base if s["apr"] >= 8.0], key=lambda s: s["idx"])
clusters = []
cur = [wins[0]]
for s in wins[1:]:
    if s["idx"] - cur[-1]["idx"] <= 72:  # within TTL window of prior trade
        cur.append(s)
    else:
        clusters.append(cur)
        cur = [s]
clusters.append(cur)
print(f"\nwinner trades grouped into {len(clusters)} non-overlapping (>72h apart) clusters "
      f"out of {len(wins)} trades")
cl_nets = [statistics.mean([s["net"] for s in c]) for c in clusters]
print("cluster mean-nets:", [round(x, 2) for x in cl_nets])
print(f"mean of cluster means: {statistics.mean(cl_nets):+.3f}, "
      f"positive clusters: {sum(1 for x in cl_nets if x > 0)}/{len(cl_nets)}")

# bootstrap: gate effect = mean(gated) - mean(all). Resample baseline trades.
nets_all = [s["net"] for s in base]
nets_g = [s["net"] for s in base if s["apr"] >= 8.0]
obs = statistics.mean(nets_g) - statistics.mean(nets_all)
random.seed(42)
# permutation test: shuffle apr labels across trades, recompute gated mean diff
aprs = [s["apr"] for s in base]
count = 0
B = 20000
for _ in range(B):
    random.shuffle(aprs)
    g = [nv for nv, a in zip(nets_all, aprs) if a >= 8.0]
    if g and statistics.mean(g) - statistics.mean(nets_all) >= obs:
        count += 1
print(f"\npermutation test (shuffle funding labels across the 49 trades): "
      f"observed diff={obs:+.3f}, p={count/B:.4f}")

# same but on event-cluster level (shuffle at cluster level to respect correlation)
base_sorted = sorted(base, key=lambda s: s["idx"])
bcl = []
cur = [base_sorted[0]]
for s in base_sorted[1:]:
    if s["idx"] - cur[-1]["idx"] <= 72:
        cur.append(s)
    else:
        bcl.append(cur)
        cur = [s]
bcl.append(cur)
cl_net = [statistics.mean([s["net"] for s in c]) for c in bcl]
cl_apr = [statistics.mean([s["apr"] for s in c]) for c in bcl]
g = [v for v, a in zip(cl_net, cl_apr) if a >= 8.0]
obs_cl = statistics.mean(g) - statistics.mean(cl_net)
print(f"baseline has {len(bcl)} clusters; cluster-level gate diff={obs_cl:+.3f} "
      f"(gated clusters n={len(g)})")
count = 0
for _ in range(B):
    random.shuffle(cl_apr)
    g = [v for v, a in zip(cl_net, cl_apr) if a >= 8.0]
    if g and statistics.mean(g) - statistics.mean(cl_net) >= obs_cl:
        count += 1
print(f"cluster-level permutation p={count/B:.4f}")
