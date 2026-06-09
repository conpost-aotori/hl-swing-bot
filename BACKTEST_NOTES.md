# Backtest Notes — 2026-06-08

First real backtest, run on 455 hourly bars (2026-05-20 → 06-08, 19 days).

## Headline numbers (baseline: score≥3.0, move/ATR≥1.0, vol_z≥1.0)

| metric | value |
|---|---|
| signals | 6 |
| signals/week | 2.55 (in SPEC target band 2–5) |
| TP / SL / expired | 4 / 2 / 0 |
| hit rate | 67% |
| expectancy (post-5bps slippage) | +0.76% / trade |
| median | +1.48% |
| best / worst | +2.58% / −1.60% |

**Taken at face value this looks great. It is not. Read on.**

## ⚠️ The result is one regime, not a validation

All 6 signals fired in a **3-day window: June 1–3**, during the early phase of a
BTC waterfall (73k → 64k). The other 16 of 19 days produced **zero signals**.

```
06-01 14:00  SHORT 71,700 → TP  +1.46%
06-02 13:00  SHORT 68,975 → TP  +1.69%
06-02 23:00  SHORT 66,345 → SL  −1.48%
06-03 04:00  SHORT 65,812 → SL  −1.50%
06-03 16:00  SHORT 66,055 → TP  +2.30%
06-03 23:00  SHORT 64,884 → TP  +2.68%
```

- **0 LONG signals, ever.** The entire dataset is a downtrend (77k → 63k, −18%).
  The long side of the strategy is completely **untested**.
- The positive expectancy is conditional on catching the onset of **one** crash.
  This is n=1 regime, not n=6 independent trades.

## Why signals stopped after June 3 (even though 06-04→06-07 crashed harder)

ATR expanded through the crash, and `move/ATR` collapsed as a result:

| day | atr_pct | move/ATR (sampled) |
|---|---|---|
| 06-01 | 0.37% | ~0.8 → fires |
| 06-03 | 0.98% | ~0.25 |
| 06-06 | 1.70% | ~0.21 → silent |

**Structural finding: this strategy is a volatility-expansion-ONSET detector, not a
trend rider.** It fires when vol starts expanding (while ATR is still catching up),
then self-silences once high vol becomes the 168-bar norm. Because stops/targets
scale with ATR (1.5×/2.5×), late entries would carry huge stops anyway — so the
silence is arguably correct behavior, but it means the bot will be quiet for long
stretches and only speak at regime changes.

## The `vol_z` gate is effectively inert (for now)

Sweeping vol_z ∈ {0.5, 1.0} changes **nothing** across every other parameter combo.
In the only regime that fired (a high-vol crash), volume was always elevated, so the
gate never bound. It is not dead code — it would matter in calmer regimes — but it
currently provides zero discrimination. Do not tune it on this data.

## Parameter sweep — the plateau is robust

| score | move | n | hit% | exp% |
|---|---|---|---|---|
| 2.0 | 1.0 | 13 | 54% | +0.32 |
| 2.5 | 1.0 | 9 | 67% | +0.77 |
| 3.0 | 1.0 | 6 | 67% | +0.76 |
| 3.5 | 1.0 | 6 | 67% | +0.80 |

score 2.5–3.5 is a stable plateau (not a knife-edge optimum) — good. score 2.0 clearly
degrades. **Keep score≥3.0.** But note all of this is measured inside the same crash.

## Conclusions

1. **The strategy is unfalsified, not validated.** It behaved sensibly in one downtrend.
   We have no evidence about uptrends, chop, or the long side.
2. **Do not tune parameters on this data.** 6 trades in one regime will overfit instantly.
   The score≥3.0 / move≥1.0 defaults are fine; leave them.
3. **The user's decision to postpone live trading is correct.** Launching now would be
   trading a strategy validated on exactly one crash.

## Next steps (priority order)

- [ ] **Keep collecting.** We need at least one sustained uptrend in the dataset to get
      any LONG signals and confirm the trend filter works symmetrically.
- [ ] **Add regime-discriminating features** (Phase 1.5) so the model isn't relying on
      price/vol alone — options skew, liquidation cascades, basis. (Survey in progress.)
- [ ] Re-run this backtest monthly; watch for the first LONG signals appearing.
- [ ] Consider logging *would-be* signals (paper-trail) even below threshold, so we
      accumulate labeled near-misses for Phase 2 ML.

---

## Multi-regime walk-forward — 2026-06-10 (THE decisive test)

Fetched the **full available HL 1h history: 5,002 bars = 208 days (2025-11-13 → 2026-06-09)**.
(HL `candleSnapshot` caps at ~5000 bars — no deeper history exists on the venue.) Ran the
EXISTING frozen params (score≥3.0, move≥1.0) — no tuning. This is 11× the prior 19-day window
and spans real uptrends, downtrends, and chop. Script: `scripts/walkforward.py`
(run with `PYTHON_JIT=0` — CPython 3.13 JIT crashes on the O(n²) hot loop).

**Headline answers:**
1. **Does LONG ever fire? → YES (38 LONG / 49 SHORT, 87 signals).** The strategy is NOT
   short-only by construction; the 19-day sample was just a downtrend. Good — that uncertainty
   is resolved.
2. **As built, there is NO edge after realistic costs.** Gross +0.16%/trade → **net ≈ −0.03%**
   at 0.19% round-trip (0.09% fees + 0.10% slippage), −0.14% at 0.30% RT. 40% hit-rate. The
   famous "+0.76%, 67%" was ONE lucky crash; over 7 months it's a coin flip.

**Per-regime (net of 0.19% round-trip) — the critical finding:**

| regime | n | L/S | gross | **net** |
|---|---|---|---|---|
| uptrend | 17 | 12/5 | −0.64% | **−0.83%** ❌ loses |
| downtrend (short) | 23 | 3/20 | +0.66% | **+0.47%** ✅ only edge |
| chop | 39 | 19/20 | +0.10% | **−0.09%** ❌ bleeds |
| **ALL** | 87 | 38/49 | +0.16% | **−0.03%** |

**Episode-level (72h same-dir merged = honest n): 87 raw → 33 independent episodes,
45% win-rate, +0.12% gross → ≈negative net.** No demonstrated edge.

**Interpretation:** The TA composite's only profitable cell is **short-side during
downtrends** (+0.47% net). It actively LOSES in uptrends (−0.83%, where the LONG side fires)
and bleeds in chop (−0.09%). So the bot is really a *short-cascade detector with a losing
long side and a losing chop habit bolted on*. The symmetric "swing both ways" thesis is
**falsified** for the current TA-only LONG branch.

**Kill-switch status:** NOT triggered (LONG fires; ≥3 regime cells; gross expectancy not <0).
But it is clearly "no edge as built" — marginal/breakeven, negative after honest costs.

**Implication for profitability (two paths):**
- **(A) Specialize → short-only downtrend bot.** Trade ONLY when trend_4h is down. Keeps the
  +0.47%-net cell, deletes the −0.83% uptrend bleed and −0.09% chop. Simplest, data-honest.
- **(B) Fix the LONG side with the liquidation bias** (squeeze detection) so longs stop losing
  — higher upside, unproven, needs the bias Spearman test first.
The data favors (A) now, (B) as the R&D track to earn back the long side.

### Path A IMPLEMENTED + validated (short-only, `ENABLE_LONG=False`)

Re-ran the 208d walk-forward with `short_only=True` (drops the losing LONG branch):

| mode | n | hit | gross | **net (0.19% RT)** | maxDD |
|---|---|---|---|---|---|
| both-ways | 87 | 40% | +0.16% | **−0.03%** | 16.3% |
| **SHORT-only** | 49 | 45% | +0.28% | **+0.09%** | **9.8%** |

Dropping longs flips net expectancy **positive** (+0.09%/trade, ≈+3.9% compounded over 7mo)
and **halves drawdown** (16.3%→9.8%). Thin but real, and structurally motivated (cascade
asymmetry). **Now live in paper as short-only.** Suppressed would-be LONGs are logged for
out-of-sample confirmation + Path-B. Caveats: still in-sample selection; +0.09% is marginal;
maxDD 9.8% still >8% graduation gate (→ needs the fractional-risk sizer). Forward paper +
the kill-switch remain the real test.

### Path A sizing added (fractional-risk + cluster cap) — DD now passes the gate

Added fixed-fractional-RISK sizing (`RISK_FRAC=0.005`, each stop-out loses exactly 0.5% of
equity regardless of the 4.6× atr_pct swing) + an aggregate `CLUSTER_RISK_CAP=1.5%` (max ~3
simultaneous shorts). Re-simulated the 208d short-only equity path with proper R-multiples:

| sizing | trades | final eq | **maxDD** | avg net R |
|---|---|---|---|---|
| naive all-in (prior) | 49 | +3.9% | 9.8% | — |
| **0.5%-risk + cluster cap** | 49 | +0.7% | **4.7%** ✅ | +0.03 |

**maxDD 4.7% now passes the <8% graduation gate.** Risk is properly controlled. BUT the
edge is genuinely thin — avg net R = +0.03, +0.7% compounded over 7 months. So: *not losing,
controlled risk, but not yet meaningfully profitable.* Real upside must come from Path B
(rebuild a profitable LONG via the bias squeeze signal) and/or sharpening short entries.
The cluster cap didn't bind on this sample (cooldown spacing kept <3 open) — safety net.

### Path B prep (liquidation-bias long rebuild)
`compute_features()` now reads `liq_bias` (forward-filled from the store) into every feature
dict, so each emitted signal records the bias at signal time. Not yet used in the score —
that waits on the Spearman(bias, forward-return) test once enough forward bias accumulates.

## Quant panel review — 2026-06-10 (Codex + Grok + 6-lens panel)

Three independent reviews converged: **as built, this is a coin-flip after costs.** The
edge (if any) is in forced-flow microstructure (liquidations, funding, positioning),
NOT in the price/TA composite score. Real bugs found by reading the code:

1. **`signal.py` composite score scale bug** — `abs(f.get("move_per_atr_z",0.0) or
   f["move_per_atr"])`: when the z-score is exactly 0.0 it fell back to the raw ratio
   (different scale) in the 0.30-weight lead term. **FIXED 2026-06-10** (+ same in
   `backtest.py`). Didn't change the crash trades (large z there) but corrupted scores
   in neutral regimes.
2. **`realized_return` booked zero costs** — paper P&L was gross. **FIXED 2026-06-10**:
   now net of round-trip fees+slippage (`COST_RT_PCT=0.19%`). Funding-over-hold still
   TODO (can be material on 72h holds).
3. **3 of 5 score terms are collinear** (move_per_atr / robust_z_168 / ret_4h all = "price
   moved far from weekly median"). ~0.70 weight on one factor. → redesign (not yet done).
4. **`vol_z` double-counted** (score term + gate) and computed as robust-z not std-z →
   near-constant offset. → redesign.
5. **`funding_bonus` is perverse** — rewards *non-extreme* funding, direction-blind;
   penalizes the crash-shorts that actually fire. → redesign.
6. **liquidation bias is unwired** — ingested to the feature store but `compute_features()`
   never reads it back, so `signal.py` can't see it. This is the #1 opportunity (only
   directional, causal, orthogonal feature). → wire as signed term (after offline test).

Honest cost re-accounting of the existing 6 trades (still n≈1 regime): survive even
heavy slippage (+0.76% → +0.56% → +0.26% at 5/15/30 bps-per-side). They survive because
crash-onset moves are large — NOT evidence of generalizable edge.

## ⛔ PRE-REGISTERED KILL-SWITCH (committed 2026-06-10, do not move the goalposts)

The SPEC's "30+ signals, expectancy>0" criterion is **unfalsifiable-by-construction** —
clustered crash-shorts can tick that box while telling us nothing about LONG/chop. Replace
it with a regime-stratified bar. **Abandon the Phase-1 rule-based approach by 2026-09-10
(90 days) if ANY of:**

- (a) **zero LONG signals** ever fired across paper + offline walk-forward, OR
- (b) **fewer than 3 temporally-independent regime-events** (signals within 72h / same
  direction collapse into ONE event), OR
- (c) **pooled NET expectancy < 0** after fees + funding×hold-hours + realistic slippage,
  across those independent events, OR
- (d) the liquidation `bias_1h` shows **|Spearman| < 0.05** with forward 4h/24h return
  outside the June 1-3 crash window.

**Graduate paper → ¥10k real ONLY when ALL of:** ≥3 distinct regime cells (up/down/chop)
each with ≥8 independent episodes; ≥1 profitable LONG episode; pooled net expectancy >0
after full costs; backtested max DD <8% of equity at 0.5%-per-trade risk. Realistically
4–6 months, not 30 raw trades.

**No LightGBM** until ≥200 independent labeled signals across ≥2 regimes. Sooner = memorizing
one waterfall.

**The single biggest self-deception:** we grade every change against one crash. Every
proposed feature (bias, cost guard, sizer) was *favorable* in June 1-3, so the in-sample
backtest will always "improve." The one experiment that exposes it: **offline anchored
walk-forward over 12–18 months of 1h BTC with FROZEN params**, reporting per-regime-cell
and per-episode (not per-trade) net expectancy, and **whether the LONG branch fires at all**
in known uptrends. That is the next high-value task.
