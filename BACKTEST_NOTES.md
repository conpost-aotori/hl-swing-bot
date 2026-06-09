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
