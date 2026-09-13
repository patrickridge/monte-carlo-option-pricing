# The Strategy Layer

This document explains the layer built on top of the pricing engine: Greeks,
positions, strategies, market data, backtesting, and performance analysis. It is
written to be read start to finish by someone who knows some Python but not
necessarily much about options.

---

## 1. Why a pricer alone is not enough

The original library answers one question: *what is this option worth?*

That is necessary but not sufficient for doing anything with options. To manage
a position you need to know four more things:

1. **How the value moves** when the market moves — the Greeks.
2. **What the whole position is worth**, not just one contract — netting across
   legs.
3. **What structures are worth building** — strategies.
4. **Whether a rule for trading them actually made money** — backtesting.

Each numbered section below is one of those, in the order they were built,
because each depends on the one before it.

---

## 2. Greeks — how value responds to the world

A Greek is a partial derivative of the option price with respect to one input.

| Greek | Derivative | Plain English |
|-------|-----------|---------------|
| delta | dV/dS | If the stock rises £1, the option gains this much |
| gamma | d²V/dS² | How fast delta itself changes — the curvature |
| vega  | dV/dσ | Sensitivity to implied volatility |
| theta | dV/dt | Value lost per day just from time passing |
| rho   | dV/dr | Sensitivity to interest rates |

### Three ways to compute them, and why there are three

**Finite difference.** Reprice at `S+h` and `S-h`, divide by `2h`. Obvious — and
the obvious implementation is badly wrong.

If you re-draw the random numbers for each bump, each price carries independent
Monte Carlo noise. The difference then has variance roughly `2σ²_MC / (2h)²`.
Shrink `h` to reduce bias and **the noise explodes as 1/h²**. There is no good
choice of `h`.

The fix is **common random numbers**: use the *same* shock matrix for both
bumps. The two prices then move together, and since

```
Var(A − B) = Var(A) + Var(B) − 2·Cov(A, B)
```

the variance collapses as the covariance rises.

`benchmarks/bench_greeks.py` measures this, and the numbers show the predicted
scaling:

| spot bump | std dev, independent draws | std dev, common random numbers | improvement |
|---|---|---|---|
| 1.0% | 0.0309 | 0.00078 | **40×** |
| 0.5% | 0.0620 | 0.00083 | **75×** |
| 0.1% | 0.3104 | 0.00086 | **361×** |

Note what happens down the columns. The CRN standard deviation is essentially
flat as the bump shrinks; the naive one diverges like 1/h. That is the whole
argument in one table, and it is why `simulate_paths.py` was refactored to
expose `draw_normals()` separately from path construction — without shared
shocks the finite-difference Greeks would be unusable.

**Pathwise derivatives.** Differentiate the payoff along each path instead of
differencing prices. Unbiased and much tighter. For a call,
`dV/dS₀ = e^{-rT}·E[1{S_T>K}·S_T/S₀]`.

It cannot produce gamma. The second derivative of a vanilla payoff is a Dirac
delta at the strike, and no finite sample of paths lands exactly on it, so the
naive estimator returns exactly zero. The implementation returns `NaN` rather
than a plausible-looking zero — the method is telling you it does not apply, and
hiding that would be worse than failing.

**Likelihood ratio.** Differentiate the probability *density* rather than the
payoff, weighting each path by a score function. Handles discontinuous payoffs
(digitals, barriers) where pathwise breaks, and produces gamma in a single pass.
The cost is higher variance.

**Rule of thumb:** pathwise where it applies, likelihood ratio when the payoff
misbehaves or you need gamma cheaply, finite difference with common random
numbers whenever the pricer is a black box — which for American exercise is
always, because the LSM exercise policy has to be re-estimated under each bump.

All three are validated against the closed form in `tests/test_greeks_vs_bs.py`.

### Conventions (these cause real bugs)

The library returns **raw derivatives in natural units**: vega per 1.00 of vol
(not per vol point), theta per year (not per day), and theta signed as
`dV/dt` so a long option decays negative. Desk conventions are one
`.scaled()` call away. Mixing `dV/dT` and `dV/dt` — they differ by a minus sign
— is a classic source of sign errors, so the formulas compute `dV/dt` directly.

---

## 3. Positions — making legs net

A *leg* is one instrument in signed size. A *position* is a list of legs.

Two decisions do most of the work:

**Expiry is a calendar date, not a time-to-maturity float.** A contract does not
*have* a `T`; `T` is a function of the date you ask on. Storing `T` would mean
mutating every contract daily and reconstructing dates from floats for every
"days to expiry" rule. `OptionContract.time_to_expiry(asof)` is a pure function
instead, and floors at zero so an expired contract prices to intrinsic rather
than producing `NaN` from negative variance.

**Quantities are signed and multipliers live on the instrument.** A short leg is
`quantity = -1`, not a separate "side" flag. Every aggregation is then a plain
sum with no branching on direction:

```python
delta_total = sum(leg_greeks(leg, market).delta for leg in position.legs)
```

Because Greeks are additive, a short call automatically contributes negative
gamma and positive theta with no special-casing. This is where sign bugs
normally live, and the data model removes the opportunity.

### One deliberate refusal

`Position.payoff_at_expiry()` raises if the legs have different expiries. A
calendar spread has no single expiry — the near leg dies while the far leg still
holds time value — so a single-expiry diagram would be a lie. Use
`portfolio.value_curve()` for those, which actually reprices the survivors.

---

## 4. Which pricer marks which leg

- **European legs** → closed-form Black–Scholes.
- **American legs** → the **CRR binomial tree**, not Longstaff–Schwartz.

That second choice deserves defending, since LSM is the headline feature of the
library. A backtest reprices every leg of every open position on every trading
day — tens of thousands of valuations. Two problems with Monte Carlo there:

1. **Speed.** LSM at 50k paths is ~0.5s per valuation; the tree is ~1ms.
2. **Noise, which is the real problem.** MC marks carry sampling error, so
   marking an *unchanged* position on two consecutive days would produce
   different values purely from the random draw. That injects fake P&L
   volatility into the equity curve and corrupts every risk statistic computed
   from it — Sharpe most of all.

A deterministic pricer means a P&L change reflects a market change and nothing
else. The tree and LSM agree to well inside MC error, so this is a
speed/determinism swap, not an accuracy compromise —
`tests/test_portfolio.py::test_binomial_tree_mark_agrees_with_lsm` pins that.

---

## 5. Strategies — specifying trades the way traders do

Every builder in `strategies.py` returns a `Position`. Because netting already
works, a covered call is genuinely just "long 100 shares, short 1 call" and
inherits valuation, Greeks, payoff diagrams and P&L for free.

Available: covered call, protective put, collar, all four verticals, straddle,
strangle, iron condor, butterfly, calendar spread.

### Strikes are chosen by delta, not by price

Traders do not say "sell the 112.50 call"; they say "sell the 25-delta call".
Delta is roughly the risk-neutral probability of finishing in the money, so a
delta target expresses a **consistent amount of risk** across different spot
levels, volatilities and maturities. A fixed strike or a fixed percentage offset
does not: 5% out of the money is a near-certain expiry in a quiet market and a
coin-flip in a volatile one.

`strike_for_delta()` inverts the pricer by bisection. Bisection rather than
Newton because the volatility surface makes delta an awkward function of strike
— vol *itself* varies with strike under skew — so a derivative-based solve would
need the surface's slope. Bisection needs only monotonicity, which always holds.

---

## 6. Market data — and the honest caveat

Underlying price history is real and easy to get. **Historical option chains are
not.** A full history of quoted bid/ask across every strike and expiry is a paid
dataset (OptionMetrics, CBOE DataShop), and no free source covers it well enough
to backtest against.

So the library does this, and says so loudly rather than burying it:

> Option prices in the backtest are **modelled, not observed**. Each option is
> marked with Black–Scholes (or the tree for American exercise) using an implied
> volatility drawn from a parameterised surface anchored to the realised
> volatility of the actual underlying price history.

**What that buys:** the underlying path is real, so the directional behaviour of
a strategy — when it gets run over, when it profits — is driven by genuine
market moves.

**What it costs:** the surface is smooth. It cannot reproduce a volatility spike
that front-runs a crash, a bid/ask that gaps to untradeable, or skew steepening
exactly when you need to buy protection. A short-volatility strategy will look
*better* here than in reality, because the model never gaps.

**So results are strategy mechanics under a stated vol model, not a claim about
historical P&L.** That distinction is the difference between a defensible
project and a misleading one.

### The surface shape

```
iv(K, T) = atm(T) + skew·k + curvature·k²,     k = ln(K/S)
```

with two features of real equity surfaces deliberately reproduced:

- **Negative skew** — low-strike puts imply higher vol than high-strike calls.
  Crash risk is one-sided in equities and demand for downside protection is
  structural. Getting this sign right is what makes put spreads correctly
  expensive relative to call spreads.
- **Upward term structure** — longer-dated options carry more vol premium in
  calm regimes.

The `variance_risk_premium` parameter is the one concession to realism: implied
vol trades systematically above subsequent realised vol, and setting it to zero
would make every option structurally cheap.

---

## 7. Backtesting — and how backtests lie

The engine walks the price history forward one day at a time: re-anchor the vol
surface, mark open positions, apply exit rules, settle expiries, consider a new
entry. Cash and mark-to-market are tracked separately so the equity curve is
auditable (`equity == cash + exposure` is asserted in the tests).

Three things separate an honest backtest from a flattering one.

### No look-ahead

Every decision on date `i` uses only information available before the close of
`i`. Realised volatility at `i` is computed from returns up to `i-1`.
`test_rolling_vol_uses_only_past_returns` verifies this directly: it tampers
with *future* prices and asserts that earlier vol estimates do not move.

Dates without enough history to estimate vol simply do not trade. Guessing a
starting vol would be look-ahead in disguise.

### Transaction costs that reflect options

Option bid/ask spreads are wide — often 1–5% of premium on liquid names. A
four-legged iron condor crosses that spread **four times on entry and four more
on exit**. Backtests assuming mid-price fills turn losing premium-selling
strategies into winners; this is probably the single most common way an options
backtest deceives.

`CostModel` charges a half-spread on every leg in both directions plus
per-contract commission. The measured effect on the bundled example:

| | frictionless | retail costs |
|---|---|---|
| total P&L | **+16.91** | **−759.57** |
| profit factor | 1.01 | 0.63 |
| win rate | 54.3% | 48.8% |
| expectancy / trade | +0.37 | **−17.66** |

The strategy goes from roughly breakeven to reliably losing. **For multi-leg
option strategies the friction term is frequently larger than the edge.**

### Early assignment

Short American options can be assigned before expiry. The engine flags a short
leg once its *extrinsic* value (mid minus intrinsic) collapses below a
threshold — the rational-exercise condition — because at that point the holder
gives up nothing by exercising. Ignoring this flatters any strategy that lets
losers run.

### The strongest correctness test

If the underlying drifts at `r`, options are marked at exactly the vol the paths
were generated with, and there is no variance risk premium and no costs, then
selling options is a **zero-expectancy game**. Any consistent profit or loss
would reveal a bias in the accounting, the settlement logic or the pricer.

Measured across independent seeds, mean total P&L came out **−285 with a
standard error of 220** — statistically indistinguishable from zero (t ≈ −1.3).
That is the check that makes every other number in the backtest trustworthy, and
it is pinned in `test_fair_priced_options_have_no_systematic_edge`.

### Simplifications, stated

- Option prices are modelled, not observed (§6). Dominant caveat.
- Assignment closes the whole position at that day's marks. Cash-settling an
  assigned leg at intrinsic and liquidating the resulting stock is
  P&L-equivalent to physical delivery, but the follow-on position an iron condor
  would leave is not modelled.
- **No margin model.** Sizing is by contract count, so the equity curve does not
  reflect the buying power a short-premium strategy actually consumes.
- Fills are always available at the modelled price. Real illiquidity means
  sometimes there is no fill at all — particularly in the stress scenarios where
  an exit matters most.

---

## 8. Performance statistics — and how to read them

`performance.py` reports returns (total, CAGR), risk (annualised vol, Sharpe,
Sortino, max drawdown, Calmar, VaR, CVaR), distribution shape (skew, excess
kurtosis, worst day) and trade statistics (win rate, profit factor, expectancy,
average hold).

Two warnings are built into the module docstring because they matter more than
the numbers:

**There is no margin model,** so every return is computed against
`initial_capital`, an arbitrary denominator. Selling one iron condor against
£100k of idle cash produces a tiny return series with a flattering Sharpe,
because the denominator dampens volatility without dampening P&L. Return-based
metrics are comparable *between runs in this framework* but are not claims about
deployed capital. **Trade-level statistics do not have this problem and are the
more honest read.**

**Sharpe assumes roughly i.i.d. normal returns.** Option selling violates both:
a long series of small gains punctuated by rare large losses is exactly the
shape Sharpe flatters most. A high Sharpe on a short-premium strategy should be
read as *"has not met its tail yet"*, not as evidence of quality — which is why
skew, excess kurtosis and worst-day are reported alongside it, and why CVaR is
computed empirically rather than from a normal approximation.

---

## 9. Try it

```bash
# Greeks three ways, against the closed form
mcop greeks --S0 100 --K 100 --sigma 0.2 --T 1 --call

# Build a delta-targeted iron condor and see its risk and payoff
mcop strategy iron_condor --spot 100 --delta 0.20 --wing-width 5 --dte 45

# Backtest it, with and without costs
mcop backtest iron_condor --no-greeks
mcop backtest iron_condor --commission 0 --spread 0 --no-greeks

# The measured results behind the tables above
python benchmarks/bench_greeks.py
python benchmarks/bench_backtest.py
```
