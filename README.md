# Monte Carlo Option Pricing

A Monte Carlo option pricing library with pure-Python and optional C++ backends.

Supports European and American options under Geometric Brownian Motion (GBM), including:

- Discounted Monte Carlo estimator with 95% confidence intervals
- American options via the **Longstaff–Schwartz (LSM)** algorithm
- Optional **C++ acceleration** (pybind11 + Eigen) — ~6× speedup for the LSM backward pass
- Validation against **Black–Scholes** (European) and **Cox–Ross–Rubinstein binomial tree** (American)
- Variance reduction via **antithetic variates** and **control variates**
- CLI for quick pricing from the command line

On top of the pricing engine sits a **strategy and risk layer**:

- **Greeks** by three estimators — finite difference under common random
  numbers, pathwise, and likelihood ratio — validated against the closed form
- **Multi-leg positions** with netted valuation and risk (covered calls,
  verticals, straddles, strangles, iron condors, butterflies, calendars)
- **Delta-targeted strike selection**, the way trades are actually specified
- **Walk-forward backtesting** with realistic transaction costs, early
  assignment, and no look-ahead
- **Performance analytics** — Sharpe, Sortino, drawdown, VaR/CVaR, trade stats

See **[docs/STRATEGIES.md](docs/STRATEGIES.md)** for a full walkthrough of that
layer, including an explicit account of what the backtest can and cannot tell
you.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the package in editable mode along with `pytest`.

---

## Quickstart

```bash
# Price an American put under GBM using LSM (Python engine)
mcop price --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1 --n-paths 50000

# Price a call instead of a put
mcop price --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1 --call

# Use the C++ engine (requires building the extension first — see below)
mcop price --engine cpp --n-paths 50000

# Greeks three ways, checked against the closed form
mcop greeks --S0 100 --K 100 --sigma 0.2 --T 1 --call

# Build a delta-targeted iron condor: value, netted Greeks, breakevens, payoff
mcop strategy iron_condor --spot 100 --delta 0.20 --wing-width 5 --dte 45

# Backtest it, with and without transaction costs
mcop backtest iron_condor --no-greeks
mcop backtest iron_condor --commission 0 --spread 0 --no-greeks

# Run tests
pytest -q
```

---

## Python API

```python
from mcop import simulate_gbm_paths, american_option_lsm, mc_price, european_call

# Simulate GBM paths
paths = simulate_gbm_paths(S0=100, r=0.05, sigma=0.2, T=1.0, n_steps=100, n_paths=50_000)

# Price an American put with LSM
price = american_option_lsm(paths, K=100, r=0.05, T=1.0, is_call=False)

# Price a European call with MC + confidence interval
payoffs = european_call(paths, K=100)
price, se, ci_low, ci_high = mc_price(payoffs, r=0.05, T=1.0)
```

### Greeks

```python
from mcop import bs_greeks, mc_greeks

# Closed form
analytic = bs_greeks(S=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, is_call=True)

# Monte Carlo, three estimators. "fd" uses common random numbers across bumps,
# which is what keeps a finite-difference Greek from drowning in its own noise.
for method in ("fd", "pathwise", "lr"):
    g = mc_greeks(100, 100, 0.05, 0.2, 1.0, is_call=True, method=method)
    print(method, g.delta, g.vega)

# American exercise (finite difference on LSM)
amer = mc_greeks(100, 100, 0.05, 0.2, 1.0, is_call=False, style="american")

# Raw derivatives by default; .scaled() gives vega per vol point, theta per day
print(analytic.scaled().theta)
```

### Positions and strategies

```python
from datetime import date
from mcop import MarketState, SkewedVolSurface, iron_condor_by_delta, report, breakevens

market = MarketState(asof=date(2024, 1, 15), spot=100.0, r=0.04,
                     vol_surface=SkewedVolSurface(atm_level=0.22))

condor = iron_condor_by_delta(market, "XYZ", date(2024, 3, 15),
                              short_delta=0.20, wing_width=5.0)

print(report(condor, market))        # netted value, delta, gamma, vega, theta, rho
print(breakevens(condor, entry_cost=-131.30))
```

### Backtesting

```python
from mcop import BacktestConfig, CostModel, run_backtest, summarize, synthetic_price_history

history = synthetic_price_history("SYNTH", S0=100, sigma=0.20, n_days=756, seed=42)

config = BacktestConfig(strategy="iron_condor", short_delta=0.20, cost=CostModel())
result = run_backtest(history, config)
print(summarize(result))
```

---

## Pricing Real Options

You can price actual market options by supplying real parameters. For example, to price an Apple (AAPL) American put:

```bash
# AAPL trading at $227, 3-month put with $220 strike
# Implied vol ~28%, risk-free rate ~4.3%, no dividend
mcop price --S0 227 --K 220 --r 0.043 --sigma 0.28 --T 0.25 --n-paths 100000
```

### Where to get each parameter

| Parameter | Description | Where to get it |
|-----------|-------------|-----------------|
| `--S0` | Current stock price | Yahoo Finance, Bloomberg, broker |
| `--K` | Strike price | Options chain (your broker or Yahoo Finance → Options tab) |
| `--r` | Risk-free rate (annualised) | 3-month US Treasury yield (e.g. [FRED](https://fred.stlouisfed.org/series/DTB3)) |
| `--sigma` | Volatility (annualised) | Implied vol from the options chain, or compute historical vol from price history |
| `--T` | Time to expiry in years | `days_to_expiry / 365` (e.g. 30 days → `0.082`) |
| `--q` | Continuous dividend yield | Annual dividend / stock price (default `0`) |
| `--n-paths` | Monte Carlo paths | More paths = more accurate. 50k–200k is typical |
| `--n-steps` | Time steps | 100 is sufficient for most options; use ~252 to match daily trading days |

> **Tip:** For European options, prices will closely match Black–Scholes. For American puts with time value and dividends, the LSM result will differ from Black–Scholes — that's expected and correct.

---

## Features

| Feature | Notes |
|---------|-------|
| GBM simulation | Exact discretization, dividend yield `q`, antithetic variates |
| European options | Call and put payoffs with discounted MC estimator |
| American options | Longstaff–Schwartz LSM, configurable polynomial basis degree |
| C++ acceleration | pybind11 + Eigen, ~6× faster LSM backward pass |
| Control variates | Variance reduction using a correlated control with known mean |
| Black–Scholes | Closed-form price, Greeks, and a robust bisection IV solver |
| Greeks | Finite difference (common random numbers), pathwise, likelihood ratio |
| Positions | Signed multi-leg legs with netted value and additive Greeks |
| Strategies | Covered call, collar, verticals, straddle, strangle, condor, butterfly, calendar |
| Strike selection | Delta-targeted strikes solved off the vol surface |
| Vol surface | Parameterised equity skew and term structure, anchored to realised vol |
| Backtesting | Walk-forward, transaction costs, early assignment, no look-ahead |
| Performance | Sharpe, Sortino, drawdown, Calmar, VaR/CVaR, full trade statistics |
| CLI | `mcop price`, `mcop greeks`, `mcop strategy`, `mcop backtest` |
| Notebooks | Demo and convergence plots in `notebooks/` |
| Benchmarks | LSM vs CRR, Python vs C++ timing, Greek estimators, cost sensitivity |

---

## Selected Results

### Common random numbers make finite-difference Greeks usable

Re-drawing the random numbers for each bump leaves two independent Monte Carlo
errors in the difference quotient, and dividing by a small `2h` amplifies them —
the noise diverges like `1/h`. Sharing one shock matrix across both bumps
removes almost all of it.

Standard deviation of the delta estimate across 25 repetitions (50k paths):

| spot bump | independent draws | common random numbers | improvement |
|-----------|------------------|----------------------|-------------|
| 1.0%      | 0.0309           | 0.00078              | **40×**     |
| 0.5%      | 0.0620           | 0.00083              | **75×**     |
| 0.1%      | 0.3104           | 0.00086              | **361×**    |

The CRN column is flat as the bump shrinks; the naive one blows up. Reproduce
with `python benchmarks/bench_greeks.py`.

### Transaction costs dominate multi-leg option strategies

An iron condor crosses the bid/ask **four times on entry and four more on
exit**. Backtesting the same rules with and without that friction:

| | frictionless | retail costs (2% spread, $0.65/contract) |
|---|---|---|
| total P&L | **+16.91** | **−759.57** |
| profit factor | 1.01 | 0.63 |
| expectancy per trade | +0.37 | **−17.66** |

The strategy goes from breakeven to reliably losing — friction costs about
**18 per round trip** against an edge that was never worth more than a few
pounds a trade.

Comparing structures under identical rules, the four-legged iron condor
(−760) does worst and the two-legged short strangle (+81) does better, which is
what you would expect when friction scales with leg count. The single-leg
results are *not* a clean continuation of that pattern, though: short puts and
covered calls both carry directional exposure, and the underlying rose 20% over
this sample, so their profits are substantially long-stock P&L rather than
evidence about costs. Reproduce with `python benchmarks/bench_backtest.py`.

> **Read these as strategy mechanics under a stated volatility model, not as
> historical P&L.** Option prices in the backtest are modelled from a
> parameterised vol surface anchored to realised volatility, because historical
> option chains are not freely available. See
> [docs/STRATEGIES.md §6](docs/STRATEGIES.md) for exactly what that does and
> does not permit you to conclude.

### The engine is unbiased

If the underlying drifts at `r`, options are marked at exactly the volatility
the paths were generated with, and there are no costs or variance risk premium,
then selling options must be a **zero-expectancy game**. Measured across
independent seeds, mean total P&L was **−285 with a standard error of 220**
(t ≈ −1.3) — statistically indistinguishable from zero. That check is what makes
every other backtest number trustworthy, and it runs in the test suite.

---

## C++ Acceleration (Optional, macOS)

The C++ backend accelerates the LSM backward induction step using Eigen for linear algebra.

### Requirements

```bash
brew install cmake eigen pybind11
```

> **Note:** The compiled extension must match your Python architecture.
> Check with: `python -c "import platform; print(platform.machine())"`

### Build

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build cpp/build -j
```

Replace `arm64` with `x86_64` if you are running x86_64 Python (e.g. under Rosetta).

The compiled `.so` is placed automatically into `src/mcop/` so that `import mcop._mcop_cpp` works.

### Verify

```bash
file src/mcop/_mcop_cpp*.so
python -c "import mcop._mcop_cpp; print('import ok')"
pytest -q tests/test_lsm_cpp_matches_python.py
```

### Performance

On macOS (x86_64 Python under Rosetta), benchmarking 200k paths × 100 steps:

```
Python LSM : 6.0698  in 2.921s
C++    LSM : 6.0699  in 0.474s
Speedup    : 6.16×
```

Benchmark script: `benchmarks/bench_lsm_cpp_vs_py.py`

---

## Project Structure

```
monte-carlo-option-pricing/
├── src/mcop/               # Python package
│   │                       # -- pricing engine --
│   ├── simulate_paths.py   # GBM paths; shock draws exposed for common random numbers
│   ├── payoffs.py          # European call/put payoff functions
│   ├── pricing.py          # Discounted MC estimator with CI
│   ├── american_lsm.py     # Longstaff–Schwartz (Python)
│   ├── american_lsm_cpp.py # Longstaff–Schwartz (C++ wrapper)
│   ├── binomial_tree.py    # CRR binomial tree (reference + fast American marks)
│   ├── variance_reduction.py # Control variate utilities
│   ├── analysis.py         # Convergence study helpers
│   │                       # -- risk and strategy layer --
│   ├── black_scholes.py    # Closed-form price, Greeks, implied vol solver
│   ├── greeks.py           # MC Greeks: finite difference / pathwise / likelihood ratio
│   ├── instruments.py      # Contracts, signed legs, multi-leg positions
│   ├── market_data.py      # Price history, realised vol, implied vol surface
│   ├── portfolio.py        # Valuation and risk netting across legs
│   ├── strategies.py       # Named structures + delta-targeted strike selection
│   ├── backtest.py         # Walk-forward engine, costs, assignment
│   ├── performance.py      # Return and risk statistics
│   └── cli.py              # Command-line interface
├── cpp/                    # C++ source (pybind11 + Eigen)
├── tests/                  # pytest test suite
├── benchmarks/             # Accuracy and performance benchmarks
├── notebooks/              # Jupyter demo notebooks
└── docs/
    ├── STRATEGIES.md       # Full walkthrough of the strategy layer
    └── ROADMAP.md          # Possible extensions
```

---

## License

MIT — see [LICENSE](LICENSE).
