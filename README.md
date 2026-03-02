# Monte Carlo Option Pricing

A Monte Carlo option pricing library with pure-Python and optional C++ backends.

Supports European and American options under Geometric Brownian Motion (GBM), including:

- Discounted Monte Carlo estimator with 95% confidence intervals
- American options via the **Longstaff–Schwartz (LSM)** algorithm
- Optional **C++ acceleration** (pybind11 + Eigen) — ~6× speedup for the LSM backward pass
- Validation against **Black–Scholes** (European) and **Cox–Ross–Rubinstein binomial tree** (American)
- Variance reduction via **antithetic variates** and **control variates**
- CLI for quick pricing from the command line

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
payoffs = european_call(paths[:, -1], K=100)
price, se, ci_low, ci_high = mc_price(payoffs, r=0.05, T=1.0)
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
| CLI | `mcop price` with full parameter control |
| Notebooks | Demo and convergence plots in `notebooks/` |
| Benchmarks | LSM accuracy vs CRR binomial, Python vs C++ timing |

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
│   ├── simulate_paths.py   # GBM path simulation (exact discretization)
│   ├── payoffs.py          # European call/put payoff functions
│   ├── pricing.py          # Discounted MC estimator with CI
│   ├── american_lsm.py     # Longstaff–Schwartz (Python)
│   ├── american_lsm_cpp.py # Longstaff–Schwartz (C++ wrapper)
│   ├── binomial_tree.py    # CRR binomial tree (reference pricer)
│   ├── variance_reduction.py # Control variate utilities
│   ├── analysis.py         # Convergence study helpers
│   └── cli.py              # Command-line interface
├── cpp/                    # C++ source (pybind11 + Eigen)
├── tests/                  # pytest test suite
├── benchmarks/             # Accuracy and performance benchmarks
├── notebooks/              # Jupyter demo notebooks
└── docs/                   # Documentation and roadmap
```

---

## License

MIT — see [LICENSE](LICENSE).
