# Monte Carlo Option Pricing

 A research-grade Monte Carlo option pricing library with Python and C++ backends.

## Setup

Create and activate a virtual environment, then install the package in editable mode:

bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip install pytest

# Run tests
pytest -q

## Features
	•	Geometric Brownian Motion (GBM) simulation under the risk-neutral measure
	•	European call and put payoffs
	•	Discounted Monte Carlo estimator with standard error and 95% confidence interval
	•	Validation against Black–Scholes closed-form pricing (pytest)
	•	American option pricing via Longstaff–Schwartz (LSM)
	•	Variance reduction utilities (control variates)
	•	Optional C++ acceleration via pybind11 and Eigen

## American Options (LSM)

American options are priced using the Longstaff–Schwartz (least-squares Monte Carlo)
algorithm. Correctness is validated against a Cox–Ross–Rubinstein (CRR) binomial tree
reference.

See notebooks/american_lsm_demo.ipynb for convergence plots versus the CRR price.

## C++ Acceleration (Optional)

This repository includes an optional C++ core (Eigen + pybind11) for the
Longstaff–Schwartz backward induction step. The Python and C++ implementations are
numerically consistent and validated by tests.

## Requirements (macOS)

Install build dependencies via Homebrew:

brew install cmake eigen pybind11

## Important: Python Architecture Must Match Build Architecture

The compiled extension must match your Python architecture
(e.g. x86_64 under Rosetta vs arm64 native).

Check your Python architecture:
python -c "import platform; print(platform.machine())"

## Build the C++ Module

From the repository root:
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=x86_64
cmake --build cpp/build -j

The compiled shared library is placed automatically into src/mcop/ so that:
import mcop._mcop_cpp

## Verify the Build
file src/mcop/_mcop_cpp*.so
python -c "import mcop._mcop_cpp; print('import ok')"
pytest -q tests/test_lsm_cpp_matches_python.py

## Using the C++ LSM Pricer

from mcop.american_lsm_cpp import american_option_lsm_cpp

price = american_option_lsm_cpp(
    paths,
    K=100.0,
    r=0.05,
    T=1.0,
    is_call=False,
    degree=3,
)

## Performance

On macOS (x86_64 Python under Rosetta), benchmarking against the pure-Python
implementation shows a significant speedup:

Python LSM: 6.069832 in 2.921s
C++    LSM: 6.069867 in 0.474s
Speedup: 6.16x

Benchmark script: benchmarks/bench_lsm_cpp_vs_py.py

## Notes
Benchmark outputs are written to artifacts/ (ignored by git)
If you switch to native arm64 Python, rebuild the extension with:
-DCMAKE_OSX_ARCHITECTURES=arm64