# Monte Carlo Option Pricing

A small, test-driven Monte Carlo option pricing library in Python.

## Setup

Create and activate a virtual environment, then install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip install pytest

# Run tests
pytest -q

## Features
- Geometric Brownian Motion (GBM) simulation under the risk-neutral measure
- European call and put payoffs
- Discounted Monte Carlo estimator with standard error and 95% confidence interval
- Validation against Black–Scholes closed-form pricing (pytest)
