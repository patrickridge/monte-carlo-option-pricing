# Roadmap

This project is intentionally modular and can be extended in several directions.

## Done

- **Greeks** — finite difference under common random numbers, pathwise, and
  likelihood ratio estimators, validated against closed-form Black–Scholes
- **Multi-leg positions** — signed legs, contract multipliers, additive Greeks
- **Strategy templates** — covered call, protective put, collar, verticals,
  straddle, strangle, iron condor, butterfly, calendar spread
- **Delta-targeted strike selection** solved off the implied vol surface
- **Walk-forward backtesting** with transaction costs, early assignment,
  listed monthly expiries, and no look-ahead
- **Performance analytics** — Sharpe, Sortino, drawdown, Calmar, VaR/CVaR,
  trade statistics

See [STRATEGIES.md](STRATEGIES.md).

## Next: real option chain data

The single biggest limitation of the backtest is that option prices are
modelled from a parameterised vol surface rather than observed. Replacing that
with real historical chains (OptionMetrics, CBOE DataShop, or a broker's
historical API) would let the backtest capture what the model cannot:
volatility spikes that front-run drawdowns, skew steepening under stress, and
spreads that widen exactly when an exit is needed.

The `VolSurface` interface exists to make this a drop-in change: implement
`iv(strike, spot, T)` against real quotes and every downstream layer works
unchanged.

## Margin and capital efficiency

Position sizing is currently by contract count, so returns are computed against
an arbitrary capital base. A span-style margin model would make returns
comparable to other strategies and let the backtest enforce realistic position
limits.

## Stochastic Volatility (Heston)

- Implement Heston dynamics using correlated Brownian motions
- Add full truncation or QE schemes for variance process
- Compare Monte Carlo prices to semi-closed Heston formulas
- Extend LSM to American options under stochastic volatility

Would also make the synthetic price generator more realistic: the current
`vol_of_vol` parameter is a crude AR(1) proxy for a real variance process.

## Variance Reduction

- Antithetic variates for all path simulations
- Control variates using Black–Scholes prices
- Stratified or Latin hypercube sampling

## Performance

- Parallel path generation (NumPy / OpenMP)
- SIMD-friendly regression basis in C++
- GPU acceleration (CuPy or CUDA kernels)
- Vectorised binomial trees for faster backtest marks

## CLI Extensions

- `mcop heston price`
- `mcop bench lsm`
- `mcop plot convergence`
- JSON / CSV output for batch runs
