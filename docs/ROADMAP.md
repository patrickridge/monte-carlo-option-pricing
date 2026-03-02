# Roadmap

This project is intentionally modular and can be extended in several directions.

## Stochastic Volatility (Heston)

- Implement Heston dynamics using correlated Brownian motions
- Add full truncation or QE schemes for variance process
- Compare Monte Carlo prices to semi-closed Heston formulas
- Extend LSM to American options under stochastic volatility

## Variance Reduction

- Antithetic variates for all path simulations
- Control variates using Black–Scholes prices
- Stratified or Latin hypercube sampling

## Performance

- Parallel path generation (NumPy / OpenMP)
- SIMD-friendly regression basis in C++
- GPU acceleration (CuPy or CUDA kernels)

## CLI Extensions

- `mcop heston price`
- `mcop bench lsm`
- `mcop plot convergence`
- JSON / CSV output for batch runs
