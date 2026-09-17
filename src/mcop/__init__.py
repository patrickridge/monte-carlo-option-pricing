"""
mcop — Monte Carlo option pricing, Greeks, and strategy analytics.

The package is layered, and each layer only depends on the ones above it:

    simulate_paths / payoffs / pricing      core Monte Carlo engine
    american_lsm / binomial_tree            early-exercise pricers
    black_scholes                           closed-form reference and fast marks
    greeks                                  MC risk sensitivities
    instruments                             contracts, legs, positions
    market_data                             price history and the vol surface
    portfolio                               valuation and risk netting
    strategies                              named multi-leg structures
    backtest                                walk-forward simulation
    performance                             return and risk statistics
"""

from .simulate_paths import simulate_gbm_paths, draw_normals, gbm_paths_from_normals
from .payoffs import european_call, european_put
from .pricing import mc_price
from .american_lsm import american_option_lsm
from .binomial_tree import american_option_crr
from .variance_reduction import control_variate_adjustment, mc_mean_se
from .analysis import convergence_study, plot_convergence

from .black_scholes import Greeks, bs_price, bs_greeks, implied_vol
from .greeks import mc_greeks

from .exotics import (
    asian_payoff,
    asian_price_with_control,
    barrier_payoff,
    barrier_survival_probability,
    digital_payoff,
    lookback_payoff,
)
from .exotic_analytic import (
    digital_asset_or_nothing,
    digital_cash_or_nothing,
    down_and_in_call,
    down_and_out_call,
    geometric_asian_price,
)

from .instruments import (
    DAYS_PER_YEAR,
    Leg,
    OptionContract,
    Position,
    Right,
    Style,
    Underlying,
)
from .market_data import (
    FlatVolSurface,
    MarketState,
    PriceHistory,
    SkewedVolSurface,
    VolSurface,
    ewma_vol,
    fetch_price_history,
    load_price_csv,
    realized_vol,
    rolling_realized_vol,
    synthetic_price_history,
)
from .portfolio import (
    breakevens,
    leg_greeks,
    leg_value,
    position_greeks,
    position_value,
    profit_curve,
    report,
    value_curve,
)
from .strategies import (
    butterfly,
    calendar_spread,
    collar,
    covered_call,
    covered_call_by_delta,
    iron_condor,
    iron_condor_by_delta,
    protective_put,
    short_strangle_by_delta,
    straddle,
    strangle,
    strike_for_delta,
    vertical_spread,
)
from .backtest import (
    BacktestConfig,
    BacktestResult,
    CostModel,
    Trade,
    monthly_expiries,
    run_backtest,
    third_friday,
)
from .performance import (
    PerformanceReport,
    historical_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize,
)

__all__ = [
    # core Monte Carlo
    "simulate_gbm_paths",
    "draw_normals",
    "gbm_paths_from_normals",
    "european_call",
    "european_put",
    "mc_price",
    "american_option_lsm",
    "american_option_crr",
    "control_variate_adjustment",
    "mc_mean_se",
    "convergence_study",
    "plot_convergence",
    # pricing and Greeks
    "Greeks",
    "bs_price",
    "bs_greeks",
    "implied_vol",
    "mc_greeks",
    # exotics
    "asian_payoff",
    "asian_price_with_control",
    "barrier_payoff",
    "barrier_survival_probability",
    "digital_payoff",
    "lookback_payoff",
    "digital_asset_or_nothing",
    "digital_cash_or_nothing",
    "down_and_in_call",
    "down_and_out_call",
    "geometric_asian_price",
    # instruments
    "DAYS_PER_YEAR",
    "Leg",
    "OptionContract",
    "Position",
    "Right",
    "Style",
    "Underlying",
    # market data
    "FlatVolSurface",
    "MarketState",
    "PriceHistory",
    "SkewedVolSurface",
    "VolSurface",
    "ewma_vol",
    "fetch_price_history",
    "load_price_csv",
    "realized_vol",
    "rolling_realized_vol",
    "synthetic_price_history",
    # portfolio
    "breakevens",
    "leg_greeks",
    "leg_value",
    "position_greeks",
    "position_value",
    "profit_curve",
    "report",
    "value_curve",
    # strategies
    "butterfly",
    "calendar_spread",
    "collar",
    "covered_call",
    "covered_call_by_delta",
    "iron_condor",
    "iron_condor_by_delta",
    "protective_put",
    "short_strangle_by_delta",
    "straddle",
    "strangle",
    "strike_for_delta",
    "vertical_spread",
    # backtest
    "BacktestConfig",
    "BacktestResult",
    "CostModel",
    "Trade",
    "monthly_expiries",
    "run_backtest",
    "third_friday",
    # performance
    "PerformanceReport",
    "historical_var",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "summarize",
]
