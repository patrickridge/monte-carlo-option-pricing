# src/mcop/cli.py

import argparse
from datetime import date, timedelta

from mcop.simulate_paths import simulate_gbm_paths
from mcop.american_lsm import american_option_lsm


def cmd_price(args: argparse.Namespace) -> None:
    """
    Price an American option using LSM (Python or C++).
    Simulates GBM paths then runs LSM backward induction.
    """
    paths = simulate_gbm_paths(
        S0=args.S0,
        r=args.r,
        sigma=args.sigma,
        T=args.T,
        n_steps=args.n_steps,
        n_paths=args.n_paths,
        q=args.q,
        seed=args.seed,
        antithetic=True,
    )

    if args.engine == "cpp":
        # Lazy import so Python engine works even if the C++ extension isn't built
        try:
            from mcop.american_lsm_cpp import american_option_lsm_cpp
        except Exception as e:
            raise SystemExit(
                "C++ engine requested but C++ extension is not available.\n"
                "Build it first (see README), or run with --engine python."
            ) from e

        price = american_option_lsm_cpp(
            paths,
            K=args.K,
            r=args.r,
            T=args.T,
            is_call=args.call,
            degree=args.degree,
        )
    else:
        price = american_option_lsm(
            paths,
            K=args.K,
            r=args.r,
            T=args.T,
            is_call=args.call,
            degree=args.degree,
            q=args.q,
        )

    opt_type = "call" if args.call else "put"
    print(
        f"American {opt_type} price (LSM, {args.engine}): {price:.6f} "
        f"[S0={args.S0}, K={args.K}, T={args.T}, r={args.r}, q={args.q}, sigma={args.sigma}, "
        f"steps={args.n_steps}, paths={args.n_paths}, degree={args.degree}]"
    )


def cmd_greeks(args: argparse.Namespace) -> None:
    """Price and risk sensitivities for a single option, analytic and by Monte Carlo."""
    from mcop.black_scholes import bs_greeks
    from mcop.greeks import mc_greeks

    opt_type = "call" if args.call else "put"
    print(
        f"{args.style} {opt_type}  S0={args.S0} K={args.K} T={args.T} "
        f"r={args.r} q={args.q} sigma={args.sigma}"
    )
    print(f"{'source':>22} {'price':>10} {'delta':>9} {'gamma':>9} {'vega':>9} {'theta':>9} {'rho':>9}")
    print("-" * 82)

    def show(label, g):
        s = g.scaled()  # vega per vol point, theta per day
        print(
            f"{label:>22} {g.price:>10.4f} {g.delta:>9.4f} {g.gamma:>9.5f} "
            f"{s.vega:>9.4f} {s.theta:>9.4f} {s.rho:>9.4f}"
        )

    if args.style == "european":
        show("Black-Scholes", bs_greeks(args.S0, args.K, args.r, args.q, args.sigma, args.T, args.call))

    methods = ["fd"] if args.style == "american" else ["fd", "pathwise", "lr"]
    for method in methods:
        g = mc_greeks(
            args.S0, args.K, args.r, args.sigma, args.T, args.call, q=args.q,
            n_paths=args.n_paths, n_steps=args.n_steps, seed=args.seed,
            style=args.style, method=method,
        )
        show(f"Monte Carlo ({method})", g)

    print("\nvega quoted per 1 vol point, theta per calendar day, rho per 1% rate move")


def cmd_strategy(args: argparse.Namespace) -> None:
    """Build a named strategy and report its value, Greeks, breakevens and payoff."""
    import numpy as np

    from mcop.market_data import MarketState, SkewedVolSurface
    from mcop.portfolio import breakevens, position_value, report
    from mcop import strategies as st

    asof = date.today() if args.asof is None else date.fromisoformat(args.asof)
    expiry = asof + timedelta(days=args.dte)

    market = MarketState(
        asof=asof, spot=args.spot, r=args.r, q=args.q,
        vol_surface=SkewedVolSurface(atm_level=args.sigma),
    )

    builders = {
        "iron_condor": lambda: st.iron_condor_by_delta(
            market, args.symbol, expiry, short_delta=args.delta, wing_width=args.wing_width
        ),
        "short_strangle": lambda: st.short_strangle_by_delta(
            market, args.symbol, expiry, target_delta=args.delta
        ),
        "covered_call": lambda: st.covered_call_by_delta(
            market, args.symbol, expiry, target_delta=args.delta
        ),
        "straddle": lambda: st.straddle(args.symbol, args.spot, expiry),
        "butterfly": lambda: st.butterfly(
            args.symbol, "call", args.spot - args.wing_width, args.spot, args.spot + args.wing_width, expiry
        ),
    }
    if args.name not in builders:
        raise SystemExit(f"unknown strategy {args.name!r}; choose from {sorted(builders)}")

    position = builders[args.name]()
    print(position)
    print()
    print(report(position, market))

    entry = position_value(position, market)
    print(f"\n  entry cost : {entry:>12,.2f}  ({'debit paid' if entry > 0 else 'credit received'})")

    if position.is_single_expiry:
        try:
            be = breakevens(position, entry_cost=entry)
            print(f"  breakevens : {', '.join(f'{b:.2f}' for b in be) if be else 'none'}")
        except ValueError:
            pass

        print(f"\n  payoff at expiry ({expiry}):")
        spots = np.linspace(0.8 * args.spot, 1.2 * args.spot, 9)
        payoff = position.payoff_at_expiry(spots) - entry
        for s, p in zip(spots, payoff):
            bar = "#" * int(max(0, min(40, 20 + p / max(1.0, abs(payoff).max()) * 20)))
            print(f"    {s:>8.2f}  {p:>10,.2f}  {bar}")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Walk-forward backtest of an option strategy."""
    from mcop.backtest import BacktestConfig, CostModel, run_backtest
    from mcop.market_data import load_price_csv, synthetic_price_history
    from mcop.performance import summarize

    if args.csv:
        history = load_price_csv(args.csv, symbol=args.symbol)
    else:
        history = synthetic_price_history(
            args.symbol, S0=args.spot, mu=args.mu, sigma=args.sigma,
            n_days=args.n_days, seed=args.seed, vol_of_vol=args.vol_of_vol,
        )

    cost = CostModel(
        commission_per_contract=args.commission,
        spread_frac_of_price=args.spread,
    )
    config = BacktestConfig(
        strategy=args.name,
        initial_capital=args.capital,
        entry_dte=args.entry_dte,
        exit_dte=args.exit_dte,
        short_delta=args.delta,
        wing_width=args.wing_width,
        contracts=args.contracts,
        profit_target=args.profit_target,
        stop_loss=args.stop_loss,
        r=args.r,
        q=args.q,
        variance_risk_premium=args.vrp,
        cost=cost,
        track_greeks=not args.no_greeks,
    )

    result = run_backtest(history, config, verbose=args.verbose)
    print()
    print(summarize(result))

    if args.trades:
        print("\nTRADE LOG")
        print(f"{'opened':>12} {'closed':>12} {'reason':>14} {'entry':>10} {'P&L':>10}")
        print("-" * 62)
        for t in result.closed_trades:
            print(
                f"{t.open_date.isoformat():>12} {t.close_date.isoformat():>12} "
                f"{t.reason:>14} {t.entry_cash:>10,.2f} {t.pnl:>10,.2f}"
            )

    print(
        "\nNote: option prices are modelled from a parameterised vol surface anchored to "
        "trailing\nrealised volatility, not observed quotes. See the mcop.market_data "
        "docstring for what\nthat does and does not let you conclude."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcop",
        description="Monte Carlo option pricing, Greeks, and strategy analytics",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- price command ----
    p_price = sub.add_parser(
        "price",
        help="Price an American option using LSM",
    )

    p_price.add_argument("--S0", type=float, default=100.0, metavar="PRICE",
                         help="Current stock price (default: 100)")
    p_price.add_argument("--K", type=float, default=100.0, metavar="STRIKE",
                         help="Strike price (default: 100)")
    p_price.add_argument("--r", type=float, default=0.05, metavar="RATE",
                         help="Annualised risk-free rate, e.g. 0.05 for 5%% (default: 0.05)")
    p_price.add_argument("--q", type=float, default=0.0, metavar="YIELD",
                         help="Continuous dividend yield (default: 0)")
    p_price.add_argument("--sigma", type=float, default=0.2, metavar="VOL",
                         help="Annualised volatility, e.g. 0.2 for 20%% (default: 0.2)")
    p_price.add_argument("--T", type=float, default=1.0, metavar="YEARS",
                         help="Time to expiry in years, e.g. 0.25 for 3 months (default: 1.0)")

    p_price.add_argument("--n-steps", dest="n_steps", type=int, default=100, metavar="N",
                         help="Number of time steps per path (default: 100)")
    p_price.add_argument("--n-paths", dest="n_paths", type=int, default=50_000, metavar="N",
                         help="Number of Monte Carlo paths (default: 50000)")
    p_price.add_argument("--degree", type=int, default=2, metavar="D",
                         help="Polynomial degree for LSM regression basis (default: 2)")
    p_price.add_argument("--seed", type=int, default=123, metavar="SEED",
                         help="RNG seed for reproducibility (default: 123)")

    p_price.add_argument(
        "--call",
        action="store_true",
        help="Price a call (default: put)",
    )

    p_price.add_argument(
        "--engine",
        choices=["python", "cpp"],
        default="python",
        help="Pricing engine to use",
    )

    p_price.set_defaults(func=cmd_price)

    # ---- greeks command ----
    p_greeks = sub.add_parser(
        "greeks",
        help="Price and Greeks for one option (analytic and Monte Carlo)",
    )
    p_greeks.add_argument("--S0", type=float, default=100.0, metavar="PRICE")
    p_greeks.add_argument("--K", type=float, default=100.0, metavar="STRIKE")
    p_greeks.add_argument("--r", type=float, default=0.05, metavar="RATE")
    p_greeks.add_argument("--q", type=float, default=0.0, metavar="YIELD")
    p_greeks.add_argument("--sigma", type=float, default=0.2, metavar="VOL")
    p_greeks.add_argument("--T", type=float, default=1.0, metavar="YEARS")
    p_greeks.add_argument("--n-steps", dest="n_steps", type=int, default=50, metavar="N")
    p_greeks.add_argument("--n-paths", dest="n_paths", type=int, default=100_000, metavar="N")
    p_greeks.add_argument("--seed", type=int, default=123, metavar="SEED")
    p_greeks.add_argument("--call", action="store_true", help="Call (default: put)")
    p_greeks.add_argument("--style", choices=["european", "american"], default="european",
                          help="Exercise style (American supports only the FD estimator)")
    p_greeks.set_defaults(func=cmd_greeks)

    # ---- strategy command ----
    p_strat = sub.add_parser(
        "strategy",
        help="Build a strategy and show its value, Greeks, breakevens and payoff",
    )
    p_strat.add_argument("name", choices=["iron_condor", "short_strangle", "covered_call",
                                          "straddle", "butterfly"])
    p_strat.add_argument("--symbol", default="XYZ")
    p_strat.add_argument("--spot", type=float, default=100.0)
    p_strat.add_argument("--sigma", type=float, default=0.22, metavar="VOL",
                         help="ATM implied volatility for the surface (default: 0.22)")
    p_strat.add_argument("--r", type=float, default=0.04)
    p_strat.add_argument("--q", type=float, default=0.0)
    p_strat.add_argument("--dte", type=int, default=45, help="Days to expiry (default: 45)")
    p_strat.add_argument("--delta", type=float, default=0.20,
                         help="Target delta for short strikes (default: 0.20)")
    p_strat.add_argument("--wing-width", dest="wing_width", type=float, default=5.0)
    p_strat.add_argument("--asof", default=None, help="Valuation date YYYY-MM-DD (default: today)")
    p_strat.set_defaults(func=cmd_strategy)

    # ---- backtest command ----
    p_bt = sub.add_parser(
        "backtest",
        help="Walk-forward backtest of an option strategy",
    )
    p_bt.add_argument("name", choices=["iron_condor", "short_strangle", "short_put", "covered_call"])
    p_bt.add_argument("--csv", default=None,
                      help="Daily close CSV for the underlying (default: synthetic history)")
    p_bt.add_argument("--symbol", default="SYNTH")
    p_bt.add_argument("--capital", type=float, default=100_000.0)
    p_bt.add_argument("--entry-dte", dest="entry_dte", type=int, default=45)
    p_bt.add_argument("--exit-dte", dest="exit_dte", type=int, default=21)
    p_bt.add_argument("--delta", type=float, default=0.20)
    p_bt.add_argument("--wing-width", dest="wing_width", type=float, default=5.0)
    p_bt.add_argument("--contracts", type=float, default=1.0)
    p_bt.add_argument("--profit-target", dest="profit_target", type=float, default=0.50)
    p_bt.add_argument("--stop-loss", dest="stop_loss", type=float, default=2.0)
    p_bt.add_argument("--commission", type=float, default=0.65, metavar="PER_CONTRACT")
    p_bt.add_argument("--spread", type=float, default=0.02, metavar="FRAC",
                      help="Bid/ask width as a fraction of option price (default: 0.02)")
    p_bt.add_argument("--vrp", type=float, default=0.15,
                      help="Variance risk premium: implied = realised * (1 + vrp)")
    p_bt.add_argument("--r", type=float, default=0.04)
    p_bt.add_argument("--q", type=float, default=0.0)
    # synthetic history controls
    p_bt.add_argument("--spot", type=float, default=100.0)
    p_bt.add_argument("--mu", type=float, default=0.08)
    p_bt.add_argument("--sigma", type=float, default=0.20)
    p_bt.add_argument("--n-days", dest="n_days", type=int, default=756)
    p_bt.add_argument("--vol-of-vol", dest="vol_of_vol", type=float, default=0.6)
    p_bt.add_argument("--seed", type=int, default=42)
    p_bt.add_argument("--trades", action="store_true", help="Print the full trade log")
    p_bt.add_argument("--no-greeks", dest="no_greeks", action="store_true",
                      help="Skip daily Greek tracking (much faster)")
    p_bt.add_argument("--verbose", action="store_true")
    p_bt.set_defaults(func=cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
