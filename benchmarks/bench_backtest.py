"""
Benchmark: what transaction costs do to an option-selling strategy.

This is the headline result of the strategy layer, and it is a negative one.
A short-premium programme that looks flat-to-slightly-profitable at mid-price
fills turns clearly unprofitable once realistic bid/ask crossing and commission
are charged on every leg. An iron condor pays that cost eight times per round
trip — four legs in, four legs out.

The lesson generalises: for multi-leg option strategies, the friction term is
not a rounding error on the edge, it is frequently larger than the edge.

Run:  python benchmarks/bench_backtest.py
"""

from __future__ import annotations

from mcop.backtest import BacktestConfig, CostModel, run_backtest
from pathlib import Path

from mcop.market_data import load_price_csv, synthetic_price_history
from mcop.performance import summarize

def _history():
    """
    Real SPY closes where available, simulated otherwise.

    The distinction matters more than it looks. A short-premium strategy is
    structurally short fat tails and volatility clustering, and geometric
    Brownian motion produces neither: over 2015-2024 SPY showed excess kurtosis
    of 13.6 against 0.6 for a simulated series of the same volatility. Measuring
    this strategy on simulated paths flatters it.
    """
    csv = Path(__file__).resolve().parents[1] / "data" / "spy_daily.csv"
    if csv.exists():
        return load_price_csv(csv, symbol="SPY")
    return synthetic_price_history(
        "SYNTH", S0=100.0, mu=0.08, sigma=0.20, n_days=756, seed=42, vol_of_vol=0.6
    )


HISTORY = _history()

FRICTIONLESS = CostModel(commission_per_contract=0.0, spread_frac_of_price=0.0, min_spread=0.0)


def cost_sensitivity() -> None:
    """Sweep the assumed bid/ask width and watch the edge disappear."""
    print("\nCOST SENSITIVITY — iron condor, 20-delta shorts, 5-wide wings")
    print(f"{'spread':>10} {'commission':>11} {'total P&L':>11} {'win rate':>9} "
          f"{'profit factor':>14} {'expectancy':>11}")
    print("-" * 70)

    scenarios = [
        ("0.0%", 0.00, 0.00),
        ("1.0%", 0.01, 0.65),
        ("2.0%", 0.02, 0.65),
        ("5.0%", 0.05, 0.65),
    ]
    for label, spread, comm in scenarios:
        cost = CostModel(commission_per_contract=comm, spread_frac_of_price=spread,
                         min_spread=0.01 if spread else 0.0)
        rpt = summarize(run_backtest(HISTORY, BacktestConfig(cost=cost, track_greeks=False)))
        print(f"{label:>10} {f'${comm:.2f}':>11} {rpt.total_pnl:>11,.2f} "
              f"{rpt.win_rate:>9.1%} {rpt.profit_factor:>14.2f} {rpt.expectancy:>11,.2f}")

    print(
        "\n  Note: P&L is not monotone in the spread, and that is expected. The profit\n"
        "  target is 50% *of the credit received*, so a wider spread shrinks the credit,\n"
        "  which lowers the absolute profit target, which makes it easier to hit. Trades\n"
        "  then close earlier and avoid some losers. Every individual fill is strictly\n"
        "  worse; the trade sequence is simply no longer the same one."
    )


def strategy_comparison() -> None:
    """
    The same rules and costs across the four built-in structures.

    Read this table carefully: it is NOT a clean like-for-like comparison. The
    condor and strangle are close to delta-neutral, so their P&L is mostly about
    premium and friction. Short puts and covered calls carry real directional
    exposure, and over a sample where the underlying rose, a large part of their
    profit is simply long-stock P&L. Leg count and friction explain the gap
    between the condor and the strangle; direction explains most of the rest.
    """
    print("\nSTRATEGY COMPARISON — retail costs, 45 DTE entry, 21 DTE exit")
    print(f"{'strategy':>16} {'trades':>7} {'total P&L':>11} {'win rate':>9} "
          f"{'max DD':>8} {'Sharpe':>8}")
    print("-" * 64)

    for strategy in ("iron_condor", "short_strangle", "short_put", "covered_call"):
        cfg = BacktestConfig(strategy=strategy, cost=CostModel(), track_greeks=False)
        rpt = summarize(run_backtest(HISTORY, cfg))
        print(f"{strategy:>16} {rpt.n_trades:>7d} {rpt.total_pnl:>11,.2f} "
              f"{rpt.win_rate:>9.1%} {rpt.max_drawdown:>8.2%} {rpt.sharpe:>8.2f}")

    print(
        f"\n  {HISTORY.symbol} moved {HISTORY.closes[0]:.2f} -> {HISTORY.closes[-1]:.2f} "
        f"({HISTORY.closes[-1] / HISTORY.closes[0] - 1:+.1%}) over this sample, which is\n"
        "  most of why the directional structures look good - they are partly just long\n"
        "  the market. Compare the condor and the strangle to each other for the friction\n"
        "  effect, since both are close to delta-neutral."
    )


def exit_rule_sweep() -> None:
    """
    Profit target and stop loss are not free parameters — they trade win rate
    against average loss, and the product is roughly conserved.
    """
    print("\nEXIT RULE SWEEP — short strangle, frictionless (isolating rule effects)")
    print(f"{'target':>8} {'stop':>8} {'trades':>7} {'win rate':>9} "
          f"{'avg win':>9} {'avg loss':>10} {'total P&L':>11}")
    print("-" * 66)

    for target, stop in [(0.25, 2.0), (0.50, 2.0), (0.75, 2.0), (None, None), (0.50, 1.0)]:
        cfg = BacktestConfig(
            strategy="short_strangle", profit_target=target, stop_loss=stop,
            cost=FRICTIONLESS, track_greeks=False,
        )
        rpt = summarize(run_backtest(HISTORY, cfg))
        t_txt = f"{target:.0%}" if target is not None else "none"
        s_txt = f"{stop:.0%}" if stop is not None else "none"
        print(f"{t_txt:>8} {s_txt:>8} {rpt.n_trades:>7d} {rpt.win_rate:>9.1%} "
              f"{rpt.avg_win:>9,.2f} {rpt.avg_loss:>10,.2f} {rpt.total_pnl:>11,.2f}")


def headline() -> None:
    """The single comparison worth quoting."""
    free = summarize(run_backtest(HISTORY, BacktestConfig(cost=FRICTIONLESS, track_greeks=False)))
    real = summarize(run_backtest(HISTORY, BacktestConfig(cost=CostModel(), track_greeks=False)))

    print("\n" + "=" * 70)
    print("HEADLINE: the cost of crossing the spread eight times per round trip")
    print("=" * 70)
    print(f"  frictionless : total P&L {free.total_pnl:>10,.2f}   "
          f"profit factor {free.profit_factor:>5.2f}   expectancy {free.expectancy:>8,.2f}")
    print(f"  retail costs : total P&L {real.total_pnl:>10,.2f}   "
          f"profit factor {real.profit_factor:>5.2f}   expectancy {real.expectancy:>8,.2f}")
    swing = free.total_pnl - real.total_pnl
    print(f"\n  frictions cost {swing:,.2f} over {real.n_trades} round trips "
          f"({swing / max(real.n_trades, 1):,.2f} per trade),")
    print("  which is what turns a marginal edge into a reliable loss.")


if __name__ == "__main__":
    print(f"underlying: {HISTORY.symbol}  {HISTORY.dates[0]} to {HISTORY.dates[-1]}  "
          f"({len(HISTORY)} days)  {HISTORY.closes[0]:.2f} -> {HISTORY.closes[-1]:.2f}")
    cost_sensitivity()
    strategy_comparison()
    exit_rule_sweep()
    headline()
