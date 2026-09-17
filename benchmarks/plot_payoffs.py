"""
Generate the payoff diagrams and backtest figures used in the documentation.

A payoff diagram is the single most useful picture in options: it plots profit
and loss at expiry against the underlying price, so the shape of a structure —
where it makes money, where it stops, what it costs — is visible at a glance
rather than inferred from four leg definitions.

Writes to docs/figures/. Run:  python benchmarks/plot_payoffs.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display in CI or a plain shell
import matplotlib.pyplot as plt
import numpy as np

from mcop.backtest import BacktestConfig, CostModel, run_backtest
from mcop.instruments import Right, Style
from mcop.market_data import MarketState, SkewedVolSurface, synthetic_price_history
from mcop.portfolio import position_value, value_curve
from mcop.performance import summarize
from mcop import strategies as st

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ASOF = date(2024, 1, 15)
EXPIRY = date(2024, 3, 15)
FAR_EXPIRY = date(2024, 6, 21)
SPOT = 100.0

INK = "#171A1F"
MUTED = "#5C6470"
PROFIT = "#2F6149"
LOSS = "#9E3B3B"
ACCENT = "#34567F"


def market() -> MarketState:
    return MarketState(asof=ASOF, spot=SPOT, r=0.04, q=0.0,
                       vol_surface=SkewedVolSurface(atm_level=0.22))


def _panel(ax, title, blurb, spots, pnl, strikes, entry):
    """One payoff diagram: P&L against underlying price at expiry."""
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)

    ax.plot(spots, pnl, color=INK, lw=1.8, zorder=4)
    ax.fill_between(spots, 0, pnl, where=(pnl >= 0), color=PROFIT, alpha=0.18, zorder=2)
    ax.fill_between(spots, 0, pnl, where=(pnl < 0), color=LOSS, alpha=0.18, zorder=2)

    for k in strikes:
        ax.axvline(k, color=ACCENT, lw=0.7, ls=(0, (3, 3)), alpha=0.65, zorder=3)

    ax.axvline(SPOT, color=MUTED, lw=0.7, ls=":", alpha=0.8, zorder=3)

    kind = "credit" if entry < 0 else "debit"
    ax.set_title(f"{title}\n{blurb}", fontsize=8.5, color=INK, loc="left", pad=6)
    ax.text(0.98, 0.04, f"{kind} {abs(entry):,.0f}", transform=ax.transAxes,
            fontsize=7, color=MUTED, ha="right", va="bottom")

    ax.tick_params(labelsize=7, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#DCDFE5")
    ax.margins(x=0)


def payoff_grid() -> None:
    """Nine structures, one panel each."""
    mkt = market()
    spots = np.linspace(70, 130, 601)

    def strikes_of(pos):
        return sorted({leg.instrument.strike for leg in pos.option_legs})

    specs = [
        (st.covered_call("XYZ", 100, 107, EXPIRY),
         "Covered call", "Long stock, short a call. Caps upside, collects premium."),
        (st.protective_put("XYZ", 100, 93, EXPIRY),
         "Protective put", "Long stock plus a put. Insurance with a deductible."),
        (st.collar("XYZ", 100, 93, 107, EXPIRY),
         "Collar", "Put bought, call sold to pay for it. Floor and ceiling."),
        (st.vertical_spread("XYZ", Right.CALL, 98, 108, EXPIRY),
         "Bull call spread", "Long low strike, short high. Capped bullish bet."),
        (st.straddle("XYZ", 100, EXPIRY),
         "Long straddle", "Call and put, same strike. Bets on a big move either way."),
        (st.strangle("XYZ", 93, 107, EXPIRY, quantity=-1),
         "Short strangle", "Sells an OTM put and call. Profits if nothing happens."),
        (st.iron_condor("XYZ", 88, 93, 107, 112, EXPIRY),
         "Iron condor", "Short strangle with wings. Same idea, capped loss."),
        (st.butterfly("XYZ", Right.CALL, 93, 100, 107, EXPIRY),
         "Long butterfly", "Bets the price pins near the middle strike."),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 10.5))
    fig.patch.set_facecolor("white")

    for ax, (pos, title, blurb) in zip(axes.flat, specs):
        entry = position_value(pos, mkt)
        pnl = np.asarray(pos.payoff_at_expiry(spots), dtype=float) - entry
        _panel(ax, title, blurb, spots, pnl, strikes_of(pos), entry)

    # Calendar spread needs different treatment: its legs expire on different
    # dates, so "the payoff at expiry" is not defined. Mark it on the near
    # expiry instead, when the short leg dies and the long leg still has value.
    cal = st.calendar_spread("XYZ", Right.CALL, 100, EXPIRY, FAR_EXPIRY)
    entry = position_value(cal, mkt)
    near = MarketState(asof=EXPIRY, spot=SPOT, r=mkt.r, q=mkt.q, vol_surface=mkt.vol_surface)
    pnl = value_curve(cal, near, spots) - entry
    _panel(axes.flat[8], "Calendar spread",
           "Short near-dated, long far-dated. Valued on the near expiry,\n"
           "because the two legs never expire together.",
           spots, pnl, strikes_of(cal), entry)

    fig.suptitle("Option strategy payoffs at expiry  ·  spot 100, ~60 days, 22% implied vol",
                 fontsize=11, color=INK, y=0.995, x=0.012, ha="left")
    fig.text(0.012, 0.005,
             "Solid line is profit and loss. Blue dashed lines mark strikes; dotted line is the "
             "starting spot. Green is profit, red is loss.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.018, 1, 0.975))

    path = OUT / "strategy_payoffs.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def backtest_figure() -> None:
    """Equity curves with and without transaction costs, plus the trade P&L spread."""
    history = synthetic_price_history("SYNTH", S0=100.0, mu=0.08, sigma=0.20,
                                      n_days=756, seed=42, vol_of_vol=0.6)
    free = CostModel(commission_per_contract=0.0, spread_frac_of_price=0.0, min_spread=0.0)

    runs = {}
    for label, cost in (("no costs", free), ("retail costs", CostModel())):
        cfg = BacktestConfig(strategy="iron_condor", cost=cost, track_greeks=False)
        runs[label] = run_backtest(history, cfg)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5),
                                   gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("white")

    # --- equity curves -----------------------------------------------------
    for label, colour in (("no costs", ACCENT), ("retail costs", LOSS)):
        res = runs[label]
        rpt = summarize(res)
        ax1.plot(res.dates, res.equity, lw=1.4, color=colour,
                 label=f"{label}  (total P&L {rpt.total_pnl:,.0f})")

    ax1.axhline(100_000, color=MUTED, lw=0.8, ls=":")
    ax1.set_title("Iron condor, 20-delta shorts, 5-wide wings — the cost of crossing the "
                  "spread eight times per round trip",
                  fontsize=10, color=INK, loc="left")
    ax1.set_ylabel("account equity", fontsize=8.5, color=MUTED)
    ax1.legend(fontsize=8, frameon=False)

    # --- per-trade P&L -----------------------------------------------------
    pnls = [t.pnl for t in runs["retail costs"].closed_trades]
    colours = [PROFIT if p >= 0 else LOSS for p in pnls]
    ax2.bar(range(len(pnls)), pnls, color=colours, alpha=0.75, width=0.8)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.axhline(float(np.mean(pnls)), color=INK, lw=1.0, ls="--",
                label=f"mean {np.mean(pnls):,.2f} per trade")
    ax2.set_title("Per-trade P&L with realistic costs — many small wins, fewer larger losses",
                  fontsize=9.5, color=INK, loc="left")
    ax2.set_xlabel("trade number", fontsize=8.5, color=MUTED)
    ax2.legend(fontsize=8, frameon=False)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7.5, colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#DCDFE5")

    fig.tight_layout()
    path = OUT / "backtest_costs.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    payoff_grid()
    backtest_figure()
