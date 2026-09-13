"""
The instrument and position data model.

This is the layer that turns a pricer into something you can express a *trade*
in. A pricer answers "what is a 100-strike call worth?". A strategy needs to
say "short one 105 call, long one 110 call, both expiring 21 March, against 100
shares" — and then get a single value, a single delta, and a single payoff
diagram back for the whole thing.

Two design decisions worth defending
------------------------------------

**Expiry is a calendar date, not a time-to-maturity float.** The pricer takes T
in years, which is right for pricing but wrong for a position that persists. A
contract does not have a T; it has an expiry, and T is a function of the date
you are asking on. Storing T would mean mutating every contract every day, and
every "days to expiry" exit rule in the backtester would be reconstructing a
date from a float. Storing the expiry makes `time_to_expiry(asof)` a pure
function and makes the DTE rules honest.

**Quantities are signed, and multipliers live on the instrument.** A short leg
is quantity = -1, not a separate "side" field. This means every aggregation —
value, delta, vega, P&L — is a plain sum over legs with no branching on
direction, which is exactly where sign bugs breed. The contract multiplier
(100 shares per US equity option, 1 per share of stock) lives on the instrument
so that mixing stock and options in one position, as a covered call does, nets
correctly without the caller remembering to scale anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import numpy as np

__all__ = [
    "Right",
    "Style",
    "OptionContract",
    "Underlying",
    "Leg",
    "Position",
    "DAYS_PER_YEAR",
]

# Calendar-day year fraction. Options decay on weekends too — a Friday-to-Monday
# hold loses three days of theta, not one — so calendar days, not trading days,
# is the correct basis for time to expiry.
DAYS_PER_YEAR = 365.0


class Right(str, Enum):
    """Call or put. Inherits from str so it serialises and prints readably."""

    CALL = "call"
    PUT = "put"


class Style(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class OptionContract:
    """
    A single listed option contract.

    Frozen because a contract is an identity, not a state: the 105 call expiring
    March 21st is the same object whatever the market does. Anything that varies
    with the market (price, Greeks, IV) is computed against a `MarketState`
    rather than stored here.
    """

    underlying: str
    right: Right
    strike: float
    expiry: date
    style: Style = Style.AMERICAN
    multiplier: float = 100.0

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")

    @property
    def is_call(self) -> bool:
        return self.right is Right.CALL

    def time_to_expiry(self, asof: date, basis: float = DAYS_PER_YEAR) -> float:
        """
        Year fraction from `asof` to expiry, floored at zero.

        Returns 0.0 for an expired contract rather than a negative number, so
        that a pricer asked about a dead contract returns intrinsic value
        instead of producing NaNs from a negative variance.
        """
        return max((self.expiry - asof).days / basis, 0.0)

    def days_to_expiry(self, asof: date) -> int:
        return max((self.expiry - asof).days, 0)

    def intrinsic(self, spot: float) -> float:
        if self.is_call:
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)

    def moneyness(self, spot: float) -> float:
        """Strike / spot. Below 1 is a low strike, above 1 is a high strike."""
        return self.strike / spot

    def __str__(self) -> str:
        return f"{self.underlying} {self.expiry:%Y-%m-%d} {self.strike:g} {self.right.value}"


@dataclass(frozen=True)
class Underlying:
    """
    The underlying instrument itself, so stock legs sit in the same position as
    option legs (covered calls, collars, delta hedges).
    """

    symbol: str
    multiplier: float = 1.0

    def __str__(self) -> str:
        return f"{self.symbol} shares"


Instrument = OptionContract | Underlying


@dataclass(frozen=True)
class Leg:
    """
    One instrument held in signed size.

    quantity > 0 is long, < 0 is short. For options the unit is contracts (each
    controlling `multiplier` shares); for the underlying it is shares.
    """

    instrument: Instrument
    quantity: float

    def __post_init__(self) -> None:
        if self.quantity == 0:
            raise ValueError("quantity must be non-zero; drop the leg instead")

    @property
    def is_option(self) -> bool:
        return isinstance(self.instrument, OptionContract)

    @property
    def signed_shares(self) -> float:
        """Share-equivalent exposure: quantity scaled by the contract multiplier."""
        return self.quantity * self.instrument.multiplier

    def payoff_at_expiry(self, spot: float | np.ndarray) -> float | np.ndarray:
        """
        Terminal value of this leg at underlying price `spot`, in currency,
        including the multiplier and the sign of the position.

        This is value, not profit: what the leg is worth at expiry, before
        subtracting what was paid for it.
        """
        spot = np.asarray(spot, dtype=float)
        if isinstance(self.instrument, Underlying):
            value = spot
        else:
            if self.instrument.is_call:
                value = np.maximum(spot - self.instrument.strike, 0.0)
            else:
                value = np.maximum(self.instrument.strike - spot, 0.0)
        return self.signed_shares * value

    def __str__(self) -> str:
        side = "long" if self.quantity > 0 else "short"
        return f"{side} {abs(self.quantity):g} x {self.instrument}"


@dataclass
class Position:
    """
    A named collection of legs treated as one trade.

    Not frozen: a live position gets legs closed or rolled during a backtest.
    """

    legs: list[Leg]
    name: str = "position"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("a position needs at least one leg")

    @property
    def option_legs(self) -> list[Leg]:
        return [leg for leg in self.legs if leg.is_option]

    @property
    def underlying_legs(self) -> list[Leg]:
        return [leg for leg in self.legs if not leg.is_option]

    @property
    def expiries(self) -> list[date]:
        return sorted({leg.instrument.expiry for leg in self.option_legs})

    @property
    def is_single_expiry(self) -> bool:
        return len(self.expiries) <= 1

    def payoff_at_expiry(self, spot: float | np.ndarray) -> float | np.ndarray:
        """
        Terminal value of the whole position across a grid of underlying prices.

        Requires every option leg to share one expiry. A calendar spread has legs
        alive at different times, so "the payoff at expiry" is not a single
        curve — the far leg still carries time value when the near one dies.
        Rather than silently draw a misleading diagram, this raises, and
        `mcop.portfolio.value_curve` handles the general case by actually
        repricing the surviving legs.
        """
        if not self.is_single_expiry:
            raise ValueError(
                f"position '{self.name}' has legs expiring on {self.expiries}; "
                "a single expiry payoff diagram is not defined. Use "
                "mcop.portfolio.value_curve to mark the position on a given date instead."
            )
        spot = np.asarray(spot, dtype=float)
        total = np.zeros_like(spot)
        for leg in self.legs:
            total = total + leg.payoff_at_expiry(spot)
        return total

    def net_shares(self) -> float:
        """Share-equivalent size of the underlying legs only."""
        return sum(leg.signed_shares for leg in self.underlying_legs)

    def with_leg(self, leg: Leg) -> "Position":
        return Position(legs=[*self.legs, leg], name=self.name, meta=dict(self.meta))

    def __str__(self) -> str:
        body = "\n  ".join(str(leg) for leg in self.legs)
        return f"{self.name}:\n  {body}"
