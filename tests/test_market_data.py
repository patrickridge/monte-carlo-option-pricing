"""Price history loading, volatility estimation, and the implied vol surface."""

import math
from datetime import date

import numpy as np
import pytest

from mcop.instruments import OptionContract, Right, Style
from mcop.market_data import (
    FlatVolSurface,
    MarketState,
    PriceHistory,
    SkewedVolSurface,
    ewma_vol,
    load_price_csv,
    realized_vol,
    rolling_realized_vol,
    synthetic_price_history,
)


# --- PriceHistory validation ----------------------------------------------

def test_price_history_rejects_bad_input():
    dates = [date(2024, 1, 1), date(2024, 1, 2)]
    with pytest.raises(ValueError, match="equal length"):
        PriceHistory("X", dates, np.array([100.0]))
    with pytest.raises(ValueError, match="at least two"):
        PriceHistory("X", dates[:1], np.array([100.0]))
    with pytest.raises(ValueError, match="positive"):
        PriceHistory("X", dates, np.array([100.0, -5.0]))
    with pytest.raises(ValueError, match="sorted ascending"):
        PriceHistory("X", [dates[1], dates[0]], np.array([100.0, 101.0]))


def test_log_returns_length_and_value():
    hist = PriceHistory("X", [date(2024, 1, i) for i in (1, 2, 3)],
                        np.array([100.0, 110.0, 121.0]))
    rets = hist.log_returns
    assert len(rets) == 2
    assert rets[0] == pytest.approx(math.log(1.1))


def test_lookup_and_slicing():
    hist = synthetic_price_history("X", n_days=60, seed=1)
    mid = hist.dates[30]
    assert hist.close_on(mid) == pytest.approx(hist.closes[30])
    sub = hist.slice(start=hist.dates[10], end=hist.dates[20])
    assert len(sub) == 11
    assert sub.dates[0] == hist.dates[10]


def test_index_of_raises_past_the_end():
    hist = synthetic_price_history("X", n_days=20, seed=1)
    with pytest.raises(KeyError):
        hist.index_of(date(2099, 1, 1))


# --- CSV loading -----------------------------------------------------------

def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_load_price_csv_basic(tmp_path):
    path = _write(tmp_path, "px.csv", "date,close\n2024-01-02,100.5\n2024-01-03,101.25\n2024-01-04,99.0\n")
    hist = load_price_csv(path)
    assert hist.symbol == "px"
    assert len(hist) == 3
    assert hist.dates[0] == date(2024, 1, 2)
    assert hist.closes[1] == pytest.approx(101.25)


def test_load_price_csv_prefers_adjusted_close(tmp_path):
    """Adjusted close avoids splits and dividends showing up as fake gaps."""
    path = _write(
        tmp_path, "adj.csv",
        "Date,Close,Adj Close\n2024-01-02,200.0,100.0\n2024-01-03,202.0,101.0\n",
    )
    hist = load_price_csv(path, symbol="AAPL")
    assert hist.symbol == "AAPL"
    np.testing.assert_allclose(hist.closes, [100.0, 101.0])


def test_load_price_csv_sorts_and_skips_blank_rows(tmp_path):
    path = _write(
        tmp_path, "messy.csv",
        "date,close\n2024-01-04,99.0\n,\n2024-01-02,100.0\n2024-01-03,\n",
    )
    hist = load_price_csv(path)
    assert len(hist) == 2
    assert hist.dates == [date(2024, 1, 2), date(2024, 1, 4)]


def test_load_price_csv_reports_unusable_headers(tmp_path):
    path = _write(tmp_path, "bad.csv", "when,price\n2024-01-02,100.0\n2024-01-03,101.0\n")
    with pytest.raises(ValueError, match="could not find a date column"):
        load_price_csv(path)


def test_load_price_csv_handles_timestamps(tmp_path):
    path = _write(
        tmp_path, "ts.csv",
        "timestamp,close\n2024-01-02 00:00:00,100.0\n2024-01-03 00:00:00,101.0\n",
    )
    assert len(load_price_csv(path)) == 2


# --- synthetic histories ---------------------------------------------------

def test_synthetic_history_is_reproducible():
    a = synthetic_price_history("S", n_days=100, seed=5)
    b = synthetic_price_history("S", n_days=100, seed=5)
    c = synthetic_price_history("S", n_days=100, seed=6)
    np.testing.assert_array_equal(a.closes, b.closes)
    assert not np.array_equal(a.closes, c.closes)


def test_synthetic_history_skips_weekends():
    hist = synthetic_price_history("S", n_days=40, seed=1)
    assert all(d.weekday() < 5 for d in hist.dates)


def test_synthetic_history_realises_roughly_the_requested_vol():
    hist = synthetic_price_history("S", sigma=0.25, n_days=3000, seed=2)
    assert realized_vol(hist.log_returns) == pytest.approx(0.25, abs=0.02)


def test_vol_of_vol_produces_time_varying_volatility():
    steady = synthetic_price_history("S", n_days=800, seed=3, vol_of_vol=0.0)
    wobbly = synthetic_price_history("S", n_days=800, seed=3, vol_of_vol=0.8)
    spread = lambda h: np.nanstd(rolling_realized_vol(h, 21))  # noqa: E731
    assert spread(wobbly) > spread(steady)


# --- volatility estimators -------------------------------------------------

def test_realized_vol_annualises_by_sqrt_252():
    daily = np.full(500, 0.01)
    assert realized_vol(daily) == pytest.approx(0.01 * math.sqrt(252))
    assert realized_vol(daily, annualise=False) == pytest.approx(0.01)


def test_realized_vol_of_empty_input_is_nan():
    assert math.isnan(realized_vol(np.array([])))
    assert math.isnan(ewma_vol(np.array([])))


def test_ewma_reacts_faster_than_equal_weighting():
    """A recent shock must move EWMA more than the flat-window estimate."""
    calm = np.full(200, 0.002)
    shocked = np.concatenate([calm, np.full(5, 0.05)])
    assert ewma_vol(shocked) > realized_vol(shocked)


def test_rolling_vol_is_nan_before_the_window_fills():
    hist = synthetic_price_history("S", n_days=100, seed=1)
    rv = rolling_realized_vol(hist, window=21)
    assert np.all(np.isnan(rv[:21]))
    assert np.all(np.isfinite(rv[21:]))


# --- vol surface -----------------------------------------------------------

def test_flat_surface_is_constant():
    s = FlatVolSurface(0.18)
    assert s.iv(80, 100, 0.5) == s.iv(120, 100, 2.0) == 0.18


def test_skewed_surface_has_negative_skew():
    """Low strikes must imply higher vol than high strikes for equities."""
    s = SkewedVolSurface(atm_level=0.20)
    vols = [s.iv(k, 100.0, 0.25) for k in (80, 90, 100, 110, 120)]
    assert vols[0] > vols[1] > vols[2]
    assert vols[2] > vols[3] > vols[4]


def test_skewed_surface_term_structure_slopes_up():
    s = SkewedVolSurface(atm_level=0.20, term_slope=0.05)
    assert s.iv(100, 100, 1.0) > s.iv(100, 100, 0.1)


def test_surface_respects_floor_and_cap():
    s = SkewedVolSurface(atm_level=0.20, skew=-5.0, curvature=0.0, floor=0.05, cap=0.60)
    assert s.iv(500.0, 100.0, 0.5) == pytest.approx(0.05)
    assert s.iv(10.0, 100.0, 0.5) == pytest.approx(0.60)


def test_surface_rejects_nonpositive_inputs():
    s = SkewedVolSurface()
    with pytest.raises(ValueError, match="positive"):
        s.iv(100.0, 0.0, 0.5)


def test_with_atm_reanchors_level_but_keeps_shape():
    base = SkewedVolSurface(atm_level=0.20, skew=-0.4, curvature=0.3)
    moved = base.with_atm(0.35)
    assert moved.atm_level == 0.35
    assert moved.skew == base.skew and moved.curvature == base.curvature
    # The whole smile shifts up by the same amount it was re-anchored.
    shift = moved.iv(90, 100, 0.5) - base.iv(90, 100, 0.5)
    assert shift == pytest.approx(0.15, abs=1e-12)


def test_market_state_prices_a_contract_off_the_surface():
    market = MarketState(asof=date(2024, 1, 15), spot=100.0, r=0.04,
                         vol_surface=SkewedVolSurface(atm_level=0.22))
    put = OptionContract("X", Right.PUT, 90.0, date(2024, 3, 15), Style.EUROPEAN)
    call = OptionContract("X", Right.CALL, 110.0, date(2024, 3, 15), Style.EUROPEAN)
    assert market.iv_for(put) > market.iv_for(call)


def test_market_state_copies_are_independent():
    market = MarketState(asof=date(2024, 1, 15), spot=100.0, r=0.04)
    assert market.with_spot(120.0).spot == 120.0
    assert market.spot == 100.0
    assert market.with_date(date(2024, 2, 1)).asof == date(2024, 2, 1)
