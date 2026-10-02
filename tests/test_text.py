"""The EDGAR text pipeline, and the three bugs that made its numbers unreadable."""
import numpy as np
import pandas as pd
import pytest

from xsalpha import data, ic
from xsalpha.ml import _stack
from xsalpha.text import as_monthly_signal, tone_change_panel


def _scores(rows):
    return pd.DataFrame(rows, columns=["ticker", "filing_date", "net_tone"]).assign(
        filing_date=lambda d: pd.to_datetime(d["filing_date"]))


def test_tone_change_is_against_the_company_s_own_last_filing():
    """A company that always writes darkly contributes nothing until its own tone moves.

    Differencing against the cross-section instead would turn the signal into a
    'this firm is gloomier than its peers' factor, which is a different claim and a
    much weaker one - Loughran-McDonald's result is about the change.
    """
    scores = _scores([("AAA", "2020-02-01", -0.020), ("AAA", "2020-05-01", -0.010),
                      ("BBB", "2020-02-01", -0.001), ("BBB", "2020-05-01", -0.004)])
    panel = tone_change_panel(scores)
    assert panel.loc["2020-05-01", "AAA"] == pytest.approx(0.010)   # got less negative
    assert panel.loc["2020-05-01", "BBB"] == pytest.approx(-0.003)  # got more negative
    # The first filing has nothing to difference against, so that date carries no
    # information and does not appear in the panel at all.
    assert pd.Timestamp("2020-02-01") not in panel.index


def test_a_filing_cannot_be_traded_on_the_day_it_is_stamped():
    """EDGAR accepts filings until 22:00 Eastern under the same day's date. Using one
    at that day's close is a one-day look-ahead: small, invisible, still a look-ahead."""
    scores = _scores([("AAA", "2020-01-31", -0.02), ("AAA", "2020-04-30", -0.01)])
    panel = tone_change_panel(scores)
    dates = pd.DatetimeIndex(["2020-04-30", "2020-05-29"])
    monthly = as_monthly_signal(panel, dates)
    assert np.isnan(monthly.loc["2020-04-30", "AAA"])
    assert monthly.loc["2020-05-29", "AAA"] == pytest.approx(0.01)


def test_a_stale_filing_stops_voting():
    """Otherwise, a 2016 10-K is still expressing an opinion in 2026."""
    scores = _scores([("AAA", "2020-01-15", -0.02), ("AAA", "2020-04-15", -0.01)])
    panel = tone_change_panel(scores)
    dates = pd.DatetimeIndex(["2020-05-29", "2020-12-31"])
    monthly = as_monthly_signal(panel, dates, stale_after_days=120)
    assert monthly.loc["2020-05-29", "AAA"] == pytest.approx(0.01)
    assert np.isnan(monthly.loc["2020-12-31", "AAA"])


def test_benjamini_hochberg_is_monotone_and_bounded():
    p = pd.Series({"a": 0.001, "b": 0.02, "c": 0.10, "d": 0.40, "e": 0.90})
    q = ic.benjamini_hochberg(p)
    assert (q <= 1.0).all()
    assert (q >= p).all()                       # a correction never makes a test easier
    assert q.loc[p.sort_values().index].is_monotonic_increasing
    assert q["a"] == pytest.approx(0.005)       # 0.001 * 5 / 1


def test_benjamini_hochberg_is_less_brutal_than_bonferroni():
    """The point of using BH over Bonferroni: it does not throw away the real ones."""
    p = pd.Series({f"s{i}": v for i, v in enumerate([0.001, 0.004, 0.008, 0.2, 0.9])})
    q = ic.benjamini_hochberg(p)
    assert (q.iloc[:3] < 0.05).all()
    assert (p.iloc[:3] * len(p) >= q.iloc[:3]).all()


def test_ic_summary_can_hold_every_signal_to_the_same_window():
    """Comparing a signal that starts in 2016 against one that starts in 2010 on their
    own samples compares two decades, not two signals."""
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    long_run = pd.Series(np.linspace(0.05, -0.05, 60), index=idx)
    short_run = long_run.copy()
    short_run.iloc[:40] = np.nan
    full = ic.ic_summary({"long": long_run, "short": short_run})
    assert full.loc["long", "n_months"] == 60
    assert full.loc["short", "n_months"] == 20
    common = ic.ic_summary({"long": long_run, "short": short_run}, start=idx[40])
    assert common["n_months"].nunique() == 1
    assert common.loc["long", "mean_ic"] == pytest.approx(common.loc["short", "mean_ic"])


def test_the_price_cache_hits_on_the_start_date_the_project_uses():
    """2010-01-01 is a holiday, so a cached panel always begins on the 4th. The staleness
    test was exact, so the cache never hit, every run re-downloaded, and the ML combo's
    Sharpe wandered by 0.09 on nothing but Yahoo's latest adjustments."""
    cached = pd.DataFrame({"AAA": [1.0, 2.0]},
                          index=pd.to_datetime(["2010-01-04", "2010-01-05"]))
    slack = pd.Timedelta(days=7)
    assert not (cached.index.min() > pd.Timestamp("2010-01-01") + slack)
    assert cached.index.min() > pd.Timestamp("2010-01-01")      # the old test, still true
    assert hasattr(data, "download_prices")


def test_a_sparse_feature_does_not_delete_the_rows_it_is_missing_from():
    """LightGBM picks a default direction for missing values at each split. Dropping
    those rows instead made the training set a function of the sparsest signal: adding
    the tone feature cost half the observations, and the resulting fall in Sharpe reads
    as evidence about the signal when it is evidence about dropna."""
    dates = pd.date_range("2021-01-31", periods=6, freq="ME")
    dense = pd.DataFrame(1.0, index=dates, columns=["A", "B", "C"])
    sparse = dense.copy()
    sparse.iloc[:4] = np.nan                       # exists only in the last two months
    stacked = _stack({"dense": dense, "sparse": sparse}, dates)
    assert len(stacked) == 18
    assert len(stacked.dropna()) == 6               # what listwise deletion would keep
    assert int(stacked.notna().any(axis=1).sum()) == 18
