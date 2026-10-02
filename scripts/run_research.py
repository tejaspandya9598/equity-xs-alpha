"""End-to-end research run.

Usage:
    python scripts/run_research.py            # full run (downloads ~500 names)
    python scripts/run_research.py --refresh  # ignore parquet cache

Produces reports/figures/*.png and reports/results.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsalpha import data, ic, ml, plots, portfolio, signals, universe

COST_BPS = 10.0


def main(refresh: bool = False, no_text: bool = False, eval_start: str | None = None) -> None:
    print("universe ...")
    tickers = universe.sp500_tickers()
    print(f"  {len(tickers)} tickers")

    print("prices ...")
    px, dvol = data.download_prices(tickers, start="2010-01-01", cache=not refresh)
    px = universe.filter_by_history(px, min_years=3)
    dvol = dvol[px.columns]
    print(f"  panel: {px.shape[0]} days x {px.shape[1]} names")

    dates = data.month_ends(px)
    fwd = data.forward_returns(px, dates).iloc[:-1]

    tone_path = Path("data/edgar/tone_scores.parquet")
    tone = None
    if no_text:
        print("--no-text: the ablation run, price and volume signals only")
    elif tone_path.exists():
        tone = pd.read_parquet(tone_path)
        print(f"EDGAR tone panel: {len(tone):,} filings, {tone.ticker.nunique()} tickers, "
              f"{tone.filing_date.min():%Y-%m} to {tone.filing_date.max():%Y-%m}")
    else:
        print("no EDGAR tone panel (run scripts/build_text_panel.py); text signal skipped")

    print("signals ...")
    sigs = signals.build_signal_panel(px, dvol, dates, tone_scores=tone)

    print("IC analysis ...")
    ics = {name: ic.ic_series(df, fwd) for name, df in sigs.items()}
    summary = ic.ic_summary(ics)
    print(summary.round(3).to_string())

    # Same table on the window every signal shares. The text signal only exists from
    # 2016 because that is where the EDGAR pull starts, and the price signals reach
    # back to 2010; the full-sample table above compares different decades.
    common_start = max(s.dropna().index.min() for s in ics.values())
    common = ic.ic_summary(ics, start=common_start)
    print(f"\ncommon window from {common_start:%Y-%m}:")
    print(common.round(3).to_string())

    decays = {name: ic.ic_decay(df, px, dates) for name, df in sigs.items()}

    print("portfolios ...")
    bts = {name: portfolio.backtest_ls(df, fwd, cost_bps=COST_BPS) for name, df in sigs.items()}
    qrets = {name: portfolio.quintile_mean_returns(df, fwd) for name, df in sigs.items()}

    print("ML combo (purged walk-forward) ...")
    scores, imp = ml.ml_scores(sigs, fwd)
    scores_clean = signals.clean(scores)
    bts["ml_combo"] = portfolio.backtest_ls(scores_clean, fwd, cost_bps=COST_BPS)
    combo = ml.equal_weight_combo(sigs)
    bts["equal_weight_combo"] = portfolio.backtest_ls(combo, fwd, cost_bps=COST_BPS)
    # Restrict baselines to the ML OOS window so the comparison is fair. That window
    # moves when the signal set changes - adding a signal that only starts in 2016
    # shortens what the walk-forward can train on - so the ablation has to be pinned to
    # the same start, or it is comparing two different samples and calling it an effect.
    oos_start = scores_clean.index.min()
    if eval_start is not None:
        oos_start = max(oos_start, pd.Timestamp(eval_start))
    print(f"  evaluation window starts {oos_start:%Y-%m}")

    print("figures ...")
    plots.plot_ic_bars(summary)
    plots.plot_ic_decay(decays)
    plots.plot_quintiles(qrets)
    plots.plot_feature_importance(imp)
    plots.plot_cumulative_ic(ics, start=common_start)
    if tone is not None:
        plots.plot_tone_panel(tone)
    curves = {name: bt["net_ret"].loc[oos_start:] for name, bt in bts.items()}
    plots.plot_equity_curves(curves)

    print("report ...")
    perf_rows = {}
    for name, bt in bts.items():
        stats = portfolio.perf_stats(bt["net_ret"].loc[oos_start:])
        stats["avg_turnover"] = float(bt["turnover"].loc[oos_start:].mean())
        perf_rows[name] = stats
    perf = pd.DataFrame(perf_rows).T.sort_values("sharpe", ascending=False)

    out = Path("reports/results_no_text.md" if no_text else "reports/results.md")
    with out.open("w") as f:
        f.write("# Research run results\n\n")
        f.write(f"Universe: {px.shape[1]} names | {dates[0]:%Y-%m} to {dates[-1]:%Y-%m} | ")
        f.write(f"costs {COST_BPS:.0f} bps | OOS window starts {oos_start:%Y-%m}\n\n")
        f.write("## Signal ICs (full sample)\n\n")
        f.write(summary.round(4).to_markdown() + "\n\n")
        f.write(f"## Signal ICs on the common window ({common_start:%Y-%m} onward)\n\n")
        f.write(common.round(4).to_markdown() + "\n\n")
        f.write("## IC decay (mean IC by horizon, months)\n\n")
        f.write(pd.DataFrame(decays).round(4).to_markdown() + "\n\n")
        f.write(f"## Long-short quintile performance, net of costs (OOS window {oos_start:%Y-%m} onward)\n\n")
        f.write(perf.round(3).to_markdown() + "\n")
    print(f"  wrote {out}")
    print(perf.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore parquet cache")
    ap.add_argument("--no-text", action="store_true",
                    help="drop the EDGAR tone signal (the ablation)")
    ap.add_argument("--eval-start", help="pin the evaluation window start (YYYY-MM-DD)")
    main(**vars(ap.parse_args()))
