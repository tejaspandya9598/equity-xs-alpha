"""Report figures. Matplotlib only, saved to reports/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = Path("reports/figures")


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ic_bars(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2b6cb0" if t > 2 else "#a0aec0" for t in summary["nw_tstat"]]
    ax.bar(summary.index, summary["mean_ic"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("mean monthly rank IC")
    ax.set_title("Signal ICs (blue = NW t-stat > 2)")
    for i, (ic, t) in enumerate(zip(summary["mean_ic"], summary["nw_tstat"])):
        ax.text(i, ic, f"t={t:.1f}", ha="center", va="bottom" if ic >= 0 else "top", fontsize=8)
    return _save(fig, "ic_summary")


def plot_ic_decay(decays: dict[str, pd.Series]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, s in decays.items():
        ax.plot(s.index, s.values, marker="o", label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("horizon (months)")
    ax.set_ylabel("mean rank IC")
    ax.set_title("IC decay by holding horizon")
    ax.legend(fontsize=8)
    return _save(fig, "ic_decay")


def plot_quintiles(qrets: dict[str, pd.Series]) -> Path:
    n = len(qrets)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2), sharey=True)
    for ax, (name, s) in zip(axes, qrets.items()):
        ax.bar(s.index, s.values * 100, color="#2b6cb0")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("mean monthly fwd ret (%)")
    fig.suptitle("Quintile monotonicity", y=1.02)
    return _save(fig, "quintiles")


def plot_equity_curves(curves: dict[str, pd.Series]) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, rets in curves.items():
        (1 + rets.dropna()).cumprod().plot(ax=ax, label=name, lw=1.4)
    ax.axhline(1, color="black", lw=0.8)
    ax.set_ylabel("growth of $1 (net of costs)")
    ax.set_title("Long-short quintile portfolios, net of 10bps costs")
    ax.legend(fontsize=8)
    return _save(fig, "equity_curves")


def plot_feature_importance(imp: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    imp.plot(ax=ax, marker="o", lw=1.2)
    ax.set_ylabel("normalized gain importance")
    ax.set_title("LightGBM feature importance across walk-forward folds")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, "feature_importance")


def plot_cumulative_ic(ics: dict, start=None) -> Path:
    """Running sum of monthly ICs. A signal that works climbs; one that does not, wanders.

    A bar chart of mean IC hides the thing that decides whether a signal is tradable,
    which is whether the IC arrives steadily or in two good years. The cumulative line
    shows it immediately.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, s in sorted(ics.items(), key=lambda kv: -kv[1].dropna().sum()):
        s = s.dropna()
        if start is not None:
            s = s[s.index >= start]
        if len(s) < 12:
            continue
        width = 2.2 if name in ("amihud", "tone_chg") else 1.0
        ax.plot(s.index, s.cumsum(), label=name, lw=width)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("cumulative monthly rank IC")
    ax.set_title("Where the information coefficient actually came from")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, "cumulative_ic")


def plot_tone_panel(tone_scores) -> Path:
    """What the EDGAR text pipeline produced, before any of it becomes a signal."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    counts = tone_scores.groupby(tone_scores["filing_date"].dt.to_period("Q")).size()
    ax.bar([p.to_timestamp() for p in counts.index], counts.to_numpy(), width=70,
           color="#2b6cb0")
    ax.set_title(f"{len(tone_scores):,} filings scored")
    ax.set_ylabel("filings per quarter")

    ax = axes[1]
    ax.hist(tone_scores["net_tone"], bins=80, color="#2b6cb0")
    ax.axvline(tone_scores["net_tone"].median(), color="#c53030", lw=1.4,
               label=f"median {tone_scores['net_tone'].median():.3f}")
    ax.set_title("Net tone: positive minus negative word share")
    ax.set_xlabel("net tone")
    ax.legend(fontsize=8)

    ax = axes[2]
    monthly = (tone_scores.set_index("filing_date")["net_tone"]
               .resample("QE").median().dropna())
    ax.plot(monthly.index, monthly, color="#2b6cb0")
    ax.set_title("Median filing tone through time")
    ax.set_ylabel("net tone")
    return _save(fig, "edgar_tone")
