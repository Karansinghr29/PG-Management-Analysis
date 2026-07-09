"""Step 3 - exploratory data analysis.

Generates the full plot pack (univariate, bivariate, multivariate, trends) as
PNGs under outputs/figures/. Uses matplotlib; seaborn is used when installed.

Run:  python -m src.eda
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import feature_engineering as fe  # noqa: E402
from src import preprocessing  # noqa: E402

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    HAVE_SNS = True
except ImportError:
    HAVE_SNS = False


def _save(fig, name: str):
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# --------------------------------------------------------------------------- #
def univariate(cleaned, feats):
    inv = cleaned["invoices"]
    # Histograms of the money columns.
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col in zip(ax, ["rent_amount", "electricity_amount", "total_amount"]):
        a.hist(inv[col], bins=40, color="#4C72B0")
        a.set_title(f"Distribution: {col}")
    _save(fig, "uni_invoice_hist.png")

    # Boxplots for outliers.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].boxplot(cleaned["electricity"]["units_consumed"].dropna(), vert=True)
    ax[0].set_title("Electricity units - outliers")
    ax[1].boxplot(inv["total_amount"], vert=True)
    ax[1].set_title("Invoice total - outliers")
    _save(fig, "uni_boxplots.png")

    # Pie: ticket status mix.
    tk = cleaned["tickets"]["status"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(tk.values, labels=tk.index, autopct="%1.0f%%", startangle=90)
    ax.set_title("Maintenance ticket status")
    _save(fig, "uni_ticket_status_pie.png")

    # Pie: bed lifecycle.
    bl = feats["bed_features"]["bed_lifecycle_status"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(bl.values, labels=bl.index, autopct="%1.0f%%", startangle=90)
    ax.set_title("Bed lifecycle mix")
    _save(fig, "uni_bed_lifecycle_pie.png")


def bivariate(cleaned):
    inv = cleaned["invoices"]
    # Violin/box of total by paid/unpaid.
    groups = [inv.loc[inv.is_unpaid == 0, "total_amount"],
              inv.loc[inv.is_unpaid == 1, "total_amount"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.violinplot(groups, showmedians=True)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["paid", "unpaid"])
    ax.set_title("Invoice total by payment status")
    _save(fig, "bi_total_by_status_violin.png")

    # Scatter units vs amount.
    elec = cleaned["electricity"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(elec["units_consumed"], elec["amount"], s=8, alpha=0.4)
    ax.set_xlabel("units_consumed"); ax.set_ylabel("amount")
    ax.set_title("Electricity: units vs amount")
    _save(fig, "bi_units_vs_amount.png")


def multivariate(feats):
    inv = feats["invoice_features"]
    num = inv.select_dtypes(include=[np.number])
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90,
                                                        fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Correlation heatmap - invoice features")
    _save(fig, "multi_corr_heatmap.png")


def trends(feats):
    pm = feats["property_month"].copy()
    pm["month"] = pm["billing_period"].astype(str)

    # Revenue + collection trend.
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(pm["month"], pm["revenue"], marker="o", color="#2A9D8F",
             label="revenue")
    ax1.set_ylabel("revenue"); ax1.tick_params(axis="x", rotation=90, labelsize=7)
    ax2 = ax1.twinx()
    ax2.plot(pm["month"], pm["collection_rate"], marker="s", color="#E76F51",
             label="collection_rate")
    ax2.set_ylabel("collection_rate")
    ax1.set_title("Monthly revenue & collection rate")
    _save(fig, "trend_revenue_collection.png")

    # Active tenants (occupancy proxy) trend.
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(pm["month"], pm["active_tenants"], marker="o", color="#264653")
    ax.set_title("Active billed tenants per month (occupancy proxy)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    _save(fig, "trend_active_tenants.png")

    # Electricity cost trend.
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(pm["month"], pm["elec_cost"], marker="o", color="#E9C46A")
    ax.set_title("Monthly electricity cost")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    _save(fig, "trend_electricity.png")

    # Revenue at risk.
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(pm["month"], pm["revenue_at_risk"], color="#C44536")
    ax.set_title("Monthly revenue at risk (unpaid invoice value)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    _save(fig, "trend_revenue_at_risk.png")


def main():
    cleaned = preprocessing.clean_all()
    feats = fe.build_all(cleaned)
    univariate(cleaned, feats)
    bivariate(cleaned)
    multivariate(feats)
    trends(feats)
    print(f"\nAll figures in {config.FIG_DIR}")


if __name__ == "__main__":
    main()
