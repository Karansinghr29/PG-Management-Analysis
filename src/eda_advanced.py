"""Advanced EDA on verified data only.

  * Seasonal decomposition of monthly revenue (statsmodels when available).
  * Apartment x month electricity heatmap (real apartment_code series).
  * Collection-rate cohort by billing month.
  * Ticket resolution-time distribution + issue-type x priority crosstab.
  * Rent distribution by bed type.

Run:  python -m src.eda_advanced
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


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def seasonal_decomposition(pm):
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        print("statsmodels absent - skipping decomposition"); return
    s = pd.Series(pm["revenue"].to_numpy(),
                  index=pm["billing_period"].dt.to_timestamp())
    res = seasonal_decompose(s, model="additive", period=12)
    fig = res.plot()
    fig.set_size_inches(12, 8)
    _save(fig, "adv_revenue_decomposition.png")


def electricity_heatmap(elec):
    piv = elec.pivot_table(index="apartment_code", columns="billing_period",
                           values="units_consumed", aggfunc="mean")
    piv = piv.reindex(sorted(piv.columns), axis=1)
    fig, ax = plt.subplots(figsize=(15, 9))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=6)
    ax.set_xticks(range(0, len(piv.columns), 2))
    ax.set_xticklabels([str(c) for c in piv.columns[::2]], rotation=90, fontsize=6)
    fig.colorbar(im, label="units_consumed")
    ax.set_title("Electricity units - apartment x month")
    _save(fig, "adv_electricity_heatmap.png")


def collection_cohort(inv):
    coh = (inv.groupby("billing_period")
              .agg(paid_rate=("is_unpaid", lambda x: 1 - x.mean()),
                   invoices=("invoice_id", "count")).reset_index())
    coh["month"] = coh["billing_period"].astype(str)
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["#C44536" if r < 0.8 else "#2A9D8F" for r in coh["paid_rate"]]
    ax.bar(coh["month"], coh["paid_rate"], color=colors)
    ax.axhline(0.9, ls="--", color="grey")
    ax.set_title("Collection rate by billing month (red < 80%)")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    _save(fig, "adv_collection_cohort.png")


def ticket_analysis(tk):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    rt = tk["resolution_hours"].dropna()
    rt = rt[(rt >= 0) & (rt < rt.quantile(0.99))]
    ax[0].hist(rt, bins=40, color="#457B9D")
    ax[0].set_title("Resolution time (hours, <99pct)")
    ct = pd.crosstab(tk["issue_type"], tk["priority"])
    im = ax[1].imshow(ct.values, aspect="auto", cmap="Blues")
    ax[1].set_yticks(range(len(ct.index))); ax[1].set_yticklabels(ct.index, fontsize=7)
    ax[1].set_xticks(range(len(ct.columns))); ax[1].set_xticklabels(ct.columns)
    ax[1].set_title("Issue type x priority")
    fig.colorbar(im, ax=ax[1])
    _save(fig, "adv_ticket_analysis.png")


def rent_by_bedtype(beds):
    types = beds["bed_type"].dropna().unique()
    data = [beds.loc[beds.bed_type == t, "current_rate"].dropna() for t in types]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.violinplot([d for d in data if len(d)], showmedians=True)
    ax.set_xticks(range(1, len(types) + 1)); ax.set_xticklabels(types)
    ax.set_title("Rent distribution by bed type")
    _save(fig, "adv_rent_by_bedtype.png")


def main():
    cleaned = preprocessing.clean_all()
    feats = fe.build_all(cleaned)
    seasonal_decomposition(feats["property_month"])
    electricity_heatmap(cleaned["electricity"])
    collection_cohort(cleaned["invoices"])
    ticket_analysis(cleaned["tickets"])
    rent_by_bedtype(feats["bed_features"])
    print(f"\nadvanced figures in {config.FIG_DIR}")


if __name__ == "__main__":
    main()
