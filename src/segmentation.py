"""Tenant segmentation via clustering on REAL billing behaviour.

Features come only from grouping invoices on the real `tenant_id` key
(tenure, billings, payment reliability, avg rent) - no cross-table joins.
Picks k by silhouette, labels each segment, saves assignments + profile.

Run:  python -m src.segmentation
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import feature_engineering as fe  # noqa: E402

FEATURES = ["tenure_months", "total_billed", "avg_rent", "unpaid_ratio",
            "invoices", "ltv_paid"]


def run():
    tf = fe.build_all()["tenant_features"].copy()
    X = StandardScaler().fit_transform(tf[FEATURES].fillna(0))

    best_k, best_s = 3, -1
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        lab = km.fit_predict(X)
        s = silhouette_score(X, lab)
        if s > best_s:
            best_k, best_s = k, s
    km = KMeans(n_clusters=best_k, random_state=config.RANDOM_STATE, n_init=10)
    tf["segment"] = km.fit_predict(X)
    print(f"chosen k={best_k}  silhouette={best_s:.3f}")

    profile = tf.groupby("segment")[FEATURES].mean().round(1)
    profile["n_tenants"] = tf["segment"].value_counts().sort_index()
    profile.to_csv(config.OUT_DIR / "tenant_segments_profile.csv")
    tf[["tenant_id", "segment"] + FEATURES].to_csv(
        config.OUT_DIR / "tenant_segments.csv", index=False)
    print(profile.to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(tf["tenure_months"], tf["ltv_paid"], c=tf["segment"],
                    cmap="viridis", s=18, alpha=0.7)
    ax.set_xlabel("tenure_months"); ax.set_ylabel("ltv_paid (₹)")
    ax.set_title(f"Tenant segments (k={best_k})")
    fig.colorbar(sc, label="segment")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "tenant_segments.png", dpi=120)
    plt.close(fig)
    return profile


if __name__ == "__main__":
    run()
