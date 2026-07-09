"""Anomaly detection on verified data.

Two independent detectors, both on real columns only:
  * electricity  - IsolationForest over (units_consumed, amount, deviation_from_avg)
                   flags meter/billing anomalies per apartment-month.
  * invoices     - IsolationForest over (rent, electricity, total, credit_days)
                   flags mis-billed / outlier invoices.

Outputs ranked anomaly CSVs + a scatter figure. No invented data or joins.

Run:  python -m src.anomaly
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import preprocessing  # noqa: E402


def _detect(df: pd.DataFrame, cols: list[str], contamination=0.02):
    X = df[cols].fillna(0)
    iso = IsolationForest(contamination=contamination,
                          random_state=config.RANDOM_STATE)
    df = df.copy()
    df["anomaly"] = (iso.fit_predict(X) == -1).astype(int)
    df["anomaly_score"] = -iso.score_samples(X)   # higher = more anomalous
    return df


def run():
    cleaned = preprocessing.clean_all()

    elec = _detect(cleaned["electricity"],
                   ["units_consumed", "amount", "deviation_from_avg"])
    inv = _detect(cleaned["invoices"],
                  ["rent_amount", "electricity_amount", "total_amount",
                   "credit_days"])

    n_e = int(elec["anomaly"].sum())
    n_i = int(inv["anomaly"].sum())
    print(f"electricity anomalies: {n_e}/{len(elec)}")
    print(f"invoice anomalies:     {n_i}/{len(inv)}")

    (elec.sort_values("anomaly_score", ascending=False)
         .head(50).to_csv(config.OUT_DIR / "anomalies_electricity.csv", index=False))
    (inv.sort_values("anomaly_score", ascending=False)
        .head(50).to_csv(config.OUT_DIR / "anomalies_invoices.csv", index=False))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, (df, x, y, title) in zip(ax, [
        (elec, "units_consumed", "amount", "Electricity anomalies"),
        (inv, "rent_amount", "electricity_amount", "Invoice anomalies")]):
        normal, anom = df[df.anomaly == 0], df[df.anomaly == 1]
        a.scatter(normal[x], normal[y], s=6, alpha=0.3, label="normal")
        a.scatter(anom[x], anom[y], s=18, color="#C44536", label="anomaly")
        a.set_xlabel(x); a.set_ylabel(y); a.set_title(title); a.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "anomalies.png", dpi=120)
    plt.close(fig)
    print(f"figure -> {config.FIG_DIR / 'anomalies.png'}")
    return {"electricity": n_e, "invoices": n_i}


if __name__ == "__main__":
    run()
