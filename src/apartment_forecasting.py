"""Apartment-wise forecasting on verified data.

The electricity table has a real (apartment_code x billing_month) grain, so each
apartment's monthly **electricity units and billing amount** form a genuine time
series that can be forecast per apartment.

IMPORTANT - apartment-wise RENT revenue is NOT forecastable from this data:
`invoices` carry no apartment_code/bed_code and the bed tables have no time
history (see reports/blocked_predictions.md). The only real apartment-level money
series is electricity `amount`, which is what we forecast here. No data invented.

Run:  python -m src.apartment_forecasting
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
from src import forecasting as fc  # noqa: E402
from src import preprocessing  # noqa: E402

MIN_MONTHS = 12   # need at least a year of history to forecast an apartment


def _apartment_series(elec: pd.DataFrame) -> pd.DataFrame:
    """Dense apartment x month panel for units_consumed and amount."""
    g = (elec.groupby(["apartment_code", "billing_period"])
             .agg(units_consumed=("units_consumed", "sum"),
                  amount=("amount", "sum")).reset_index())
    return g.sort_values(["apartment_code", "billing_period"])


def forecast_apartments(steps: int = 6):
    elec = preprocessing.clean_all()["electricity"]
    panel = _apartment_series(elec)
    all_periods = pd.period_range(panel["billing_period"].min(),
                                  panel["billing_period"].max(), freq="M")

    rows, skipped = [], []
    for apt, sub in panel.groupby("apartment_code"):
        sub = sub.set_index("billing_period").reindex(all_periods)
        y_amt = sub["amount"].interpolate().bfill().ffill().to_numpy(dtype=float)
        y_units = sub["units_consumed"].interpolate().bfill().ffill().to_numpy(float)
        if np.count_nonzero(~np.isnan(y_amt)) < MIN_MONTHS:
            skipped.append(apt); continue
        m_amt, bt_amt = fc.select_method(y_amt)
        m_un, bt_un = fc.select_method(y_units)
        f_amt = fc._fit_forecast(y_amt, steps, m_amt)
        f_un = fc._fit_forecast(y_units, steps, m_un)
        future = pd.period_range(all_periods.max() + 1, periods=steps, freq="M")
        for p, fa, fu in zip(future, f_amt, f_un):
            rows.append({"apartment_code": apt, "billing_period": str(p),
                         "amount_forecast": round(float(fa), 1),
                         "units_forecast": round(float(fu), 1),
                         "amount_method": m_amt, "amount_mape": round(bt_amt["MAPE"], 2)})

    out = pd.DataFrame(rows)
    out.to_csv(config.OUT_DIR / "forecast_apartment_electricity.csv", index=False)

    # Summary: next-month forecast + backtest error per apartment.
    nxt = (out.sort_values("billing_period").groupby("apartment_code")
              .first().reset_index()
              .rename(columns={"amount_forecast": "next_month_amount",
                               "units_forecast": "next_month_units"}))
    nxt = nxt.sort_values("next_month_amount", ascending=False)
    nxt.to_csv(config.OUT_DIR / "forecast_apartment_summary.csv", index=False)

    print(f"forecast {out['apartment_code'].nunique()} apartments x {steps} months")
    if skipped:
        print(f"skipped {len(skipped)} apartments (< {MIN_MONTHS} months history): "
              f"{skipped}")
    print("\nTop 10 apartments by next-month electricity amount:")
    print(nxt[["apartment_code", "next_month_amount", "next_month_units",
               "amount_method", "amount_mape"]].head(10).to_string(index=False))

    # Plot top 6 apartments actual + forecast.
    top = nxt["apartment_code"].head(6).tolist()
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, apt in zip(axes.ravel(), top):
        hist = panel[panel.apartment_code == apt].set_index("billing_period") \
            .reindex(all_periods)["amount"].interpolate().bfill().ffill()
        fpart = out[out.apartment_code == apt]
        ax.plot([str(p) for p in all_periods], hist.to_numpy(), marker="o",
                label="actual", ms=3)
        ax.plot(fpart["billing_period"], fpart["amount_forecast"], marker="s",
                ls="--", color="#C44536", label="forecast", ms=4)
        ax.set_title(f"Apt {apt} - electricity amount")
        ax.tick_params(axis="x", rotation=90, labelsize=5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "forecast_apartments.png", dpi=110)
    plt.close(fig)
    print(f"\nfigure -> {config.FIG_DIR / 'forecast_apartments.png'}")
    return nxt


if __name__ == "__main__":
    forecast_apartments()
