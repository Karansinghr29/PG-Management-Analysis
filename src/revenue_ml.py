"""Second revenue model: supervised ML on the real monthly `property_month` panel.

This is a COMPLEMENT to (not a replacement for) the primary time-series model in
`src/forecasting.py`. It frames next-month revenue as a supervised regression on
lagged + calendar features and compares XGBoost / RandomForest / LinearRegression
against the time-series forecaster on identical walk-forward windows.

Leakage discipline (consistent with the rest of the project):
  * Only features KNOWN BEFORE month t are used - lagged revenue, a lagged
    3-month rolling mean, lagged tenant count, and the deterministic calendar.
  * EXCLUDED as target leakage / not-known-at-forecast-time:
      - arpu  (= revenue / active_tenants -> literally contains the target)
      - revenue_roll3 (pandas trailing window includes month t)
      - revenue_mom   (pct-change includes month t)
      - contemporaneous active_tenants / collection_rate / units /
        electricity_billed (month-t values are unknown when forecasting month t)
  * Chronological expanding-window (walk-forward) validation only - no random
    split, no shuffling.

Run:  python -m src.revenue_ml
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402
from src import feature_engineering as fe  # noqa: E402
from src import forecasting as tsf  # noqa: E402

TARGET = "revenue"
# Leakage-safe features computed below (all lagged / deterministic).
FEATURES = ["rev_lag1", "rev_lag2", "rev_lag3", "rev_lag12", "rev_roll3_lag",
            "tenants_lag1", "month_num", "year"]
N_TEST = 12          # evaluate the most recent 12 months, expanding train window
TS_METHOD = "seasonal_naive"   # primary model's selected method for revenue


def _build_frame(pm: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe supervised frame from the real monthly series."""
    df = pm.sort_values("billing_period").reset_index(drop=True).copy()
    rev = df[TARGET]
    df["rev_lag1"] = rev.shift(1)
    df["rev_lag2"] = rev.shift(2)
    df["rev_lag3"] = rev.shift(3)
    df["rev_lag12"] = rev.shift(12)
    df["rev_roll3_lag"] = rev.shift(1).rolling(3).mean()   # mean of t-1,t-2,t-3
    df["tenants_lag1"] = df["active_tenants"].shift(1)
    return df


def _models():
    m = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor
        m["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=config.RANDOM_STATE, verbosity=0)
    except ImportError:
        pass
    return m


def run():
    t0 = time.time()
    pm = fe.build_all()["property_month"]
    df = _build_frame(pm)
    rev_series = df[TARGET].to_numpy(dtype=float)
    months = df["billing_period"].astype(str).to_numpy()

    # Usable rows have all lag features (drops the first 12 months).
    usable = df.dropna(subset=FEATURES).index.to_numpy()
    test_idx = usable[usable >= len(df) - N_TEST]
    if len(test_idx) < 4:
        test_idx = usable[-max(4, len(usable) // 3):]

    models = _models()
    preds = {name: [] for name in models}
    preds["TimeSeries"] = []
    actuals, test_months = [], []

    for i in test_idx:
        tr = [u for u in usable if u < i]         # strictly-past training rows
        if len(tr) < 6:
            continue
        X_tr = df.loc[tr, FEATURES].to_numpy(dtype=float)
        y_tr = df.loc[tr, TARGET].to_numpy(dtype=float)
        X_te = df.loc[[i], FEATURES].to_numpy(dtype=float)
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            preds[name].append(float(model.predict(X_te)[0]))
        # Time-series model on the SAME origin, one-step ahead (fair comparison).
        ts_pred = tsf._fit_forecast(rev_series[:i], 1, TS_METHOD)[0]
        preds["TimeSeries"].append(float(ts_pred))
        actuals.append(float(rev_series[i]))
        test_months.append(months[i])

    actuals = np.array(actuals)
    rows = []
    for name, p in preds.items():
        m = utils.regression_metrics(actuals, np.array(p))
        m["Model"] = name
        rows.append(m)
    comp = (pd.DataFrame(rows).set_index("Model")
              [["MAE", "RMSE", "MAPE", "R2"]].round(2)
              .sort_values("MAPE"))
    comp.to_csv(config.OUT_DIR / "comparison_revenue_models.csv")

    # ML leaderboard (exclude the time-series row) + pick best ML model.
    ml_board = comp.drop(index="TimeSeries", errors="ignore")
    best_ml = ml_board.index[0]
    ml_board.to_csv(config.OUT_DIR / "leaderboard_revenue_ml.csv")

    # Pred-vs-actual for the dashboard (best ML + time-series on same months).
    pd.DataFrame({
        "billing_period": test_months, "actual": actuals,
        "ml_predicted": preds[best_ml], "ts_predicted": preds["TimeSeries"],
    }).to_csv(config.OUT_DIR / "backtest_revenue_ml.csv", index=False)

    # Refit best ML on ALL usable rows; forecast next month from real lag values.
    final = _models()[best_ml]
    Xall = df.loc[usable, FEATURES].to_numpy(dtype=float)
    yall = df.loc[usable, TARGET].to_numpy(dtype=float)
    final.fit(Xall, yall)
    next_feat = _next_month_features(df)
    next_pred = float(final.predict(next_feat)[0]) if next_feat is not None else None

    import joblib
    joblib.dump(final, config.MODEL_DIR / "revenue_ml_best.pkl")

    ts_mape = float(comp.loc["TimeSeries", "MAPE"])
    ml_mape = float(comp.loc[best_ml, "MAPE"])
    winner = "TimeSeries" if ts_mape <= ml_mape else best_ml
    meta = {
        "primary_model": "TimeSeries (Holt-Winters / seasonal-naive) — unchanged",
        "ml_model": best_ml,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "n_train_months_total": int(len(usable)),
        "n_test_months": int(len(actuals)),
        "validation": "chronological expanding-window walk-forward (one-step)",
        "excluded_leakage_features": ["arpu", "revenue_roll3", "revenue_mom",
                                      "contemporaneous active_tenants/"
                                      "collection_rate/units/electricity_billed"],
        "winner": winner,
        "ts_mape": ts_mape, "ml_mape": ml_mape,
        "ml_next_month_revenue": round(next_pred, 1) if next_pred else None,
        "trained_at": _utc_now(),
        "training_duration_sec": round(time.time() - t0, 1),
        "model_version": _utc_version(),
    }
    (config.OUT_DIR / "model_meta_revenue_ml.json").write_text(
        json.dumps(meta, indent=2))

    _plot(test_months, actuals, preds, best_ml)

    print("Revenue model comparison (walk-forward, "
          f"{len(actuals)} test months):")
    print(comp.to_string())
    print(f"\nBest ML model: {best_ml} | Winner overall: {winner}")
    print("ML next-month revenue prediction: "
          f"{'NA' if next_pred is None else f'Rs {next_pred/1e5:.2f} L'}")
    return comp


def _next_month_features(df):
    """Feature row for the month after the last observed one (real lags)."""
    rev = df[TARGET]
    last = df.iloc[-1]
    row = {
        "rev_lag1": rev.iloc[-1], "rev_lag2": rev.iloc[-2], "rev_lag3": rev.iloc[-3],
        "rev_lag12": rev.iloc[-12] if len(rev) >= 12 else rev.iloc[0],
        "rev_roll3_lag": rev.iloc[-3:].mean(),
        "tenants_lag1": last["active_tenants"],
        "month_num": (int(last["month_num"]) % 12) + 1,
        "year": int(last["year"]) + (1 if last["month_num"] == 12 else 0),
    }
    return np.array([[row[f] for f in FEATURES]], dtype=float)


def _plot(months, actuals, preds, best_ml):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(months, actuals, marker="o", color="#2A9D8F", label="actual", lw=2)
    ax.plot(months, preds["TimeSeries"], marker="s", ls="--", color="#E76F51",
            label="Time-Series forecast")
    ax.plot(months, preds[best_ml], marker="^", ls=":", color="#264653",
            label=f"ML ({best_ml})")
    ax.set_title("Revenue: actual vs Time-Series vs ML (walk-forward)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "revenue_model_comparison.png", dpi=120)
    plt.close(fig)


def _utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _utc_version():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("v%Y%m%d.%H%M")


if __name__ == "__main__":
    run()
