"""Step 9 - inference. Load a persisted best model and score new rows.

Usage:
    from predict import predict
    predict("late_payment", new_dataframe)      # returns predictions

CLI smoke test:
    python predict.py late_payment              # scores a sample of the data
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from src import feature_engineering as fe  # noqa: E402
from src import train  # noqa: E402


def load_model(problem: str):
    path = config.MODEL_DIR / f"{problem}_best.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No trained model for '{problem}'. Run train.py.")
    return joblib.load(path)


def predict(problem: str, X: pd.DataFrame):
    """Score a feature frame with the saved best model for `problem`."""
    model = load_model(problem)
    return model.predict(X)


def _sample_features(problem: str) -> pd.DataFrame:
    feats = fe.build_all()
    X = train.PROBLEMS[problem](feats)[0]
    return X.replace([float("inf"), float("-inf")], 0).fillna(0)


if __name__ == "__main__":
    prob = sys.argv[1] if len(sys.argv) > 1 else "late_payment"
    X = _sample_features(prob).head(10)
    preds = predict(prob, X)
    out = X.copy()
    out["prediction"] = preds
    print(f"Predictions for '{prob}':")
    print(out[["prediction"]].to_string())
