"""
evaluate.py
------------
Explainability pass over the model saved by train.py: which features
actually drove its decisions? Useful for your project report / viva -
"why does the model think fast + erratic typing means anger?" etc.

Produces:
    models/feature_importance.png   (built-in importances, if the best
                                      model exposes them e.g. RandomForest/XGBoost)
    models/shap_summary.png         (SHAP beeswarm plot, model-agnostic)

Usage:
    python ml/evaluate.py --data data/training_data.csv
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NON_FEATURE_COLS = {"label", "user_id", "session_id"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "training_data.csv"))
    ap.add_argument("--model-dir", default=str(ROOT / "models"))
    ap.add_argument("--sample-size", type=int, default=200,
                     help="rows to sample for SHAP (it's slow on large datasets)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    pipe = joblib.load(model_dir / "best_model.joblib")
    feature_cols = pd.read_json(model_dir / "feature_columns.json", typ="series").tolist()

    df = pd.read_csv(args.data)
    X = df[feature_cols]
    if len(X) > args.sample_size:
        X = X.sample(args.sample_size, random_state=42)

    scaler = pipe.named_steps.get("scaler")
    clf = pipe.named_steps.get("clf")
    X_scaled = scaler.transform(X) if scaler is not None else X.values

    # ---- 1) built-in feature importances (tree models only) -------------
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        order = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(order)), importances[order][::-1])
        plt.yticks(range(len(order)), [feature_cols[i] for i in order][::-1])
        plt.xlabel("Importance")
        plt.title(f"Top-15 feature importances ({type(clf).__name__})")
        plt.tight_layout()
        plt.savefig(model_dir / "feature_importance.png", dpi=150)
        plt.close()
        print(f"Saved {model_dir / 'feature_importance.png'}")
    else:
        print(f"{type(clf).__name__} has no .feature_importances_ - skipping built-in importance plot "
              f"(SVM/LogisticRegression don't expose this; SHAP below still works).")

    # ---- 2) SHAP (model-agnostic, slower) --------------------------------
    try:
        import shap
        print("Computing SHAP values (this can take a minute)...")
        explainer = shap.Explainer(clf.predict_proba, X_scaled, feature_names=feature_cols)
        shap_values = explainer(X_scaled[:min(100, len(X_scaled))])
        shap.summary_plot(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(model_dir / "shap_summary.png", dpi=150)
        plt.close()
        print(f"Saved {model_dir / 'shap_summary.png'}")
    except ImportError:
        print("shap not installed - run `pip install shap` for this part (optional).")
    except Exception as e:
        print(f"SHAP step skipped due to error: {e}")


if __name__ == "__main__":
    main()
