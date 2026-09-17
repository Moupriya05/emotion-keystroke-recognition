"""
train.py
--------
Trains and compares several classifiers on the keystroke-dynamics feature
table produced by build_training_data.py (or load_emosurv.py), then saves
the best one for use by the FastAPI backend (backend/app.py).

WHY GROUP-AWARE SPLITTING MATTERS:
If the same person's sessions can land in both the train and test sets, the
model can partly learn to recognise the PERSON rather than the EMOTION, and
your accuracy will look great but be meaningless. Every split below is done
by `user_id`, so a given person's sessions are entirely in train OR entirely
in test - never both. This is the single most important methodological
detail in this whole project; don't skip it even if it hurts your headline
accuracy number.

Usage:
    python ml/train.py --data data/training_data.csv
"""

from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


NON_FEATURE_COLS = {"label", "user_id", "session_id", "text_type"}


def build_models(random_state: int = 42) -> dict:
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=None,
                                            random_state=random_state, n_jobs=-1)),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=5.0, gamma="scale", probability=True,
                        random_state=random_state)),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, random_state=random_state)),
        ]),
    }
    if HAS_XGB:
        models["xgboost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
                random_state=random_state, n_jobs=-1,
            )),
        ])
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "training_data.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "models"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values
    y_raw = df["label"].values
    groups = df["user_id"].values

    n_classes = len(set(y_raw))
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features, "
          f"{n_classes} classes, {df['user_id'].nunique()} unique users.")
    print("Class balance:\n", df["label"].value_counts(), "\n")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # ---- person-independent train/test split ----------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    print(f"Train: {len(X_train)} sessions / {len(set(groups_train))} users | "
          f"Test: {len(X_test)} sessions / {len(set(groups[test_idx]))} users\n")

    # ---- compare models with group-aware cross-validation ---------------
    models = build_models(args.seed)
    n_splits = min(args.cv_folds, pd.Series(groups_train).nunique())
    n_splits = max(n_splits, 2)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    results = {}
    print(f"Cross-validating {len(models)} models ({n_splits}-fold, group-aware, f1_macro)...\n")
    for name, pipe in models.items():
        try:
            scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                      groups=groups_train, scoring="f1_macro", n_jobs=-1)
            results[name] = scores
            print(f"  {name:22s} f1_macro = {scores.mean():.3f} (+/- {scores.std():.3f})")
        except Exception as e:
            print(f"  {name:22s} FAILED: {e}")

    if not results:
        print("All models failed cross-validation - check your data.")
        return

    best_name = max(results, key=lambda k: results[k].mean())
    print(f"\nBest model by CV f1_macro: {best_name}")

    # ---- refit best model on the full training set, evaluate on test ----
    best_pipe = models[best_name]
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nHeld-out (person-independent) test accuracy: {test_acc:.3f}")
    print(f"Held-out (person-independent) test f1_macro : {test_f1:.3f}\n")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    # ---- confusion matrix plot -------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}\ntest accuracy={test_acc:.2f}, f1_macro={test_f1:.2f}")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ---- save everything app.py needs at inference time ------------------
    joblib.dump(best_pipe, out_dir / "best_model.joblib")
    joblib.dump(le, out_dir / "label_encoder.joblib")
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    (out_dir / "report.txt").write_text(
        f"Best model: {best_name}\n\n"
        f"CV results (train set, group-aware {n_splits}-fold, f1_macro):\n"
        + "\n".join(f"  {k}: {v.mean():.3f} +/- {v.std():.3f}" for k, v in results.items())
        + f"\n\nHeld-out test accuracy: {test_acc:.3f}\nHeld-out test f1_macro: {test_f1:.3f}\n\n"
        + report
    )

    print(f"\nSaved model artifacts to {out_dir}/ "
          f"(best_model.joblib, label_encoder.joblib, feature_columns.json, "
          f"confusion_matrix.png, report.txt)")


if __name__ == "__main__":
    main()
