"""
build_training_data.py
------------------------
Reads every session out of the SQLite database (real, collected via
data_collection/collector.html, OR synthetic, via generate_synthetic_data.py)
and turns each one into a row of features + label, ready for train.py.

Usage:
    python ml/build_training_data.py --out data/training_data.csv
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend import database  # noqa: E402
from ml import features as feat_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "training_data.csv"))
    args = ap.parse_args()

    sessions = database.get_all_sessions()
    rows = []
    skipped = 0
    for s in sessions:
        if not s["emotion_label"]:
            skipped += 1
            continue  # unlabeled session - can't train on it
        f = feat_mod.extract_features(
            s["events"],
            prompt_text=s["prompt_text"] or None,
            typed_text=s["typed_text"] or None,
        )
        f["label"] = s["emotion_label"]
        f["user_id"] = s["user_id"]
        f["session_id"] = s["session_id"]
        rows.append(f)

    if not rows:
        print("No labeled sessions found. Run generate_synthetic_data.py, "
              "load_emosurv.py, or collect data via the collector app first.")
        return

    df = pd.DataFrame(rows)
    # keep a stable, explicit column order: features, then metadata
    feature_cols = [c for c in df.columns if c not in ("label", "user_id", "session_id")]
    df = df[feature_cols + ["label", "user_id", "session_id"]]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df)} labeled rows ({skipped} unlabeled sessions skipped) -> {args.out}")
    print("\nClass balance:")
    print(df["label"].value_counts())
    print(f"\nUnique users: {df['user_id'].nunique()}")


if __name__ == "__main__":
    main()
