"""
load_emosurv.py
-----------------
Converts the public EmoSurv dataset (Maalej & Kallel, 2020) into the same
training_data.csv format that build_training_data.py produces from your own
collected sessions - so train.py works unmodified on either source.

EmoSurv is free but requires a (free) IEEE account to download:
    https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using
    DOI: 10.21227/eae6-pk42
    Citation: A. Maalej and I. Kallel, "Does Keystroke Dynamics tell us about
    Emotions? A Systematic Literature Review and Dataset Construction,"
    2020 16th Intl. Conf. on Intelligent Environments (IE), 2020.
Use of the dataset is subject to its non-commercial research/education license.

It ships 4 CSVs. This script uses:
  - "Fixed Text Typing Dataset.csv"  (per-keystroke rows with precomputed
     digraph/trigraph latencies: D1U1, D1U2, D1D2, U1D2, U1U2, D1U3, D1D3)
  - "Free Text Typing Dataset.csv"   (same schema, free-typing task)
  - "Frequency Dataset.csv"          (per-session DelFreq/LeftFreq/TotTime)

Emotion Index values map as: H=Happy, S=Sad, A=Angry, C=Calm, N=Neutral
(matches the 5 classes used everywhere else in this project).

Usage:
    python ml/load_emosurv.py \
        --fixed "path/to/Fixed Text Typing Dataset.csv" \
        --free  "path/to/Free Text Typing Dataset.csv" \
        --freq  "path/to/Frequency Dataset.csv" \
        --out data/training_data_emosurv.csv
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

EMOTION_MAP = {"H": "Happy", "S": "Sad", "A": "Angry", "C": "Calm", "N": "Neutral"}

# The precomputed timing columns documented for EmoSurv's per-keystroke rows.
TIMING_COLS_CANDIDATES = ["D1U1", "D1U2", "D1D2", "U1D2", "U1U2", "D1U3", "D1D3"]


def _norm(col: str) -> str:
    """Normalise a column name for fuzzy matching: lowercase, strip, no spaces."""
    return re.sub(r"[^a-z0-9]", "", col.lower())


def _find_col(df: pd.DataFrame, *candidates: str) -> str:
    """Find the actual column in df matching any of the candidate names,
    tolerant of spacing/case differences between the docs and the real CSV."""
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def load_typing_file(path: str, text_type: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",")
    user_col = _find_col(df, "User Id", "UserID", "User ID")
    emo_col = _find_col(df, "Emotion Index", "EmotionIndex")

    present_timing_cols = [c for c in TIMING_COLS_CANDIDATES
                            if _norm(c) in {_norm(x) for x in df.columns}]
    if not present_timing_cols:
        print(f"WARNING: no recognised timing columns found in {path}. "
              f"Columns present: {list(df.columns)}")

    real_timing_cols = [_find_col(df, c) for c in present_timing_cols]

    agg = df.groupby([user_col, emo_col])[real_timing_cols].agg(
        ["mean", "median", "std", "min", "max"]
    )
    RENAME_MAP = {
        "D1U1": "dwell", "D1D2": "DD", "U1D2": "UD_flight",
        "U1U2": "UU", "D1U2": "DU", "D1D3": "DD_tri", "D1U3": "DU_tri",
    }
    col_to_prefix = {real: RENAME_MAP[cand] for cand, real in zip(present_timing_cols, real_timing_cols)}
    agg.columns = [f"{col_to_prefix.get(col, col)}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index().rename(columns={user_col: "user_id", emo_col: "emotion_code"})
    agg["text_type"] = text_type
    agg = agg.fillna(0.0)
    return agg


def load_frequency_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",")
    user_col = _find_col(df, "User Id", "UserID", "User ID")
    emo_col = _find_col(df, "Emotion Index", "EmotionIndex")
    text_col = _find_col(df, "textIndex", "TextIndex")
    del_col = _find_col(df, "DelFreq")
    left_col = _find_col(df, "LeftFreq")
    tot_col = _find_col(df, "TotTime")

    out = df[[user_col, emo_col, text_col, del_col, left_col, tot_col]].copy()
    out.columns = ["user_id", "emotion_code", "text_type_code", "del_freq", "backspace_freq", "total_time"]
    out["text_type"] = out["text_type_code"].map({"FI": "fixed", "FR": "free"}).fillna(out["text_type_code"])
    return out.drop(columns=["text_type_code"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True, help="path to Fixed Text Typing Dataset.csv")
    ap.add_argument("--free", required=True, help="path to Free Text Typing Dataset.csv")
    ap.add_argument("--freq", required=False, help="path to Frequency Dataset.csv (optional)")
    ap.add_argument("--out", default="data/training_data_emosurv.csv")
    args = ap.parse_args()

    fixed_agg = load_typing_file(args.fixed, "fixed")
    free_agg = load_typing_file(args.free, "free")
    combined = pd.concat([fixed_agg, free_agg], ignore_index=True)

    if args.freq:
        freq = load_frequency_file(args.freq)
        combined = combined.merge(
            freq, on=["user_id", "emotion_code", "text_type"], how="left"
        )
        combined = combined.fillna(0.0)

    combined["label"] = combined["emotion_code"].map(EMOTION_MAP).fillna(combined["emotion_code"])
    combined["session_id"] = combined["user_id"].astype(str) + "_" + combined["emotion_code"].astype(str) + "_" + combined["text_type"]
    combined = combined.drop(columns=["emotion_code"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"Wrote {len(combined)} rows -> {args.out}")
    print("\nClass balance:")
    print(combined["label"].value_counts())


if __name__ == "__main__":
    main()
