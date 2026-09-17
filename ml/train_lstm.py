"""
train_lstm.py  (OPTIONAL / ADVANCED)
--------------------------------------
The main pipeline (train.py) aggregates a whole session into ~46 summary
statistics (mean/std/etc of dwell & flight times) and feeds that flat
vector to classical ML models. That throws away the SHAPE of the typing
rhythm over time.

This script instead treats each session as a SEQUENCE of per-keystroke
vectors [dwell_i, flight_to_next_i, is_correction_i] and feeds the whole
sequence into an LSTM, which can learn temporal patterns like "starts calm,
gets erratic near the end" that summary statistics can't capture. This is
the same family of idea as recent transformer/LSTM keystroke-emotion work
(e.g. the DSTER model, Maalej & Kallel's follow-ups).

This is genuinely optional - only try it once the classical pipeline in
train.py is working and you want a "stretch" component for your report.
It needs meaningfully more data than the classical models to beat them.

Usage:
    python ml/train_lstm.py --epochs 30
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend import database  # noqa: E402

MAX_SEQ_LEN = 80  # sessions are truncated/padded to this many keystrokes


def session_to_sequence(events: list[dict]) -> np.ndarray:
    """
    Converts one session's raw events into a (T, 3) array of
    [dwell_ms, flight_to_next_ms, is_correction] per keystroke,
    normalised in scale (raw ms, scaler applied later).
    """
    from ml.features import _pair_down_up, CORRECTION_CODES  # reuse pairing logic

    keystrokes = _pair_down_up(events)
    if len(keystrokes) < 2:
        return np.zeros((0, 3), dtype=np.float32)

    seq = []
    for i in range(len(keystrokes) - 1):
        a, b = keystrokes[i], keystrokes[i + 1]
        dwell = a["up"] - a["down"]
        flight = b["down"] - a["up"]
        is_corr = 1.0 if a["code"] in CORRECTION_CODES else 0.0
        seq.append([dwell, flight, is_corr])
    return np.array(seq, dtype=np.float32)


class KeystrokeSeqDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray], labels: np.ndarray, scaler: StandardScaler):
        self.sequences = sequences
        self.labels = labels
        self.scaler = scaler

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:MAX_SEQ_LEN]
        length = len(seq)
        if length == 0:
            seq = np.zeros((1, 3), dtype=np.float32)
            length = 1
        seq = self.scaler.transform(seq)
        padded = np.zeros((MAX_SEQ_LEN, 3), dtype=np.float32)
        padded[:length] = seq[:length]
        return (
            torch.tensor(padded, dtype=torch.float32),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class KeystrokeLSTM(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                             batch_first=True, bidirectional=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * num_directions, batch, hidden) -> take last layer, both directions
        last_fwd, last_bwd = h_n[-2], h_n[-1]
        combined = torch.cat([last_fwd, last_bwd], dim=1)
        return self.classifier(combined)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", default=str(ROOT / "models"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sessions = [s for s in database.get_all_sessions() if s["emotion_label"]]
    if len(sessions) < 20:
        print("Need at least ~20 labeled sessions for the LSTM. "
              "Run generate_synthetic_data.py or collect more data first.")
        return

    sequences = [session_to_sequence(s["events"]) for s in sessions]
    labels_raw = [s["emotion_label"] for s in sessions]
    user_ids = np.array([s["user_id"] for s in sessions])

    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(gss.split(sequences, labels, user_ids))

    # fit the scaler ONLY on training keystrokes to avoid leakage
    train_stack = np.concatenate([sequences[i] for i in train_idx if len(sequences[i]) > 0], axis=0)
    scaler = StandardScaler().fit(train_stack)

    train_ds = KeystrokeSeqDataset([sequences[i] for i in train_idx], labels[train_idx], scaler)
    test_ds = KeystrokeSeqDataset([sequences[i] for i in test_idx], labels[test_idx], scaler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KeystrokeLSTM(num_classes=len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    print(f"Training LSTM on {len(train_ds)} sessions, testing on {len(test_ds)} "
          f"({len(le.classes_)} classes), device={device}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        avg_loss = total_loss / len(train_ds)

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, lengths, y in test_loader:
                    x, lengths, y = x.to(device), lengths.to(device), y.to(device)
                    preds = model(x, lengths).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += len(y)
            print(f"  epoch {epoch:3d}/{args.epochs}  train_loss={avg_loss:.4f}  test_acc={correct/total:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "lstm_model.pt")
    import joblib
    joblib.dump(scaler, out_dir / "lstm_scaler.joblib")
    joblib.dump(le, out_dir / "lstm_label_encoder.joblib")
    (out_dir / "lstm_config.json").write_text(json.dumps(
        {"max_seq_len": MAX_SEQ_LEN, "hidden_dim": 64, "num_classes": len(le.classes_)}, indent=2))
    print(f"\nSaved LSTM artifacts to {out_dir}/ "
          f"(lstm_model.pt, lstm_scaler.joblib, lstm_label_encoder.joblib, lstm_config.json)")


if __name__ == "__main__":
    main()
