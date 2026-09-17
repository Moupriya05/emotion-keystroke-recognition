# ⌨️ Emotion Recognition from Keystroke Dynamics

A full-stack ML system that detects a user's emotional state — **Happy / Sad / Angry / Calm / Neutral** — purely from *how* they type: keystroke timing, rhythm, and pause patterns. Not from *what* they type.

Built with **Python**, **FastAPI**, and **scikit-learn / XGBoost** — with a live browser demo and a data collection module.

---

## ✨ Features

- **Live Emotion Detection** — Type a sentence in the browser; get a predicted emotion + confidence chart in real time
- **Keystroke Feature Engineering** — 46 features per session: dwell time, flight time, digraph/trigraph latencies, typing speed, backspace rate, pause patterns
- **Multi-Model Training** — Trains and compares Logistic Regression, Random Forest, SVM, Gradient Boosting, and XGBoost; picks the best by macro-F1
- **Person-Independent Evaluation** — Uses group-aware cross-validation so the model can't cheat by recognising *who* is typing
- **Data Collection Module** — Browser-based session recorder for collecting real labeled typing data
- **REST API** — FastAPI backend with auto-generated Swagger docs at `/docs`
- **Public Dataset Support** — Loader for the EmoSurv dataset (124 participants, IEEE DataPort)
- **Optional LSTM Track** — Sequence model in PyTorch for a deep-learning component
- **Explainability** — SHAP feature-importance plots for reports and viva

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| API Framework | FastAPI + Uvicorn |
| Data / Features | pandas, numpy |
| Classical ML | scikit-learn, XGBoost, joblib |
| Deep Learning (optional) | PyTorch (LSTM) |
| Explainability (optional) | SHAP |
| Storage | SQLite (stdlib `sqlite3`) |
| Frontend | Vanilla HTML / CSS / JS |
| Data Validation | Pydantic v2 |

---

## 📐 Architecture

```
Browser (data_collection/collector.html)
        │  POST /api/sessions  (labeled typing session)
        ▼
┌─────────────────────────────────┐
│         FastAPI Backend         │
│         backend/app.py          │
│  POST /api/sessions             │
│  GET  /api/sessions             │
│  POST /api/predict   ◄──────────┼── Browser (demo/live_demo.html)
└────────────┬────────────────────┘
             │
     ┌───────▼────────┐
     │   SQLite DB    │  ← stores labeled sessions
     └───────┬────────┘
             │
     ┌───────▼────────────────────────┐
     │      ML Pipeline               │
     │  ml/build_training_data.py     │  sessions → feature CSV
     │  ml/features.py  (46 features) │  dwell, flight, digraph latencies
     │  ml/train.py                   │  cross-val → best model saved
     │  ml/evaluate.py (optional)     │  SHAP plots
     └───────┬────────────────────────┘
             │
     ┌───────▼────────────────────────┐
     │     models/                    │
     │  best_model.joblib             │
     │  feature_columns.json          │
     │  label_encoder.joblib          │
     │  confusion_matrix.png          │
     │  report.txt                    │
     └────────────────────────────────┘
```

---

## 📁 Project Structure

```
emotion-keystroke-recognition/
├── README.md
├── requirements.txt              # core dependencies
├── requirements-optional.txt     # torch (LSTM) + shap
├── backend/
│   ├── app.py                    # FastAPI: /api/sessions, /api/predict
│   └── database.py               # SQLite session storage
├── ml/
│   ├── features.py               # ← the heart of the project (46 features)
│   ├── generate_synthetic_data.py
│   ├── load_emosurv.py           # EmoSurv public dataset loader
│   ├── build_training_data.py    # DB sessions → feature CSV
│   ├── train.py                  # trains + compares all models
│   ├── train_lstm.py             # OPTIONAL: PyTorch sequence model
│   └── evaluate.py               # OPTIONAL: SHAP / feature importance
├── data_collection/
│   └── collector.html            # browser-based data collection
├── demo/
│   └── live_demo.html            # type → see predicted emotion live
├── data/                         # datasets + generated CSVs
└── models/                       # trained model artifacts (auto-generated)
```

---

## ⚡ Getting Started

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Moupriya05/emotion-keystroke-recognition.git
cd emotion-keystroke-recognition
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

# Optional: PyTorch LSTM + SHAP explainability
pip install -r requirements-optional.txt
```

### 4. Generate synthetic data (quickest way to test the full pipeline)

```bash
python ml/generate_synthetic_data.py --users 30 --sessions-per-emotion 6
```

### 5. Train the model

```bash
python ml/build_training_data.py   # sessions → feature CSV
python ml/train.py                 # trains all models, saves the best one
```

### 6. Start the API server

```bash
uvicorn backend.app:app --reload --port 8000
```

Swagger docs available at → **http://127.0.0.1:8000/docs**

### 7. Open the live demo

Double-click `demo/live_demo.html` in your file explorer while the server is running, type a sentence, and click **Detect my emotion**.

---

## 📖 API Reference

### Sessions

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/sessions` | Store a labeled typing session |
| GET | `/api/sessions` | List all stored sessions |

### Inference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Predict emotion from raw keystroke log |

### Example — Predict emotion

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {"code": "KeyH", "type": "keydown", "t": 0},
      {"code": "KeyH", "type": "keyup",   "t": 85},
      {"code": "KeyE", "type": "keydown", "t": 140},
      {"code": "KeyE", "type": "keyup",   "t": 210}
    ],
    "typed_text": "he",
    "prompt_text": "hello"
  }'
```

---

## 🧠 The Feature Set

46 features are extracted per typing session:

| Category | Features |
|---|---|
| Digraph latencies | Dwell, DD, UD/flight, UU, DU — mean/median/std/min/max per metric |
| Trigraph latencies | DD_tri, DU_tri — mean/median/std/min/max |
| Pace | `typing_speed_cps`, `typing_speed_wpm` |
| Error patterns | `backspace_count`, `backspace_ratio` |
| Pause patterns | `long_pause_count`, `long_pause_total_ms`, `pause_ratio` |
| Key usage | `modifier_key_ratio` |
| Session info | `num_keystrokes`, `session_duration_ms`, `accuracy_ratio` |

---

## 🗂️ Using a Real Dataset

The project includes a loader for the **EmoSurv** public dataset (124 participants):

```bash
python ml/load_emosurv.py \
  --fixed "Fixed Text Typing Dataset.csv" \
  --free  "Free Text Typing Dataset.csv" \
  --freq  "Frequency Dataset.csv" \
  --out data/training_data_emosurv.csv
```

Download from IEEE DataPort (free account required): https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using

---

## 🧪 Tested Results (Synthetic Data)

| Model | Test Accuracy |
|---|---|
| Logistic Regression | **98.3%** (best in CV) |
| LSTM (15 epochs) | 93.3% |

> ⚠️ Synthetic numbers are high by design — real human emotional data is noisier. Published results on real datasets typically range from 70–90% for binary tasks and lower for fine-grained multi-class setups.

---

## 👩‍💻 Author

**Moupriya Sarkar**  
B.Tech in Information Technology — Techno International New Town, Kolkata  
[GitHub](https://github.com/Moupriya05) · [LinkedIn](https://linkedin.com/in/moupriya-sarkar)

---

## 📄 License

MIT — feel free to use and build on this project.
