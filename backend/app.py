"""
app.py
------
FastAPI backend with two jobs:

1. DATA COLLECTION MODE
   POST /api/sessions   -> store a labeled typing session (used by
                            data_collection/collector.html)
   GET  /api/sessions    -> list stored sessions (debugging)

2. LIVE INFERENCE MODE
   POST /api/predict    -> take a raw keystroke log (no label) and return
                            the predicted emotion + class probabilities
                            (used by demo/live_demo.html)

Run with:
    uvicorn backend.app:app --reload --port 8000
(run this from the project root so the `backend` and `ml` packages resolve)
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `ml.features` and `backend.database` import cleanly

from backend import database
from ml import features as feat_mod

MODEL_PATH = ROOT / "models" / "best_model.joblib"
LABEL_ENCODER_PATH = ROOT / "models" / "label_encoder.joblib"
FEATURE_COLUMNS_PATH = ROOT / "models" / "feature_columns.json"

app = FastAPI(title="Keystroke Emotion Recognition API", version="1.0")

# Allow the demo HTML pages (opened as local files, or served from any host
# while you develop) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

database.init_db()

# Lazily-loaded model artifacts (None until the first prediction request,
# or until train.py has actually produced them).
_model = None
_label_encoder = None
_feature_columns: Optional[List[str]] = None


def _load_model_if_needed():
    global _model, _label_encoder, _feature_columns
    if _model is not None:
        return
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="No trained model found yet. Run ml/train.py first, "
                   "then restart the API.",
        )
    import joblib
    _model = joblib.load(MODEL_PATH)
    _label_encoder = joblib.load(LABEL_ENCODER_PATH)
    _feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())


class KeyEvent(BaseModel):
    key: Optional[str] = None
    code: Optional[str] = None
    type: str  # "down" | "up"
    t: float   # milliseconds


class SessionIn(BaseModel):
    user_id: str
    events: List[KeyEvent]
    emotion_label: Optional[str] = None
    text_type: str = "free"
    prompt_text: str = ""
    typed_text: str = ""


class PredictIn(BaseModel):
    events: List[KeyEvent]
    prompt_text: str = ""
    typed_text: str = ""


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/api/sessions")
def create_session(payload: SessionIn) -> Dict[str, Any]:
    events = [e.model_dump() for e in payload.events]
    if len(events) < 4:
        raise HTTPException(status_code=400, detail="Session too short to be useful (need >=2 keystrokes).")
    session_id = database.insert_session(
        user_id=payload.user_id,
        events=events,
        emotion_label=payload.emotion_label,
        text_type=payload.text_type,
        prompt_text=payload.prompt_text,
        typed_text=payload.typed_text,
    )
    return {"session_id": session_id, "num_events_stored": len(events)}


@app.get("/api/sessions")
def list_sessions() -> Dict[str, Any]:
    sessions = database.get_all_sessions()
    summary = [
        {"session_id": s["session_id"], "user_id": s["user_id"],
         "emotion_label": s["emotion_label"], "text_type": s["text_type"],
         "num_events": len(s["events"])}
        for s in sessions
    ]
    return {"count": len(summary), "sessions": summary}


@app.post("/api/predict")
def predict(payload: PredictIn) -> Dict[str, Any]:
    _load_model_if_needed()
    events = [e.model_dump() for e in payload.events]
    if len(events) < 4:
        raise HTTPException(status_code=400, detail="Too few keystrokes to classify reliably.")

    raw_feats = feat_mod.extract_features(
        events, prompt_text=payload.prompt_text or None, typed_text=payload.typed_text or None
    )
    # Align to the exact column order the model was trained on.
    x = [[raw_feats.get(col, 0.0) for col in _feature_columns]]

    proba = _model.predict_proba(x)[0]
    pred_idx = int(proba.argmax())
    pred_label = _label_encoder.inverse_transform([pred_idx])[0]

    probabilities = {
        _label_encoder.inverse_transform([i])[0]: float(p)
        for i, p in enumerate(proba)
    }
    return {
        "predicted_emotion": pred_label,
        "probabilities": probabilities,
        "num_keystrokes": raw_feats["num_keystrokes"],
        "typing_speed_wpm": round(raw_feats["typing_speed_wpm"], 1),
    }
