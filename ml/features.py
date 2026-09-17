"""
features.py
------------
Turns a raw keystroke event log into a fixed-length numeric feature vector
that describes HOW someone typed (rhythm, hesitation, errors, speed) -
not WHAT they typed.

Raw event format (one dict per key-down / key-up), timestamps in milliseconds
as floats (e.g. from JS performance.now(), or Python time.perf_counter()*1000):

    {"key": "h", "code": "KeyH", "type": "down", "t": 1234.5}
    {"key": "h", "code": "KeyH", "type": "up",   "t": 1310.2}

This module is intentionally dependency-light (only stdlib + numpy) so it can
be imported by the FastAPI backend for live inference AND by the offline
training scripts without dragging in sklearn/torch.
"""

from __future__ import annotations
import difflib
import statistics
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional

# Keys we treat as "correction" keys (frustration / error signal)
CORRECTION_CODES = {"Backspace", "Delete"}
# Keys we exclude from timing stats because they are usually held for
# non-typing reasons (modifiers) - but we still count their usage.
MODIFIER_CODES = {"ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight",
                   "AltLeft", "AltRight", "MetaLeft", "MetaRight", "CapsLock"}

# A flight/pause longer than this many ms counts as a "long pause"
# (hesitation). This is configurable per call.
DEFAULT_LONG_PAUSE_MS = 1000.0



def _pair_down_up(events: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Pair each key-down with the next key-up of the SAME `code`, using a
    per-code FIFO queue. This correctly handles fast/rollover typing where
    key2 goes down before key1 comes up.

    Returns a list of keystrokes: {"code": str, "down": float, "up": float}
    ordered by down-time.
    """
    events = sorted(events, key=lambda e: e["t"])
    pending = defaultdict(deque)
    keystrokes = []

    for e in events:
        code = e.get("code") or e.get("key")
        if e["type"] == "down":
            pending[code].append(e["t"])
        elif e["type"] == "up":
            if pending[code]:
                down_t = pending[code].popleft()
                keystrokes.append({"code": code, "down": down_t, "up": e["t"]})
            # else: stray key-up with no matching key-down -> ignore (can
            # happen if the log started mid key-press).

    keystrokes.sort(key=lambda k: k["down"])
    return keystrokes


def _stats(values: List[float], prefix: str) -> Dict[str, float]:
    """Mean / median / std / min / max of a list, safely handling <2 items."""
    if not values:
        return {f"{prefix}_mean": 0.0, f"{prefix}_median": 0.0,
                f"{prefix}_std": 0.0, f"{prefix}_min": 0.0, f"{prefix}_max": 0.0}
    return {
        f"{prefix}_mean": float(statistics.mean(values)),
        f"{prefix}_median": float(statistics.median(values)),
        f"{prefix}_std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        f"{prefix}_min": float(min(values)),
        f"{prefix}_max": float(max(values)),
    }


def extract_features(
    events: List[Dict[str, Any]],
    prompt_text: Optional[str] = None,
    typed_text: Optional[str] = None,
    long_pause_ms: float = DEFAULT_LONG_PAUSE_MS,
) -> Dict[str, float]:
    """
    Main entry point. Converts one typing session's raw event log into a
    flat dict of named numeric features.

    prompt_text / typed_text are optional - if you ran a FIXED-TEXT task
    (participant was asked to copy a given sentence) you can pass both and
    get an extra "accuracy" feature. For FREE-TEXT typing, leave them None.
    """
    keystrokes = _pair_down_up(events)
    n = len(keystrokes)

    feats: Dict[str, float] = {}
    feats["num_keystrokes"] = float(n)

    if n == 0:
        # Degenerate/empty session - return a zeroed vector with the right keys
        # so downstream code never has to special-case this.
        for prefix in ["dwell", "DD", "UD_flight", "UU", "DU",
                        "DD_tri", "DU_tri"]:
            feats.update(_stats([], prefix))
        feats.update({
            "session_duration_ms": 0.0, "typing_speed_cps": 0.0,
            "typing_speed_wpm": 0.0, "backspace_count": 0.0,
            "backspace_ratio": 0.0, "long_pause_count": 0.0,
            "long_pause_total_ms": 0.0, "pause_ratio": 0.0,
            "modifier_key_ratio": 0.0, "accuracy_ratio": 1.0,
        })
        return feats

    # ---- per-key dwell time (hold duration) -----------------------------
    dwell = [k["up"] - k["down"] for k in keystrokes]
    feats.update(_stats(dwell, "dwell"))

    # ---- digraph features (between consecutive keystrokes i, i+1) ------
    DD, UD_flight, UU, DU = [], [], [], []
    for i in range(n - 1):
        a, b = keystrokes[i], keystrokes[i + 1]
        DD.append(b["down"] - a["down"])          # down(i)   -> down(i+1)
        UD_flight.append(b["down"] - a["up"])      # up(i)     -> down(i+1)  ("flight time")
        UU.append(b["up"] - a["up"])               # up(i)     -> up(i+1)
        DU.append(b["up"] - a["down"])             # down(i)   -> up(i+1)
    feats.update(_stats(DD, "DD"))
    feats.update(_stats(UD_flight, "UD_flight"))
    feats.update(_stats(UU, "UU"))
    feats.update(_stats(DU, "DU"))

    # ---- trigraph features (span of 2, i.e. keystrokes i and i+2) ------
    DD_tri, DU_tri = [], []
    for i in range(n - 2):
        a, c = keystrokes[i], keystrokes[i + 2]
        DD_tri.append(c["down"] - a["down"])
        DU_tri.append(c["up"] - a["down"])
    feats.update(_stats(DD_tri, "DD_tri"))
    feats.update(_stats(DU_tri, "DU_tri"))

    # ---- session-level speed / timing ----------------------------------
    session_duration_ms = keystrokes[-1]["up"] - keystrokes[0]["down"]
    session_duration_ms = max(session_duration_ms, 1.0)  # avoid div-by-zero
    feats["session_duration_ms"] = float(session_duration_ms)
    feats["typing_speed_cps"] = n / (session_duration_ms / 1000.0)
    feats["typing_speed_wpm"] = (n / 5.0) / (session_duration_ms / 60000.0)

    # ---- corrections / errors (frustration signal) ----------------------
    backspaces = sum(1 for k in keystrokes if k["code"] in CORRECTION_CODES)
    feats["backspace_count"] = float(backspaces)
    feats["backspace_ratio"] = backspaces / n

    # ---- hesitation / long pauses ---------------------------------------
    long_pauses = [f for f in UD_flight if f > long_pause_ms]
    feats["long_pause_count"] = float(len(long_pauses))
    feats["long_pause_total_ms"] = float(sum(long_pauses))
    feats["pause_ratio"] = feats["long_pause_total_ms"] / session_duration_ms

    # ---- modifier key usage ----------------------------------------------
    modifiers = sum(1 for k in keystrokes if k["code"] in MODIFIER_CODES)
    feats["modifier_key_ratio"] = modifiers / n

    # ---- accuracy vs. prompt (fixed-text tasks only) ---------------------
    if prompt_text is not None and typed_text is not None and len(prompt_text) > 0:
        ratio = difflib.SequenceMatcher(None, prompt_text, typed_text).ratio()
        feats["accuracy_ratio"] = float(ratio)
    else:
        feats["accuracy_ratio"] = 1.0

    return feats


def get_feature_names() -> List[str]:
    """
    Returns the canonical, ordered list of feature names by running the
    extractor once on a tiny dummy session. Used to guarantee train.py and
    app.py always build feature vectors in the exact same column order.
    """
    dummy_events = [
        {"key": "a", "code": "KeyA", "type": "down", "t": 0.0},
        {"key": "a", "code": "KeyA", "type": "up", "t": 80.0},
        {"key": "b", "code": "KeyB", "type": "down", "t": 150.0},
        {"key": "b", "code": "KeyB", "type": "up", "t": 230.0},
        {"key": "c", "code": "KeyC", "type": "down", "t": 300.0},
        {"key": "c", "code": "KeyC", "type": "up", "t": 370.0},
    ]
    return sorted(extract_features(dummy_events).keys())




if __name__ == "__main__":
    # Quick self-test when run directly: python ml/features.py
    demo_events = [
        {"key": "t", "code": "KeyT", "type": "down", "t": 0},
        {"key": "t", "code": "KeyT", "type": "up", "t": 90},
        {"key": "h", "code": "KeyH", "type": "down", "t": 140},
        {"key": "h", "code": "KeyH", "type": "up", "t": 210},
        {"key": "e", "code": "KeyE", "type": "down", "t": 260},
        {"key": "e", "code": "KeyE", "type": "up", "t": 330},
    ]
    f = extract_features(demo_events, prompt_text="the", typed_text="the")
    print(f"Extracted {len(f)} features:")
    for k in sorted(f):
        print(f"  {k}: {f[k]:.3f}")
