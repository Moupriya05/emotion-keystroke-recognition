"""
generate_synthetic_data.py
---------------------------
Generates plausible (NOT real) keystroke sessions for 5 emotions so you can
run the ENTIRE pipeline - build_training_data.py -> train.py -> app.py ->
demo - before you have real participants or have downloaded EmoSurv.

The per-emotion distributions below are deliberately caricatured from
patterns reported in the keystroke-dynamics/emotion literature (e.g. Epp,
Lippold & Mandryk 2011; Maalej & Kallel 2020 - EmoSurv):
    - Angry    : fast but erratic, more corrections
    - Happy    : fast, fluid, few errors
    - Calm     : medium speed, very low variance ("steady")
    - Sad      : slow, long hesitation pauses
    - Neutral  : baseline / average of everything

This is a STAND-IN for real data. Swap it out for ml/load_emosurv.py or your
own collected sessions (via data_collection/collector.html) as soon as you
can - a model trained only on this synthetic data has learned the shape of
this simulation, not real human behaviour.

Usage:
    python ml/generate_synthetic_data.py --users 30 --sessions-per-emotion 6
"""

from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend import database  # noqa: E402

PANGRAMS = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "sphinx of black quartz judge my vow",
]

# (dwell_mean, dwell_sd, flight_mean, flight_sd, backspace_prob, long_pause_prob, long_pause_extra_ms)
EMOTION_PROFILES = {
    "Angry":   dict(dwell=(70, 25), flight=(80, 55), backspace_p=0.10, pause_p=0.04, pause_extra=350),
    "Happy":   dict(dwell=(75, 12), flight=(90, 25), backspace_p=0.03, pause_p=0.02, pause_extra=200),
    "Calm":    dict(dwell=(100, 8), flight=(150, 15), backspace_p=0.02, pause_p=0.02, pause_extra=150),
    "Sad":     dict(dwell=(130, 25), flight=(220, 50), backspace_p=0.05, pause_p=0.12, pause_extra=900),
    "Neutral": dict(dwell=(95, 15), flight=(140, 35), backspace_p=0.04, pause_p=0.05, pause_extra=300),
}


def simulate_session(text: str, profile: dict) -> list[dict]:
    events = []
    t = 0.0
    dwell_mu, dwell_sd = profile["dwell"]
    flight_mu, flight_sd = profile["flight"]

    for ch in text:
        # occasional stray backspace-then-retype to simulate a correction
        if random.random() < profile["backspace_p"]:
            bs_dwell = max(20.0, random.gauss(dwell_mu, dwell_sd))
            events.append({"key": "Backspace", "code": "Backspace", "type": "down", "t": t})
            events.append({"key": "Backspace", "code": "Backspace", "type": "up", "t": t + bs_dwell})
            t += bs_dwell + max(10.0, random.gauss(flight_mu, flight_sd))

        # occasional hesitation before the next key
        if random.random() < profile["pause_p"]:
            t += profile["pause_extra"] + abs(random.gauss(0, profile["pause_extra"] / 3))

        dwell = max(15.0, random.gauss(dwell_mu, dwell_sd))
        code = "Space" if ch == " " else f"Key{ch.upper()}" if ch.isalpha() else f"Digit{ch}"
        events.append({"key": ch, "code": code, "type": "down", "t": t})
        events.append({"key": ch, "code": code, "type": "up", "t": t + dwell})

        flight = max(10.0, random.gauss(flight_mu, flight_sd))
        t += dwell + flight

    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=25, help="number of simulated participants")
    ap.add_argument("--sessions-per-emotion", type=int, default=5,
                     help="typing sessions per emotion, per user")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    database.init_db()

    total = 0
    for u in range(args.users):
        user_id = f"synthetic_user_{u:03d}"
        for emotion, profile in EMOTION_PROFILES.items():
            for _ in range(args.sessions_per_emotion):
                prompt = random.choice(PANGRAMS)
                events = simulate_session(prompt, profile)
                # simulate typed_text == prompt except when a backspace event
                # occurred (kept simple: we don't reconstruct exact retyped
                # text, accuracy_ratio will just be slightly noisy - fine for
                # a synthetic baseline).
                database.insert_session(
                    user_id=user_id, events=events, emotion_label=emotion,
                    text_type="fixed", prompt_text=prompt, typed_text=prompt,
                )
                total += 1

    print(f"Inserted {total} synthetic sessions for {args.users} users "
          f"x {len(EMOTION_PROFILES)} emotions x {args.sessions_per_emotion} sessions.")
    print(f"Database: {database.DB_PATH}")


if __name__ == "__main__":
    main()
