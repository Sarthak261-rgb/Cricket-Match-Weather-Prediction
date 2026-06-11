"""
Inference script — predict weather conditions for an upcoming cricket match.
Usage:
    python src/predict.py --venue "Mumbai (Wankhede)" --match_type ODI --date 2025-06-15
    python src/predict.py --interactive
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

MODEL_DIR  = "models"
DATA_PATH  = "data/cricket_weather_data.csv"
SEQ_LEN    = 10

WEATHER_FEATURES = [
    "pre_temp_avg", "pre_humidity_avg", "pre_pressure_avg",
    "pre_wind_speed_avg", "pre_cloud_cover_avg", "pre_rainfall_total",
    "pre_visibility_avg", "start_temp", "start_humidity", "start_pressure",
    "start_wind_speed", "start_cloud_cover", "start_rainfall"
]
DERIVED_FEATURES = [
    "temp_humidity_index", "pressure_drop", "rain_flag",
    "month_sin", "month_cos", "venue_enc", "type_enc"
]
ALL_FEATURES = WEATHER_FEATURES + DERIVED_FEATURES


def load_artifacts():
    """Load all saved models, scalers and encoders."""
    artifacts = {}
    for task in ["match_outcome", "pitch_condition", "play_possible"]:
        model_file = f"{MODEL_DIR}/{task}_best.keras"
        scaler_file = f"{MODEL_DIR}/scaler_{task}.pkl"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model not found: {model_file}\nRun src/train_model.py first.")
        artifacts[task] = {
            "model":  load_model(model_file),
            "scaler": joblib.load(scaler_file)
        }

    encoders = {}
    for name in ["venue", "match_type", "outcome", "pitch"]:
        encoders[name] = joblib.load(f"{MODEL_DIR}/encoder_{name}.pkl")

    return artifacts, encoders


def build_input_sequence(venue: str, match_type: str, date_str: str,
                          df_hist: pd.DataFrame, encoders: dict) -> np.ndarray:
    """
    Pull the last SEQ_LEN historical records for this venue,
    add derived features, and return a (1, SEQ_LEN, n_features) array.
    """
    venue_hist = df_hist[df_hist["venue"] == venue].sort_values("match_date").tail(SEQ_LEN)

    if len(venue_hist) < SEQ_LEN:
        # Pad with global averages if insufficient history
        pad_needed = SEQ_LEN - len(venue_hist)
        avg_row = df_hist[WEATHER_FEATURES].mean().to_dict()
        padding = pd.DataFrame([avg_row] * pad_needed)
        venue_hist = pd.concat([padding, venue_hist[WEATHER_FEATURES].reset_index(drop=True)])
        venue_hist = venue_hist.reset_index(drop=True)

    match_date = pd.to_datetime(date_str)
    venue_enc  = encoders["venue"].transform([venue])[0]
    type_enc   = encoders["match_type"].transform([match_type])[0]
    month      = match_date.month

    records = []
    for _, row in venue_hist.iterrows():
        temp  = float(row.get("pre_temp_avg", row.get("temperature_c", 25)))
        hum   = float(row.get("pre_humidity_avg", 60))
        press = float(row.get("pre_pressure_avg", 1013))
        rec = {
            "pre_temp_avg":        float(row.get("pre_temp_avg", temp)),
            "pre_humidity_avg":    hum,
            "pre_pressure_avg":    press,
            "pre_wind_speed_avg":  float(row.get("pre_wind_speed_avg", 12)),
            "pre_cloud_cover_avg": float(row.get("pre_cloud_cover_avg", 30)),
            "pre_rainfall_total":  float(row.get("pre_rainfall_total", 0)),
            "pre_visibility_avg":  float(row.get("pre_visibility_avg", 10)),
            "start_temp":          float(row.get("start_temp", temp)),
            "start_humidity":      float(row.get("start_humidity", hum)),
            "start_pressure":      float(row.get("start_pressure", press)),
            "start_wind_speed":    float(row.get("start_wind_speed", 12)),
            "start_cloud_cover":   float(row.get("start_cloud_cover", 30)),
            "start_rainfall":      float(row.get("start_rainfall", 0)),
            "temp_humidity_index": (temp * hum) / 100,
            "pressure_drop":       0.0,
            "rain_flag":           int(row.get("pre_rainfall_total", 0) > 0),
            "month_sin":           float(np.sin(2 * np.pi * month / 12)),
            "month_cos":           float(np.cos(2 * np.pi * month / 12)),
            "venue_enc":           venue_enc,
            "type_enc":            type_enc,
        }
        records.append(rec)

    X = np.array([[r[f] for f in ALL_FEATURES] for r in records], dtype=np.float32)
    return X.reshape(1, SEQ_LEN, len(ALL_FEATURES))


def predict(venue: str, match_type: str, date_str: str,
            artifacts: dict, encoders: dict, df_hist: pd.DataFrame):
    """Run all three predictions and return a results dict."""
    X = build_input_sequence(venue, match_type, date_str, df_hist, encoders)

    results = {}
    for task, info in artifacts.items():
        scaler = info["scaler"]
        model  = info["model"]

        n_s, seq, n_f = X.shape
        X_flat = X.reshape(-1, n_f)
        X_sc   = scaler.transform(X_flat).reshape(n_s, seq, n_f)

        probs  = model.predict(X_sc, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        if task == "match_outcome":
            labels = list(encoders["outcome"].classes_)
        elif task == "pitch_condition":
            labels = list(encoders["pitch"].classes_)
        else:
            labels = ["No Play", "Play Possible"]

        results[task] = {
            "prediction":   labels[pred_idx],
            "confidence":   round(float(probs[pred_idx]) * 100, 1),
            "probabilities": {lbl: round(float(p) * 100, 1) for lbl, p in zip(labels, probs)}
        }

    return results


def print_report(venue, match_type, date_str, results):
    print("\n" + "╔" + "═"*52 + "╗")
    print("║   CRICKET MATCH WEATHER PREDICTION REPORT" + " "*9 + "║")
    print("╠" + "═"*52 + "╣")
    print(f"║  Venue      : {venue:<37}║")
    print(f"║  Match Type : {match_type:<37}║")
    print(f"║  Date       : {date_str:<37}║")
    print("╠" + "═"*52 + "╣")

    icons = {"match_outcome": "🏏", "pitch_condition": "🌿", "play_possible": "☀️"}
    titles = {"match_outcome": "Match Outcome", "pitch_condition": "Pitch Condition", "play_possible": "Play Possible"}

    for task, res in results.items():
        print(f"║                                                    ║")
        print(f"║  {icons.get(task,'•')} {titles[task]:<48}║")
        print(f"║     Prediction : {res['prediction']:<35}║")
        print(f"║     Confidence : {res['confidence']}%{' '*(34-len(str(res['confidence'])))}║")
        print(f"║     Breakdown  :                                   ║")
        for lbl, pct in res["probabilities"].items():
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"║       {lbl:<16} {bar} {pct:5.1f}%    ║")

    print("╚" + "═"*52 + "╝\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Cricket Weather Predictor")
    parser.add_argument("--venue",      type=str, default=None)
    parser.add_argument("--match_type", type=str, default=None)
    parser.add_argument("--date",       type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def interactive_mode(artifacts, encoders, df_hist):
    print("\n=== Cricket Match Weather Predictor (Interactive) ===\n")
    venues = list(encoders["venue"].classes_)

    print("Available venues:")
    for i, v in enumerate(venues, 1):
        print(f"  {i}. {v}")

    idx = int(input("\nSelect venue number: ")) - 1
    venue = venues[idx]

    match_type = input("Match type (Test / ODI / T20): ").strip()
    date_str   = input("Match date (YYYY-MM-DD): ").strip()

    results = predict(venue, match_type, date_str, artifacts, encoders, df_hist)
    print_report(venue, match_type, date_str, results)


def main():
    args = parse_args()

    print("Loading models...")
    artifacts, encoders = load_artifacts()
    df_hist = pd.read_csv(DATA_PATH, parse_dates=["match_date"])
    print("Models loaded.\n")

    if args.interactive:
        interactive_mode(artifacts, encoders, df_hist)
    elif args.venue and args.match_type and args.date:
        results = predict(args.venue, args.match_type, args.date, artifacts, encoders, df_hist)
        print_report(args.venue, args.match_type, args.date, results)
    else:
        # Demo prediction
        venue      = "Mumbai (Wankhede)"
        match_type = "ODI"
        date_str   = "2025-03-15"
        print(f"Running demo prediction for {venue} — {match_type} on {date_str}")
        results = predict(venue, match_type, date_str, artifacts, encoders, df_hist)
        print_report(venue, match_type, date_str, results)


if __name__ == "__main__":
    main()
