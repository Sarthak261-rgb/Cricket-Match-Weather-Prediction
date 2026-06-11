"""
LSTM-based Cricket Match Weather Prediction Model
Trains on historical weather data to forecast:
  1. Match outcome (Completed / Rain Interrupted / Abandoned / Delayed)
  2. Pitch condition (Flat/Batting, Humid/Swing, Wet/Green)
  3. Play possibility (binary)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import joblib
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

DATA_PATH  = "data/cricket_weather_data.csv"
MODEL_DIR  = "models"
PLOTS_DIR  = "plots"
RESULTS_DIR = "results"

SEQ_LEN   = 10   # look-back window (past matches at the venue)
BATCH_SIZE = 32
EPOCHS     = 80
LSTM_UNITS = 128

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Feature engineering ──────────────────────────────────────────────────────
WEATHER_FEATURES = [
    "pre_temp_avg", "pre_humidity_avg", "pre_pressure_avg",
    "pre_wind_speed_avg", "pre_cloud_cover_avg", "pre_rainfall_total",
    "pre_visibility_avg", "start_temp", "start_humidity", "start_pressure",
    "start_wind_speed", "start_cloud_cover", "start_rainfall"
]

def load_and_prepare(path: str):
    df = pd.read_csv(path, parse_dates=["match_date"])
    df.sort_values(["venue", "match_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Derived features ─────────────────────────────────────────────────────
    df["temp_humidity_index"] = (df["pre_temp_avg"] * df["pre_humidity_avg"]) / 100
    df["pressure_drop"]       = df.groupby("venue")["pre_pressure_avg"].diff().fillna(0)
    df["rain_flag"]           = (df["pre_rainfall_total"] > 0).astype(int)
    df["month"]               = df["match_date"].dt.month
    df["day_of_year"]         = df["match_date"].dt.dayofyear

    # Cyclical encoding for month/day
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Encode categoricals ──────────────────────────────────────────────────
    le_venue   = LabelEncoder().fit(df["venue"])
    le_type    = LabelEncoder().fit(df["match_type"])
    le_outcome = LabelEncoder().fit(df["match_outcome"])
    le_pitch   = LabelEncoder().fit(df["pitch_condition"])

    df["venue_enc"]   = le_venue.transform(df["venue"])
    df["type_enc"]    = le_type.transform(df["match_type"])
    df["outcome_enc"] = le_outcome.transform(df["match_outcome"])
    df["pitch_enc"]   = le_pitch.transform(df["pitch_condition"])

    encoders = {
        "venue": le_venue, "match_type": le_type,
        "outcome": le_outcome, "pitch": le_pitch
    }
    return df, encoders


def build_sequences(df, feature_cols, target_col, seq_len=SEQ_LEN):
    """Build LSTM sequences grouped by venue."""
    X_all, y_all = [], []

    for venue in df["venue"].unique():
        subset = df[df["venue"] == venue].reset_index(drop=True)
        if len(subset) < seq_len + 1:
            continue
        X_vals = subset[feature_cols].values
        y_vals = subset[target_col].values
        for i in range(seq_len, len(subset)):
            X_all.append(X_vals[i - seq_len : i])
            y_all.append(y_vals[i])

    return np.array(X_all, dtype=np.float32), np.array(y_all)


def build_lstm(input_shape, n_classes, dropout=0.3):
    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(n_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_history(history, task_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History — {task_name}", fontsize=14, fontweight="bold")

    ax1.plot(history.history["loss"],     label="Train Loss",     color="#e74c3c")
    ax1.plot(history.history["val_loss"], label="Val Loss",       color="#3498db")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Categorical Cross-Entropy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["accuracy"],     label="Train Acc",  color="#27ae60")
    ax2.plot(history.history["val_accuracy"], label="Val Acc",    color="#f39c12")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe = task_name.lower().replace(" ", "_")
    plt.savefig(f"{PLOTS_DIR}/training_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training plot → plots/training_{safe}.png")


def plot_confusion(y_true, y_pred, labels, task_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix — {task_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    safe = task_name.lower().replace(" ", "_")
    plt.savefig(f"{PLOTS_DIR}/confusion_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix → plots/confusion_{safe}.png")


def train_task(name, df, feature_cols, target_col, n_classes, label_names, scaler=None):
    print(f"\n{'='*55}")
    print(f"  Training Task: {name}")
    print(f"{'='*55}")

    X, y = build_sequences(df, feature_cols, target_col, SEQ_LEN)

    # Scale features
    n_samples, seq, n_feat = X.shape
    X_flat = X.reshape(-1, n_feat)
    if scaler is None:
        scaler = MinMaxScaler()
        X_flat = scaler.fit_transform(X_flat)
    else:
        X_flat = scaler.transform(X_flat)
    X = X_flat.reshape(n_samples, seq, n_feat)

    y_cat = to_categorical(y, num_classes=n_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train_f, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    print(f"  Shapes  → Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    model = build_lstm((SEQ_LEN, len(feature_cols)), n_classes)
    model.summary(print_fn=lambda x: None)   # suppress verbose output

    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-5, verbose=0),
        ModelCheckpoint(
            f"{MODEL_DIR}/{name.lower().replace(' ', '_')}_best.keras",
            save_best_only=True, monitor="val_accuracy", verbose=0
        )
    ]

    history = model.fit(
        X_train, y_train_f,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Test Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_history(history, name)
    plot_confusion(y_true, y_pred, label_names, name)

    # ── Save scaler ──────────────────────────────────────────────────────────
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_{name.lower().replace(' ', '_')}.pkl")

    metrics = {
        "task": name,
        "test_accuracy": round(float(acc), 4),
        "best_val_accuracy": round(float(max(history.history["val_accuracy"])), 4),
        "epochs_trained": len(history.history["loss"]),
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples":  int(X_test.shape[0])
    }
    return model, scaler, metrics


def main():
    print("\n" + "="*55)
    print("  Cricket Match Weather Prediction — LSTM Training")
    print("="*55)

    df, encoders = load_and_prepare(DATA_PATH)

    # Save encoders
    for k, enc in encoders.items():
        joblib.dump(enc, f"{MODEL_DIR}/encoder_{k}.pkl")
    print(f"\nLoaded {len(df)} match records across {df['venue'].nunique()} venues.")

    # ── Feature sets ─────────────────────────────────────────────────────────
    base_features = WEATHER_FEATURES + [
        "temp_humidity_index", "pressure_drop", "rain_flag",
        "month_sin", "month_cos", "venue_enc", "type_enc"
    ]

    all_metrics = []

    # ── Task 1: Match Outcome ─────────────────────────────────────────────────
    outcome_labels = list(encoders["outcome"].classes_)
    model_outcome, scaler_outcome, m1 = train_task(
        "Match Outcome",
        df, base_features, "outcome_enc",
        len(outcome_labels), outcome_labels
    )
    all_metrics.append(m1)

    # ── Task 2: Pitch Condition ───────────────────────────────────────────────
    pitch_labels = list(encoders["pitch"].classes_)
    model_pitch, scaler_pitch, m2 = train_task(
        "Pitch Condition",
        df, base_features, "pitch_enc",
        len(pitch_labels), pitch_labels
    )
    all_metrics.append(m2)

    # ── Task 3: Play Possible (binary) ────────────────────────────────────────
    df["play_bin_enc"] = df["play_possible"]
    model_play, scaler_play, m3 = train_task(
        "Play Possible",
        df, base_features, "play_bin_enc",
        2, ["No Play", "Play Possible"]
    )
    all_metrics.append(m3)

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(f"{RESULTS_DIR}/training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "="*55)
    print("  Summary")
    print("="*55)
    for m in all_metrics:
        print(f"  {m['task']:20s}  →  Test Acc: {m['test_accuracy']*100:.2f}%")
    print("\nAll models, scalers, and plots saved.")
    print("="*55)


if __name__ == "__main__":
    main()
