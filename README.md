# Cricket Match Weather Prediction using LSTM

Predicting match conditions before a cricket game starts can make a real difference — for captains deciding whether to bat or bowl, for broadcasters managing schedules, and for analysts building pre-match strategies. This project builds an end-to-end LSTM pipeline that takes historical weather sequences at a given venue and forecasts three things simultaneously: whether the match will be completed or disrupted, what the pitch surface is likely to behave like, and whether play is possible at all.

---

## What this project does

Given historical weather observations at a cricket venue (temperature, humidity, pressure, wind, cloud cover, rainfall, visibility), the model learns temporal patterns across past matches and produces three predictions for an upcoming fixture:

| Prediction | Classes |
|---|---|
| **Match Outcome** | Completed / Rain Interrupted / Abandoned / Delayed Start |
| **Pitch Condition** | Flat/Batting · Humid/Swing · Wet/Green · Dry/Dusty |
| **Play Possible** | Yes / No |

---

## Repository layout

```
cricket-weather-predictor/
│
├── data/
│   └── cricket_weather_data.csv      # 1 200 match records, 12 venues, 2015–2024
│
├── src/
│   ├── generate_dataset.py           # builds the synthetic-but-realistic dataset
│   ├── eda.py                        # 8 publication-ready exploratory plots
│   ├── train_model.py                # LSTM training for all three tasks
│   └── predict.py                    # inference — CLI and interactive modes
│
├── models/                           # saved .keras weights + scalers + encoders
├── plots/                            # training curves, confusion matrices, EDA charts
├── results/
│   └── training_metrics.json        # per-task accuracy summary
│
├── tests/
│   └── test_pipeline.py             # pytest suite (dataset, features, artefacts)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/cricket-weather-predictor.git
cd cricket-weather-predictor

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running the pipeline

### 1 — Generate the dataset
```bash
python src/generate_dataset.py
```
Produces `data/cricket_weather_data.csv` with 1 200 match records across 12 international venues spanning 2015–2024. Weather patterns are calibrated to each venue's actual climate (monsoon months, temperature ranges, humidity profiles).

### 2 — Exploratory analysis
```bash
python src/eda.py
```
Saves 8 charts to `plots/`:
- outcome distribution by venue
- feature correlation heatmap
- monthly rainfall patterns
- temperature–humidity scatter coloured by outcome
- pitch condition box plots
- rainfall KDE by outcome
- feature discriminability bar chart
- pressure trend by venue

### 3 — Train the LSTM models
```bash
python src/train_model.py
```
Trains three separate LSTM models (one per prediction task). Training uses early stopping and learning-rate reduction on plateau. All weights, scalers and label encoders are saved to `models/`.

Typical results after training:

| Task | Test Accuracy |
|---|---|
| Pitch Condition | ~85% |
| Play Possible | ~65% |
| Match Outcome | ~62% |

Match outcome is a genuinely hard problem even for humans — weather alone does not always determine interruptions. The pitch condition model performs well because humidity, cloud cover and recent rainfall are strong physical predictors of surface behaviour.

### 4 — Predict a match

**CLI mode**
```bash
python src/predict.py --venue "Mumbai (Wankhede)" --match_type ODI --date 2025-06-15
```

**Interactive mode**
```bash
python src/predict.py --interactive
```

**Demo (no arguments)**
```bash
python src/predict.py
```

### 5 — Run tests
```bash
pytest tests/ -v
```

---

## Model architecture

Each task uses the same two-layer LSTM architecture:

```
Input  (seq_len=10, n_features=20)
  └─ LSTM(128, return_sequences=True)
  └─ BatchNorm → Dropout(0.3)
  └─ LSTM(64)
  └─ BatchNorm → Dropout(0.3)
  └─ Dense(64, relu)
  └─ Dropout(0.2)
  └─ Dense(n_classes, softmax)
```

The look-back window (`seq_len=10`) means the model sees the last 10 matches played at that venue before making a forecast. This lets it pick up on venue-specific patterns — e.g. late-season pressure drops at Lords, or the humidity spike preceding Mumbai's monsoon months.

---

## Feature set

**Raw weather features (13)**
`pre_temp_avg`, `pre_humidity_avg`, `pre_pressure_avg`, `pre_wind_speed_avg`, `pre_cloud_cover_avg`, `pre_rainfall_total`, `pre_visibility_avg`, `start_temp`, `start_humidity`, `start_pressure`, `start_wind_speed`, `start_cloud_cover`, `start_rainfall`

**Derived features (7)**
`temp_humidity_index` (heat stress proxy), `pressure_drop` (trend over previous match), `rain_flag` (binary), `month_sin`, `month_cos` (cyclical encoding), `venue_enc`, `type_enc`

---

## Venues covered

| Venue | Country |
|---|---|
| Wankhede Stadium, Mumbai | India |
| MA Chidambaram Stadium, Chennai | India |
| Eden Gardens, Kolkata | India |
| Feroz Shah Kotla, Delhi | India |
| M. Chinnaswamy Stadium, Bengaluru | India |
| Lord's Cricket Ground, London | England |
| The Oval, London | England |
| Melbourne Cricket Ground | Australia |
| Sydney Cricket Ground | Australia |
| National Stadium, Karachi | Pakistan |
| Kensington Oval, Bridgetown | West Indies |
| Newlands, Cape Town | South Africa |

---

## Future improvements

- Pull real historical weather data from the Open-Meteo or ERA5 APIs
- Add actual match result data from Cricinfo / ESPNcricinfo scraping
- Experiment with Transformer-based temporal models
- Build a simple web interface using Streamlit
- Incorporate toss decision and team lineup as additional context features

---

## Tech stack

- **Python 3.10+**
- **TensorFlow / Keras** — LSTM model
- **scikit-learn** — preprocessing, evaluation metrics
- **pandas + NumPy** — data manipulation
- **Matplotlib + Seaborn** — visualisation
- **pytest** — testing

---

## License

MIT
