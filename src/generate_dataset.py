
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# Major cricket venues with their climate profiles
VENUES = {
    "Mumbai (Wankhede)": {
        "lat": 18.93, "lon": 72.83,
        "season_temps": {"winter": (20, 28), "summer": (28, 38), "monsoon": (25, 33)},
        "humidity_base": 75,
        "rain_months": [6, 7, 8, 9],
        "city": "Mumbai"
    },
    "Chennai (Chepauk)": {
        "lat": 13.06, "lon": 80.27,
        "season_temps": {"winter": (22, 30), "summer": (30, 40), "monsoon": (24, 34)},
        "humidity_base": 70,
        "rain_months": [10, 11, 12],
        "city": "Chennai"
    },
    "Kolkata (Eden Gardens)": {
        "lat": 22.56, "lon": 88.34,
        "season_temps": {"winter": (14, 22), "summer": (28, 38), "monsoon": (26, 35)},
        "humidity_base": 68,
        "rain_months": [6, 7, 8],
        "city": "Kolkata"
    },
    "Delhi (Feroz Shah Kotla)": {
        "lat": 28.64, "lon": 77.22,
        "season_temps": {"winter": (5, 18), "summer": (32, 45), "monsoon": (25, 36)},
        "humidity_base": 45,
        "rain_months": [7, 8],
        "city": "Delhi"
    },
    "Bengaluru (Chinnaswamy)": {
        "lat": 12.97, "lon": 77.60,
        "season_temps": {"winter": (16, 26), "summer": (22, 32), "monsoon": (18, 26)},
        "humidity_base": 60,
        "rain_months": [5, 6, 9, 10],
        "city": "Bengaluru"
    },
    "Lords (London)": {
        "lat": 51.52, "lon": -0.17,
        "season_temps": {"winter": (2, 8), "summer": (16, 26), "monsoon": (10, 18)},
        "humidity_base": 72,
        "rain_months": [10, 11, 12, 1, 2],
        "city": "London"
    },
    "The Oval (London)": {
        "lat": 51.48, "lon": -0.11,
        "season_temps": {"winter": (2, 9), "summer": (15, 25), "monsoon": (10, 18)},
        "humidity_base": 73,
        "rain_months": [10, 11, 12, 1, 2],
        "city": "London"
    },
    "MCG (Melbourne)": {
        "lat": -37.82, "lon": 144.98,
        "season_temps": {"winter": (8, 14), "summer": (22, 34), "monsoon": (12, 22)},
        "humidity_base": 55,
        "rain_months": [7, 8, 9],
        "city": "Melbourne"
    },
    "SCG (Sydney)": {
        "lat": -33.89, "lon": 151.22,
        "season_temps": {"winter": (10, 18), "summer": (22, 32), "monsoon": (15, 24)},
        "humidity_base": 60,
        "rain_months": [6, 7, 8],
        "city": "Sydney"
    },
    "Karachi (National Stadium)": {
        "lat": 24.87, "lon": 67.07,
        "season_temps": {"winter": (14, 26), "summer": (30, 42), "monsoon": (28, 36)},
        "humidity_base": 65,
        "rain_months": [7, 8],
        "city": "Karachi"
    },
    "Bridgetown (Kensington Oval)": {
        "lat": 13.08, "lon": -59.61,
        "season_temps": {"winter": (23, 29), "summer": (25, 32), "monsoon": (24, 30)},
        "humidity_base": 78,
        "rain_months": [6, 7, 8, 9, 10, 11],
        "city": "Bridgetown"
    },
    "Cape Town (Newlands)": {
        "lat": -33.93, "lon": 18.46,
        "season_temps": {"winter": (10, 17), "summer": (22, 32), "monsoon": (15, 24)},
        "humidity_base": 55,
        "rain_months": [5, 6, 7, 8],
        "city": "Cape Town"
    }
}

MATCH_TYPES = ["Test", "ODI", "T20"]
MATCH_OUTCOMES = ["Completed", "Rain Interrupted", "Abandoned", "Delayed Start"]


def get_season(month, venue_name):
    """Determine season based on month and hemisphere."""
    venue = VENUES[venue_name]
    is_southern = venue["lat"] < 0

    if is_southern:
        if month in [12, 1, 2]:
            return "summer"
        elif month in [6, 7, 8]:
            return "winter"
        else:
            return "monsoon"
    else:
        if month in [12, 1, 2]:
            return "winter"
        elif month in [4, 5, 6]:
            return "summer"
        else:
            return "monsoon"


def generate_weather_sequence(venue_name, date, n_hours=48):
    """Generate an hourly weather sequence for a match period."""
    venue = VENUES[venue_name]
    month = date.month
    season = get_season(month, venue_name)

    temp_range = venue["season_temps"][season]
    base_humidity = venue["humidity_base"]
    is_rain_month = month in venue["rain_months"]

    records = []
    base_temp = np.random.uniform(*temp_range)
    base_pressure = np.random.normal(1013, 8)
    base_wind = np.random.exponential(12)

    # Rain probability based on season
    rain_prob = 0.35 if is_rain_month else 0.08

    for hour in range(n_hours):
        current_dt = date + timedelta(hours=hour)

        # Diurnal temperature variation
        hour_of_day = current_dt.hour
        temp_offset = 4 * np.sin(np.pi * (hour_of_day - 6) / 12)
        temperature = base_temp + temp_offset + np.random.normal(0, 1.2)

        # Humidity inversely related to temperature during day
        humidity = base_humidity - (temp_offset * 1.5) + np.random.normal(0, 5)
        humidity = np.clip(humidity, 20, 98)

        # Pressure tends to drop before rain
        pressure = base_pressure + np.random.normal(0, 3)

        # Wind speed varies
        wind_speed = base_wind + np.random.normal(0, 3)
        wind_speed = max(0, wind_speed)
        wind_direction = np.random.randint(0, 360)

        # Cloud cover
        cloud_cover = np.random.beta(2, 3) * 100 if not is_rain_month else np.random.beta(3, 2) * 100

        # Rainfall
        is_raining = np.random.random() < rain_prob
        rainfall = np.random.exponential(5) if is_raining else 0

        # Visibility reduces with rain/humidity
        visibility = max(1, 15 - rainfall * 2 - (humidity - 70) * 0.1 + np.random.normal(0, 1))

        # Dew point
        dew_point = temperature - ((100 - humidity) / 5)

        records.append({
            "datetime": current_dt,
            "temperature_c": round(temperature, 2),
            "humidity_pct": round(humidity, 2),
            "pressure_hpa": round(pressure, 2),
            "wind_speed_kmh": round(wind_speed, 2),
            "wind_direction_deg": wind_direction,
            "cloud_cover_pct": round(cloud_cover, 2),
            "rainfall_mm": round(rainfall, 2),
            "visibility_km": round(visibility, 2),
            "dew_point_c": round(dew_point, 2)
        })

    return records


def generate_match_outcome(weather_records):
    """Determine match outcome based on weather patterns."""
    total_rain = sum(r["rainfall_mm"] for r in weather_records[:24])
    avg_cloud = np.mean([r["cloud_cover_pct"] for r in weather_records[:10]])
    max_temp = max(r["temperature_c"] for r in weather_records[:24])

    if total_rain > 50:
        return "Abandoned"
    elif total_rain > 20:
        return "Rain Interrupted"
    elif weather_records[0]["rainfall_mm"] > 5:
        return "Delayed Start"
    else:
        return "Completed"


def compute_pitch_condition(venue_name, weather_records):
    """Derive pitch condition label from weather context."""
    avg_humidity = np.mean([r["humidity_pct"] for r in weather_records[:6]])
    avg_temp = np.mean([r["temperature_c"] for r in weather_records[:6]])
    total_rain_prev = sum(r["rainfall_mm"] for r in weather_records[:3])

    if total_rain_prev > 10 or avg_humidity > 85:
        return "Wet/Green"
    elif avg_temp > 35 and avg_humidity < 40:
        return "Dry/Dusty"
    elif avg_humidity > 65:
        return "Humid/Swing"
    else:
        return "Flat/Batting"


def generate_dataset(n_matches=1200):
    """Generate the full dataset."""
    all_records = []
    venue_names = list(VENUES.keys())

    start_date = datetime(2015, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    for match_id in range(1, n_matches + 1):
        venue_name = random.choice(venue_names)
        match_date = start_date + timedelta(days=random.randint(0, date_range_days))
        match_type = random.choice(MATCH_TYPES)

        weather_seq = generate_weather_sequence(venue_name, match_date)
        outcome = generate_match_outcome(weather_seq)
        pitch_condition = compute_pitch_condition(venue_name, weather_seq)

        pre_match = weather_seq[:6]   # 6 hours before match
        match_start = weather_seq[6]  # match start hour

        record = {
            "match_id": match_id,
            "venue": venue_name,
            "match_date": match_date.strftime("%Y-%m-%d"),
            "match_type": match_type,
            "season": get_season(match_date.month, venue_name),
            # Pre-match averages (6 hours)
            "pre_temp_avg": round(np.mean([r["temperature_c"] for r in pre_match]), 2),
            "pre_humidity_avg": round(np.mean([r["humidity_pct"] for r in pre_match]), 2),
            "pre_pressure_avg": round(np.mean([r["pressure_hpa"] for r in pre_match]), 2),
            "pre_wind_speed_avg": round(np.mean([r["wind_speed_kmh"] for r in pre_match]), 2),
            "pre_cloud_cover_avg": round(np.mean([r["cloud_cover_pct"] for r in pre_match]), 2),
            "pre_rainfall_total": round(sum(r["rainfall_mm"] for r in pre_match), 2),
            "pre_visibility_avg": round(np.mean([r["visibility_km"] for r in pre_match]), 2),
            # Match start conditions
            "start_temp": match_start["temperature_c"],
            "start_humidity": match_start["humidity_pct"],
            "start_pressure": match_start["pressure_hpa"],
            "start_wind_speed": match_start["wind_speed_kmh"],
            "start_cloud_cover": match_start["cloud_cover_pct"],
            "start_rainfall": match_start["rainfall_mm"],
            # During-match weather (next 12 hours)
            "match_temp_avg": round(np.mean([r["temperature_c"] for r in weather_seq[6:18]]), 2),
            "match_humidity_avg": round(np.mean([r["humidity_pct"] for r in weather_seq[6:18]]), 2),
            "match_rainfall_total": round(sum(r["rainfall_mm"] for r in weather_seq[6:18]), 2),
            "match_cloud_avg": round(np.mean([r["cloud_cover_pct"] for r in weather_seq[6:18]]), 2),
            "match_wind_avg": round(np.mean([r["wind_speed_kmh"] for r in weather_seq[6:18]]), 2),
            # Targets
            "pitch_condition": pitch_condition,
            "match_outcome": outcome,
            "play_possible": 1 if outcome in ["Completed", "Delayed Start"] else 0,
        }
        all_records.append(record)

        if match_id % 100 == 0:
            print(f"  Generated {match_id} matches...")

    return pd.DataFrame(all_records)


if __name__ == "__main__":
    print("Generating cricket match weather dataset...")
    df = generate_dataset(n_matches=1200)
    df.to_csv("/home/claude/cricket-weather-predictor/data/cricket_weather_data.csv", index=False)
    print(f"\nDataset saved: {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"\nOutcome distribution:\n{df['match_outcome'].value_counts()}")
    print(f"\nPitch condition distribution:\n{df['pitch_condition'].value_counts()}")
    print(f"\nVenue distribution:\n{df['venue'].value_counts()}")
