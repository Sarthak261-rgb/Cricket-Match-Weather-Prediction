"""
Exploratory Data Analysis — Cricket Match Weather Dataset
Generates publication-ready plots for the project report.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

DATA_PATH = "data/cricket_weather_data.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
PALETTE = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]


def load():
    df = pd.read_csv(DATA_PATH, parse_dates=["match_date"])
    df["month"] = df["match_date"].dt.month
    df["month_name"] = df["match_date"].dt.strftime("%b")
    return df


# ── 1. Outcome distribution by venue ─────────────────────────────────────────
def plot_outcome_by_venue(df):
    pivot = df.groupby(["venue", "match_outcome"]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(13, 6))
    pivot_pct.plot(kind="bar", stacked=True, ax=ax,
                   color=PALETTE[:len(pivot_pct.columns)], edgecolor="white", linewidth=0.5)
    ax.set_title("Match Outcome Distribution by Venue (%)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.legend(title="Outcome", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/eda_outcome_by_venue.png", bbox_inches="tight")
    plt.close()
    print("  Saved: eda_outcome_by_venue.png")


# ── 2. Weather feature correlations ──────────────────────────────────────────
def plot_correlation(df):
    cols = ["pre_temp_avg", "pre_humidity_avg", "pre_pressure_avg",
            "pre_wind_speed_avg", "pre_cloud_cover_avg", "pre_rainfall_total",
            "pre_visibility_avg", "start_temp", "start_humidity",
            "start_rainfall", "start_cloud_cover"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.4, annot_kws={"size": 8})
    ax.set_title("Weather Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/eda_correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  Saved: eda_correlation_heatmap.png")


# ── 3. Monthly rainfall patterns ─────────────────────────────────────────────
def plot_monthly_rainfall(df):
    month_rain = df.groupby("month")["pre_rainfall_total"].agg(["mean", "std"]).reset_index()
    month_rain.columns = ["month", "mean", "std"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(month_rain["month"], month_rain["mean"],
                  color=[PALETTE[i % len(PALETTE)] for i in range(12)],
                  edgecolor="white", linewidth=0.5, zorder=3)
    ax.errorbar(month_rain["month"], month_rain["mean"], yerr=month_rain["std"],
                fmt="none", color="black", capsize=4, linewidth=1.2, alpha=0.7)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_title("Average Pre-Match Rainfall by Month (All Venues)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Rainfall (mm)")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/eda_monthly_rainfall.png", bbox_inches="tight")
    plt.close()
    print("  Saved: eda_monthly_rainfall.png")


# ── 4. Temperature vs Humidity scatter by outcome ────────────────────────────
def plot_temp_humidity(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    outcomes = df["match_outcome"].unique()
    colors = dict(zip(outcomes, PALETTE))

    for outcome in outcomes:
        sub = df[df["match_outcome"] == outcome]
        ax.scatter(sub["pre_temp_avg"], sub["pre_humidity_avg"],
                   label=outcome, alpha=0.55, s=30,
                   color=colors[outcome], edgecolors="none")

    ax.set_xlabel("Pre-Match Temperature (°C)", fontsize=11)
    ax.set_ylabel("Pre-Match Humidity (%)", fontsize=11)
    ax.set_title("Temperature vs Humidity — Coloured by Match Outcome",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Outcome", framealpha=0.85)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/eda_temp_humidity_scatter.png", bbox_inches="tight")
    plt.close()
    print("  Saved: eda_temp_humidity_scatter.png")


# ── 5. Pitch condition vs cloud cover / humidity box plots ───────────────────
def plot_pitch_weather(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    order = sorted(df["pitch_condition"].unique())

    sns.boxplot(data=df, x="pitch_condition", y="pre_humidity_avg",
                order=order, palette=PALETTE[:3], ax=axes[0], linewidth=1.2)
    axes[0].set_title("Humidity Distribution by Pitch Condition", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Pre-Match Humidity (%)")

    sns.boxplot(data=df, x="pitch_condition", y="pre_cloud_cover_avg",
                order=order, palette=PALETTE[:3], ax=axes[1], linewidth=1.2)
    axes[1].set_title("Cloud Cover Distribution by Pitch Condition", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Cloud Cover (%)")

    for ax in axes:
        ax.yaxis.grid(True, alpha=0.3)

    plt.suptitle("How Weather Shapes Pitch Conditions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/eda_pitch_weather_box.png", bbox_inches="tight")
    plt.close()
    print("  Saved: eda_pitch_weather_box.png")


# ── 6. Rainfall distribution — KDE by outcome ────────────────────────────────
def plot_rainfall_kde(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    outcomes = df["match_outcome"].unique()

    for i, outcome in enumerate(outcomes):
        sub = df[df["match_outcome"] == outcome]["pre_rainfall_total"]
        sub = sub[sub < sub.quantile(0.97)]   # trim extreme outliers for cleaner plot
        sub.plot.kde(ax=ax, label=outcome, color=PALETTE[i], linewidth=2)

    ax.set_title("Pre-Match Rainfall Distribution by Outcome (KDE)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Density")
    ax.set_xlim(left=0)
    ax.legend(title="Outcome")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/eda_rainfall_kde.png", bbox_inches="tight")
    plt.close()
    print("  Saved: eda_rainfall_kde.png")


# ── 7. Feature importance proxy (variance across outcomes) ───────────────────
def plot_feature_importance_proxy(df):
    features = ["pre_temp_avg", "pre_humidity_avg", "pre_pressure_avg",
                "pre_wind_speed_avg", "pre_cloud_cover_avg", "pre_rainfall_total",
                "start_rainfall", "start_cloud_cover", "start_humidity", "pre_visibility_avg"]

    importance = {}
    for feat in features:
        group_means = df.groupby("match_outcome")[feat].mean()
        importance[feat] = group_means.std()

    imp_df = pd.Series(importance).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(imp_df))]
    imp_df.plot.barh(ax=ax, color=colors, edgecolor="white")
    ax.set_title("Feature Discriminability (Std of Group Means across Outcomes)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Std of Group Means")
    ax.xaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/eda_feature_importance_proxy.png", bbox_inches="tight")
    plt.close()
    print("  Saved: eda_feature_importance_proxy.png")


# ── 8. Pressure trend overview ───────────────────────────────────────────────
def plot_pressure_trend(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, venue in enumerate(df["venue"].unique()[:5]):
        sub = df[df["venue"] == venue].sort_values("match_date")
        ax.plot(sub["match_date"], sub["pre_pressure_avg"],
                alpha=0.6, linewidth=0.9, label=venue, color=PALETTE[i % len(PALETTE)])

    ax.set_title("Pre-Match Pressure Trends (Sample Venues)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_xlabel("")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/eda_pressure_trend.png", bbox_inches="tight")
    plt.close()
    print("  Saved: eda_pressure_trend.png")


def main():
    print("\n" + "="*50)
    print("  Exploratory Data Analysis")
    print("="*50)
    df = load()
    print(f"  Dataset: {len(df)} records, {df.shape[1]} columns\n")

    plot_outcome_by_venue(df)
    plot_correlation(df)
    plot_monthly_rainfall(df)
    plot_temp_humidity(df)
    plot_pitch_weather(df)
    plot_rainfall_kde(df)
    plot_feature_importance_proxy(df)
    plot_pressure_trend(df)

    print(f"\nAll EDA plots saved to: {PLOTS_DIR}/")
    print("="*50)


if __name__ == "__main__":
    main()
