import fastf1
import pandas as pd
import requests
import matplotlib.pyplot as plt
import os

from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

from pathlib import Path

# Env/Secrets
load_dotenv()
API_KEY = os.getenv("OWM_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Missing OWM_API_KEY. Create a .env with OWM_API_KEY=... or export it in your shell."
    )

# Create Cache
CACHE_DIR = Path(__file__).with_name("f1_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# FastF1 Cache
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Load Monaco 2024 Session + Laps
session_2024 = fastf1.get_session(2024, 8, "R")
session_2024.load()

laps_2024 = session_2024.laps[
    ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
].copy()
laps_2024.dropna(inplace=True)

# Convert Lab/Sector Times To Seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate Sector Times By Driver
sector_times_2024 = (
    laps_2024.groupby("Driver")
    .agg(
        {
            "Sector1Time (s)": "mean",
            "Sector2Time (s)": "mean",
            "Sector3Time (s)": "mean",
        }
    )
    .reset_index()
)

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"]
    + sector_times_2024["Sector2Time (s)"]
    + sector_times_2024["Sector3Time (s)"]
)

# Clean Air Race Pace (True Lap Speed When Not Stuck In Traffic)
clean_air_race_pace = {
    "VER": 93.191067,
    "HAM": 94.020622,
    "LEC": 93.418667,
    "NOR": 93.428600,
    "ALO": 94.784333,
    "PIA": 93.232111,
    "RUS": 93.833378,
    "SAI": 94.497444,
    "STR": 95.318250,
    "HUL": 95.345455,
    "OCO": 95.682128,
}

# Monaco Qualification Data
qualifying_2025 = pd.DataFrame(
    {
        "Driver": [
            "VER",
            "NOR",
            "PIA",
            "RUS",
            "SAI",
            "ALB",
            "LEC",
            "OCO",
            "HAM",
            "STR",
            "GAS",
            "ALO",
            "HUL",
        ],
        "QualifyingTime (s)": [
            70.669,  # VER (1:10.669)
            69.954,  # NOR (1:09.954)
            70.129,  # PIA (1:10.129)
            None,  # RUS (DNF)
            71.362,  # SAI (1:11.362)
            71.213,  # ALB (1:11.213)
            70.063,  # LEC (1:10.063)
            70.942,  # OCO (1:10.942)
            70.382,  # HAM (1:10.382)
            72.563,  # STR (1:12.563)
            71.994,  # GAS (1:11.994)
            70.924,  # ALO (1:10.924)
            71.596,  # HUL (1:11.596)
        ],
    }
)

qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(
    clean_air_race_pace
)

# Weather (OpenWeatherMap API; Monaco Lat/Lon)
lat, lon = 43.7384, 7.4246
weather_url = "https://api.openweathermap.org/data/2.5/forecast"
params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}

try:
    r = requests.get(weather_url, params=params, timeout=20)
    data = r.json()
    if not r.ok or "list" not in data:
        raise RuntimeError(
            f"OWM error/shape: status={r.status_code}, body_keys={list(data.keys())}"
        )

    target_dt = pd.to_datetime("2025-05-25 13:00:00", utc=True)

    def parse(e):
        return pd.to_datetime(e["dt_txt"], utc=True)

    nearest = min(data["list"], key=lambda e: abs(parse(e) - target_dt))
    rain_probability = float(nearest.get("pop", 0.0))
    temperature = float(nearest.get("main", {}).get("temp", 20.0))

except Exception as e:
    print(f"[weather] fallback due to: {e}")
    rain_probability = 0.0
    temperature = 20.0

# If Heavy Rain Risk (>=0.75), Add 8% To Qualification Time; Else No Change.
qualifying_2025["WetPerformanceFactor"] = 1.08 if rain_probability >= 0.75 else 1.00

# Apply Weather Adjustment
qualifying_2025["QualifyingTime"] = (
    qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceFactor"]
)

# Constructor Points -> TeamPerformanceScore
team_points = {
    "McLaren": 279,
    "Mercedes": 147,
    "Red Bull": 131,
    "Williams": 51,
    "Ferrari": 114,
    "Haas": 20,
    "Aston Martin": 14,
    "Kick Sauber": 6,
    "Racing Bulls": 10,
    "Alpine": 7,
}

max_points = max(team_points.values())
team_performance_score = {
    team: points / max_points for team, points in team_points.items()
}

driver_to_team = {
    "VER": "Red Bull",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "LEC": "Ferrari",
    "RUS": "Mercedes",
    "HAM": "Mercedes",
    "GAS": "Alpine",
    "ALO": "Aston Martin",
    "TSU": "Racing Bulls",
    "SAI": "Ferrari",
    "HUL": "Kick Sauber",
    "OCO": "Alpine",
    "STR": "Aston Martin",
    "ALB": "Williams",
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(
    team_performance_score
)

# Average Position Change At Monaco (2024 Data)
average_position_change_monaco = {
    "VER": -1.0,
    "NOR": 1.0,
    "PIA": 0.2,
    "RUS": 0.5,
    "SAI": -0.3,
    "ALB": 0.8,
    "LEC": -1.5,
    "OCO": -0.2,
    "HAM": 0.3,
    "STR": 1.1,
    "GAS": -0.4,
    "ALO": -0.6,
    "HUL": 0.0,
}
qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(
    average_position_change_monaco
)

# Merge and Features
merged_data = qualifying_2025.merge(
    sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left"
)
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Only Keep Drivers That Existed In 2024 Monaco Dataset
valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
merged_data = merged_data[valid_drivers].reset_index(drop=True)

# Target: Average Lap Time Per Driver At Monaco 2024 (Seconds), Aligned To Merged_Data Order
y = (
    laps_2024.groupby("Driver")["LapTime (s)"]
    .mean()
    .reindex(merged_data["Driver"])
    .values
)

# Feature Matrix
X = merged_data[
    [
        "QualifyingTime",
        "RainProbability",
        "Temperature",
        "TeamPerformanceScore",
        "CleanAirRacePace (s)",
        "AveragePositionChange",
    ]
]

# Impute Missing Values (e.g., RUS Qualification Time  = None, Missing Clean-Air Pace For Some Drivers)
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train/Test Split + Model Training
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.3, random_state=37
)
model = GradientBoostingRegressor(
    n_estimators=120, learning_rate=0.1, max_depth=3, random_state=37
)
model.fit(X_train, y_train)

# Predictions For All Rows
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

# Results
final_results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
print("\n🏁 Predicted 2025 Monaco GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])

y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Delete """ To Enable Plots and Visualize Feature Importance
"""
# Plots (Visualizations)
plt.figure(figsize=(12, 8))
plt.scatter(
    final_results["CleanAirRacePace (s)"], final_results["PredictedRaceTime (s)"]
)
for i, row in final_results.iterrows():
    plt.annotate(
        row["Driver"],
        (row["CleanAirRacePace (s)"], row["PredictedRaceTime (s)"]),
        xytext=(5, 5),
        textcoords="offset points",
    )
plt.xlabel("Clean air race pace (s)")
plt.ylabel("Predicted race time (s)")
plt.title("Effect of clean air race pace on predicted race results")
plt.tight_layout()
plt.show()

feature_importance = model.feature_importances_
features = X.columns
plt.figure(figsize=(8, 5))
plt.barh(features, feature_importance)
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()
"""

# Podium Presentation
podium = final_results.loc[:2, ["Driver", "PredictedRaceTime (s)"]]
print("\n🏆 Predicted Top 3 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")
