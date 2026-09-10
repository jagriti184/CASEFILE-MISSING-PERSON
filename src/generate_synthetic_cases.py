import pandas as pd
import numpy as np
import os

# ==================================
# 1. LOAD PROCESSED GPS DATA
# ==================================

df = pd.read_csv(
    "data/processed/anomaly_gps_data.csv"
)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==================================
# 2. CONVERT TIME
# ==================================

df["time"] = pd.to_datetime(
    df["time"],
    errors="coerce"
)

df = df.dropna(subset=["time"])

# Sort trajectory data
df = df.sort_values(
    by=["user", "tid", "time"]
).reset_index(drop=True)


# ==================================
# 3. CREATE PREVIOUS LOCATION
# ==================================

df["previous_lat"] = df.groupby(
    ["user", "tid"]
)["lat"].shift(1)

df["previous_lon"] = df.groupby(
    ["user", "tid"]
)["lon"].shift(1)


# ==================================
# 4. CREATE TARGET LOCATION
# ==================================

df["target_lat"] = df.groupby(
    ["user", "tid"]
)["lat"].shift(-1)

df["target_lon"] = df.groupby(
    ["user", "tid"]
)["lon"].shift(-1)


# Remove rows where previous/next point missing
df = df.dropna(
    subset=[
        "previous_lat",
        "previous_lon",
        "target_lat",
        "target_lon"
    ]
)


# ==================================
# 5. CREATE AREA USING CLUSTERS
# ==================================

# Current cluster = current/last known area
df["last_area"] = (
    "Area_" + df["cluster"].astype(str)
)

# Next GPS point's cluster = Target Area
df["target_cluster"] = df.groupby(
    ["user", "tid"]
)["cluster"].shift(-1)

df = df.dropna(
    subset=["target_cluster"]
)

df["target_cluster"] = (
    df["target_cluster"].astype(int)
)

df["target_area"] = (
    "Area_" +
    df["target_cluster"].astype(str)
)


# ==================================
# 6. CREATE CASE DATASET
# ==================================

cases = pd.DataFrame()

cases["case_id"] = [
    f"MP-{i:05d}"
    for i in range(1, len(df) + 1)
]

cases["person_id"] = df["user"].values

# Age group is fictional/synthetic
np.random.seed(42)

age_groups = [
    "18-25",
    "26-35",
    "36-45",
    "46-60"
]

cases["age_group"] = np.random.choice(
    age_groups,
    size=len(df)
)

# Last known location
cases["last_latitude"] = df["lat"].values
cases["last_longitude"] = df["lon"].values

# Previous location
cases["previous_latitude"] = (
    df["previous_lat"].values
)

cases["previous_longitude"] = (
    df["previous_lon"].values
)

# Time information
cases["last_seen_time"] = df["time"].values

cases["hour"] = df["hour"].values
cases["day"] = df["weekday"].values

# Weather is fictional/synthetic
weather_options = [
    "Clear",
    "Cloudy",
    "Rain"
]

cases["weather"] = np.random.choice(
    weather_options,
    size=len(df)
)

# Movement features
cases["average_distance"] = (
    df["total_distance"].values
)

cases["average_speed"] = (
    df["average_speed"].values
)

cases["time_since_last_seen"] = (
    df["time_diff"].values
)

# Areas
cases["usual_area"] = (
    "Area_" +
    df["cluster"].astype(str).values
)

cases["previous_area"] = (
    "Area_" +
    df["cluster"].astype(str).values
)

# Target for ML
cases["target_area"] = (
    df["target_area"].values
)


# ==================================
# 7. SAVE SYNTHETIC DATASET
# ==================================

os.makedirs(
    "data/synthetic",
    exist_ok=True
)

output_path = (
    "data/synthetic/missing_person_cases.csv"
)

cases.to_csv(
    output_path,
    index=False
)


print("\nSynthetic Missing Person Cases Created!")
print("Total Cases:", len(cases))

print("\nColumns:")
print(cases.columns.tolist())

print("\nFirst 5 Rows:")
print(cases.head())

print("\nSaved at:")
print(output_path)