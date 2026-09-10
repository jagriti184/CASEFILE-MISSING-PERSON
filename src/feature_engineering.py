import pandas as pd
import os

# ==============================
# 1. LOAD CLEANED DATA
# ==============================

file_path = "data/processed/cleaned_gps_data.csv"

df = pd.read_csv(file_path)

print("Cleaned dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==============================
# 2. CONVERT TIME AGAIN
# ==============================

df["time"] = pd.to_datetime(df["time"])


# ==============================
# 3. SORT DATA
# ==============================

df = df.sort_values(
    by=["user", "tid", "time"]
).reset_index(drop=True)


# ==============================
# 4. HANDLE NUMERIC COLUMNS
# ==============================

numeric_columns = [
    "distance",
    "speed",
    "acceleration",
    "time_diff"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[numeric_columns] = df[numeric_columns].fillna(0)


# ==============================
# 5. TRAJECTORY FEATURES
# ==============================

# Total distance travelled in each trajectory
df["total_distance"] = df.groupby(
    ["user", "tid"]
)["distance"].transform("sum")


# Average speed of each trajectory
df["average_speed"] = df.groupby(
    ["user", "tid"]
)["speed"].transform("mean")


# Maximum speed of each trajectory
df["max_speed"] = df.groupby(
    ["user", "tid"]
)["speed"].transform("max")


# Average acceleration
df["average_acceleration"] = df.groupby(
    ["user", "tid"]
)["acceleration"].transform("mean")


# Total movement duration
df["movement_duration"] = df.groupby(
    ["user", "tid"]
)["time_diff"].transform("sum")


# Number of GPS points in trajectory
df["trajectory_points"] = df.groupby(
    ["user", "tid"]
)["lat"].transform("count")


# ==============================
# 6. VISIT FREQUENCY
# ==============================

df["visit_frequency"] = df.groupby(
    ["user", "lat", "lon"]
)["lat"].transform("count")


# ==============================
# 7. SAVE FEATURE ENGINEERED DATA
# ==============================

os.makedirs(
    "data/processed",
    exist_ok=True
)

output_path = "data/processed/featured_gps_data.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nFeature Engineering Completed Successfully!")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nNew Features Added:")

new_features = [
    "total_distance",
    "average_speed",
    "max_speed",
    "average_acceleration",
    "movement_duration",
    "trajectory_points",
    "visit_frequency"
]

print(new_features)

print("\nSaved at:")
print(output_path)

print("\nFirst 5 rows:")
print(df.head())