import pandas as pd
import os

# ==============================
# 1. LOAD DATASET
# ==============================

file_path = "data/raw/geolife_train_filtered_df.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==============================
# 2. MISSING VALUES
# ==============================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df = df.dropna()


# ==============================
# 3. REMOVE DUPLICATES
# ==============================

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print("\nDuplicate Rows Removed:",
      before_duplicates - after_duplicates)


# ==============================
# 4. INVALID GPS COORDINATES
# ==============================

df = df[
    (df["lat"].between(-90, 90)) &
    (df["lon"].between(-180, 180))
]

print("\nAfter Removing Invalid GPS Coordinates:")
print(df.shape)


# ==============================
# 5. CONVERT TIME COLUMN
# ==============================

df["time"] = pd.to_datetime(
    df["time"],
    errors="coerce"
)

df = df.dropna(subset=["time"])


# ==============================
# 6. CREATE TIME FEATURES
# ==============================

df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["weekday"] = df["time"].dt.day_name()

# Monday=0, Tuesday=1 ... Sunday=6
df["weekday_number"] = df["time"].dt.weekday

# Weekend = 1, Weekday = 0
df["is_weekend"] = (
    df["weekday_number"] >= 5
).astype(int)


# ==============================
# 7. SORT DATA
# ==============================

df = df.sort_values(
    by=["user", "tid", "time"]
)


# ==============================
# 8. CREATE OUTPUT FOLDER
# ==============================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ==============================
# 9. SAVE CLEANED DATA
# ==============================

output_path = "data/processed/cleaned_gps_data.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nPreprocessing Completed Successfully!")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nCleaned Dataset Saved At:")
print(output_path)

print("\nFinal Columns:")
print(df.columns.tolist())