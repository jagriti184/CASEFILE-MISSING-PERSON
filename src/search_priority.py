import pandas as pd
import numpy as np
import os


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv(
    "data/processed/anomaly_gps_data.csv"
)

print("=" * 60)
print("SEARCH PRIORITY ANALYSIS")
print("=" * 60)

print("Dataset Shape:", df.shape)


# =========================================================
# 2. CREATE AREA
# =========================================================

df["area"] = (
    "Area_" +
    df["cluster"].astype(str)
)


# =========================================================
# 3. CALCULATE AREA STATISTICS
# =========================================================

area_summary = df.groupby("area").agg(

    latitude=("lat", "mean"),

    longitude=("lon", "mean"),

    visit_count=("area", "count"),

    average_speed=("speed", "mean"),

    average_distance=("distance", "mean"),

    anomaly_count=("anomaly", lambda x:
                   (x == -1).sum())

).reset_index()


# =========================================================
# 4. NORMALIZATION FUNCTION
# =========================================================

def normalize(series):

    min_value = series.min()

    max_value = series.max()

    if max_value == min_value:

        return pd.Series(
            [1.0] * len(series),
            index=series.index
        )

    return (
        (series - min_value) /
        (max_value - min_value)
    )


# =========================================================
# 5. NORMALIZE FEATURES
# =========================================================

area_summary["visit_score"] = normalize(
    area_summary["visit_count"]
)

area_summary["anomaly_score"] = normalize(
    area_summary["anomaly_count"]
)

area_summary["movement_score"] = normalize(
    area_summary["average_distance"]
)


# =========================================================
# 6. CALCULATE SEARCH PRIORITY SCORE
# =========================================================

area_summary["priority_score"] = (

    0.50 * area_summary["visit_score"]

    + 0.30 * area_summary["movement_score"]

    + 0.20 * area_summary["anomaly_score"]
)


# Convert to 0-100
area_summary["priority_score"] = (
    area_summary["priority_score"] * 100
)


# =========================================================
# 7. PRIORITY LEVEL
# =========================================================

def get_priority(score):

    if score >= 70:
        return "High"

    elif score >= 40:
        return "Medium"

    else:
        return "Low"


area_summary["priority_level"] = (
    area_summary["priority_score"]
    .apply(get_priority)
)


# =========================================================
# 8. SORT BY PRIORITY
# =========================================================

area_summary = area_summary.sort_values(
    by="priority_score",
    ascending=False
).reset_index(drop=True)


# =========================================================
# 9. DISPLAY RESULTS
# =========================================================

print("\nArea Search Priority:")

print(
    area_summary[
        [
            "area",
            "latitude",
            "longitude",
            "visit_count",
            "anomaly_count",
            "priority_score",
            "priority_level"
        ]
    ].to_string(index=False)
)


# =========================================================
# 10. SAVE RESULTS
# =========================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

output_path = (
    "data/processed/search_priority.csv"
)

area_summary.to_csv(
    output_path,
    index=False
)


# =========================================================
# 11. SAVE REPORT
# =========================================================

os.makedirs(
    "reports",
    exist_ok=True
)

area_summary.to_csv(
    "reports/search_priority_report.csv",
    index=False
)


# =========================================================
# 12. FINAL RESULT
# =========================================================

print("\n" + "=" * 60)

print("SEARCH PRIORITY SCORE COMPLETED!")

print("=" * 60)

print(
    "Highest Priority Area:",
    area_summary.iloc[0]["area"]
)

print(
    "Priority Score:",
    round(
        area_summary.iloc[0]["priority_score"],
        2
    )
)

print(
    "Priority Level:",
    area_summary.iloc[0]["priority_level"]
)

print(
    "\nSaved at:",
    output_path
)