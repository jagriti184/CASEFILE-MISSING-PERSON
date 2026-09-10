import pandas as pd
import numpy as np
import os
import joblib


# =========================================================
# 1. LOAD CLUSTERED GPS DATA
# =========================================================

file_path = "data/processed/clustered_gps_data.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("CLUSTERED GPS DATA LOADED")
print("=" * 60)

print("Dataset Shape:", df.shape)


# =========================================================
# 2. SORT DATA
# =========================================================

df["time"] = pd.to_datetime(
    df["time"],
    errors="coerce"
)

df = df.dropna(
    subset=["time"]
)

df = df.sort_values(
    by=["user", "tid", "time"]
).reset_index(drop=True)


# =========================================================
# 3. CREATE CURRENT AND NEXT AREA
# =========================================================

df["current_area"] = (
    "Area_" +
    df["cluster"].astype(str)
)

df["next_cluster"] = df.groupby(
    ["user", "tid"]
)["cluster"].shift(-1)

df = df.dropna(
    subset=["next_cluster"]
)

df["next_cluster"] = (
    df["next_cluster"].astype(int)
)

df["next_area"] = (
    "Area_" +
    df["next_cluster"].astype(str)
)


# =========================================================
# 4. CREATE TRANSITION TABLE
# =========================================================

transition_counts = pd.crosstab(
    df["current_area"],
    df["next_area"]
)

print("\nTransition Counts:")
print(transition_counts)


# =========================================================
# 5. CREATE TRANSITION PROBABILITY MATRIX
# =========================================================

transition_probability = (
    transition_counts.div(
        transition_counts.sum(axis=1),
        axis=0
    )
    .fillna(0)
)

print("\nTransition Probability Matrix:")
print(
    transition_probability.round(3)
)


# =========================================================
# 6. SAVE TRANSITION MATRIX
# =========================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

transition_probability.to_csv(
    "data/processed/transition_probability_matrix.csv"
)


# =========================================================
# 7. FUNCTION TO PREDICT NEXT AREA
# =========================================================

def predict_next_area(current_area):

    if current_area not in transition_probability.index:

        return None

    probabilities = (
        transition_probability
        .loc[current_area]
        .sort_values(
            ascending=False
        )
    )

    return probabilities.index[0]


# =========================================================
# 8. TEST ROUTE PREDICTION
# =========================================================

print("\n")
print("=" * 60)
print("NEXT AREA PREDICTIONS")
print("=" * 60)

areas = sorted(
    df["current_area"].unique()
)

for area in areas:

    prediction = predict_next_area(
        area
    )

    print(
        area,
        "->",
        prediction
    )


# =========================================================
# 9. CREATE ROUTE PREDICTION DATA
# =========================================================

route_predictions = []

for area in areas:

    if area in transition_probability.index:

        probabilities = (
            transition_probability
            .loc[area]
            .sort_values(
                ascending=False
            )
        )

        # Top 3 possible next areas
        top_3 = probabilities.head(3)

        for next_area, probability in top_3.items():

            route_predictions.append({

                "current_area": area,

                "predicted_next_area": next_area,

                "probability": probability

            })


route_prediction_df = pd.DataFrame(
    route_predictions
)


# =========================================================
# 10. SAVE ROUTE PREDICTIONS
# =========================================================

route_prediction_df.to_csv(
    "data/processed/route_predictions.csv",
    index=False
)


# =========================================================
# 11. SAVE MODEL
# =========================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    transition_probability,
    "models/route_transition_model.pkl"
)


# =========================================================
# 12. SHOW EXAMPLE ROUTE
# =========================================================

print("\n")
print("=" * 60)
print("EXAMPLE ROUTE PREDICTION")
print("=" * 60)

example_area = areas[0]

print(
    "Current Area:",
    example_area
)

print(
    "Predicted Next Area:",
    predict_next_area(
        example_area
    )
)


# =========================================================
# 13. FINAL MESSAGE
# =========================================================

print("\n")
print("=" * 60)
print("ROUTE PREDICTION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "Transition matrix saved at:"
)

print(
    "data/processed/transition_probability_matrix.csv"
)

print(
    "Route predictions saved at:"
)

print(
    "data/processed/route_predictions.csv"
)

print(
    "Route model saved at:"
)

print(
    "models/route_transition_model.pkl"
)