import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================================================
# 1. LOAD SYNTHETIC CASE DATA
# =========================================================

file_path = "data/synthetic/missing_person_cases.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("Original Dataset Shape:", df.shape)


# =========================================================
# 2. REDUCE DATASET SIZE FOR FASTER TRAINING
# =========================================================

MAX_ROWS = 50000

if len(df) > MAX_ROWS:

    df = df.sample(
        n=MAX_ROWS,
        random_state=42
    ).reset_index(drop=True)

print("Dataset Used for Training:", df.shape)


# =========================================================
# 3. CHECK TARGET DISTRIBUTION
# =========================================================

print("\nTarget Area Distribution:")

print(
    df["target_area"].value_counts()
)


# =========================================================
# 4. SELECT FEATURES
# =========================================================

features = [
    "last_latitude",
    "last_longitude",
    "hour",
    "average_distance",
    "average_speed",
    "time_since_last_seen",
    "age_group",
    "day",
    "weather",
    "usual_area",
    "previous_area"
]

target = "target_area"


X = df[features].copy()

y = df[target].copy()


# =========================================================
# 5. HANDLE MISSING VALUES
# =========================================================

numeric_features = [
    "last_latitude",
    "last_longitude",
    "hour",
    "average_distance",
    "average_speed",
    "time_since_last_seen"
]

for col in numeric_features:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

    X[col] = X[col].fillna(
        X[col].median()
    )


categorical_features = [
    "age_group",
    "day",
    "weather",
    "usual_area",
    "previous_area"
]

for col in categorical_features:

    X[col] = X[col].fillna("Unknown")

    X[col] = X[col].astype(str)


# =========================================================
# 6. ENCODE CATEGORICAL FEATURES
# =========================================================

feature_encoders = {}

for col in categorical_features:

    encoder = LabelEncoder()

    X[col] = encoder.fit_transform(
        X[col]
    )

    feature_encoders[col] = encoder


# =========================================================
# 7. ENCODE TARGET
# =========================================================

target_encoder = LabelEncoder()

y_encoded = target_encoder.fit_transform(
    y.astype(str)
)


print("\nTarget Classes:")

print(
    target_encoder.classes_
)


# =========================================================
# 8. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# =========================================================
# 9. FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# =========================================================
# 10. CREATE ML MODELS
# =========================================================

models = {

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),

    "Hist Gradient Boosting":
        HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=31,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        )
}


# =========================================================
# 11. TRAIN MODELS
# =========================================================

results = []

trained_models = {}

predictions_dict = {}


for model_name, model in models.items():

    print("\n")
    print("=" * 60)
    print("TRAINING:", model_name)
    print("=" * 60)

    # -----------------------------------------------------
    # KNN needs scaled data
    # -----------------------------------------------------

    if model_name == "KNN":

        model.fit(
            X_train_scaled,
            y_train
        )

        predictions = model.predict(
            X_test_scaled
        )

    else:

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )


    # -----------------------------------------------------
    # Calculate Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    # -----------------------------------------------------
    # Print Results
    # -----------------------------------------------------

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


    # -----------------------------------------------------
    # Store Results
    # -----------------------------------------------------

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1

    })


    trained_models[model_name] = model

    predictions_dict[model_name] = predictions


# =========================================================
# 12. MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


# =========================================================
# 13. SAVE MODEL COMPARISON
# =========================================================

os.makedirs(
    "reports",
    exist_ok=True
)

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)


# =========================================================
# 14. SELECT BEST MODEL
# =========================================================

best_index = results_df[
    "F1 Score"
].idxmax()


best_model_name = results_df.loc[
    best_index,
    "Model"
]


best_model = trained_models[
    best_model_name
]


best_predictions = predictions_dict[
    best_model_name
]


print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Best Model:",
    best_model_name
)


# =========================================================
# 15. SAVE BEST MODEL
# =========================================================

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    best_model,
    "models/location_model.pkl"
)


joblib.dump(
    scaler,
    "models/location_scaler.pkl"
)


joblib.dump(
    feature_encoders,
    "models/feature_encoders.pkl"
)


joblib.dump(
    target_encoder,
    "models/target_encoder.pkl"
)


# =========================================================
# 16. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)


plt.figure(
    figsize=(8, 6)
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_encoder.classes_
)


display.plot(
    xticks_rotation=45
)


plt.title(
    "Location Prediction - Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    "reports/location_prediction_confusion_matrix.png"
)


plt.show()


# =========================================================
# 17. FEATURE IMPORTANCE
# =========================================================

if best_model_name in [
    "Random Forest",
    "Hist Gradient Boosting"
]:

    importance = (
        best_model.feature_importances_
    )


    feature_importance = pd.DataFrame({

        "Feature": features,

        "Importance": importance

    })


    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
    )


    print("\n")
    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print(
        feature_importance.to_string(
            index=False
        )
    )


    # Save CSV

    feature_importance.to_csv(
        "reports/feature_importance.csv",
        index=False
    )


    # Plot

    plt.figure(
        figsize=(10, 6)
    )


    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )


    plt.xlabel(
        "Importance"
    )


    plt.ylabel(
        "Feature"
    )


    plt.title(
        "Location Prediction Feature Importance"
    )


    plt.gca().invert_yaxis()


    plt.tight_layout()


    plt.savefig(
        "reports/feature_importance.png"
    )


    plt.show()


# =========================================================
# 18. FINAL MESSAGE
# =========================================================

print("\n")
print("=" * 60)
print("LOCATION PREDICTION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "Best Model:",
    best_model_name
)

print(
    "Model saved at:",
    "models/location_model.pkl"
)

print(
    "Comparison saved at:",
    "reports/model_comparison.csv"
)

print(
    "Confusion Matrix saved at:",
    "reports/location_prediction_confusion_matrix.png"
)

print(
    "Feature Importance saved at:",
    "reports/feature_importance.csv"
)