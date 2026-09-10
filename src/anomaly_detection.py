import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler



# 1. LOAD CLUSTERED DATA


df = pd.read_csv(
    "data/processed/clustered_gps_data.csv"
)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)



# 2. SELECT FEATURES


features = [
    "lat",
    "lon",
    "speed",
    "distance",
    "acceleration"
]

X = df[features].copy()



# 3. HANDLE MISSING VALUES


X = X.fillna(0)



# 4. SCALE THE DATA


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# 5. CREATE ISOLATION FOREST MODEL

anomaly_model = IsolationForest(
    contamination=0.05,
    random_state=42
)



# 6. DETECT ANOMALIES


df["anomaly"] = anomaly_model.fit_predict(
    X_scaled
)

# -1 = Anomaly
#  1 = Normal

df["anomaly_label"] = df["anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})



# 7. COUNT RESULTS


print("\nAnomaly Detection Results:")

print(
    df["anomaly_label"].value_counts()
)


# ==============================
# 8. SAVE DATA
# ==============================

os.makedirs(
    "data/processed",
    exist_ok=True
)

output_path = (
    "data/processed/anomaly_gps_data.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ==============================
# 9. SAVE MODEL
# ==============================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    anomaly_model,
    "models/anomaly_model.pkl"
)

joblib.dump(
    scaler,
    "models/anomaly_scaler.pkl"
)


# ==============================
# 10. VISUALIZE ANOMALIES
# ==============================

normal_data = df[
    df["anomaly"] == 1
]

anomaly_data = df[
    df["anomaly"] == -1
]


plt.figure(figsize=(10, 7))

plt.scatter(
    normal_data["lon"],
    normal_data["lat"],
    alpha=0.3,
    label="Normal"
)

plt.scatter(
    anomaly_data["lon"],
    anomaly_data["lat"],
    marker="x",
    label="Anomaly"
)

plt.title("GPS Movement Anomaly Detection")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.legend()

plt.savefig(
    "reports/anomaly_detection.png"
)

plt.show()


print("\nAnomaly Detection Completed Successfully!")
print("Saved dataset:", output_path)
print("Model saved in models folder.")