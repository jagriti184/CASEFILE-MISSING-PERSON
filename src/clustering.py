import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================
# 1. LOAD FEATURED DATA
# ==============================

df = pd.read_csv(
    "data/processed/featured_gps_data.csv"
)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)


# ==============================
# 2. SELECT GPS FEATURES
# ==============================

X = df[["lat", "lon"]].copy()


# ==============================
# 3. SCALE THE DATA
# ==============================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==============================
# 4. ELBOW METHOD
# ==============================

inertia = []

k_values = range(2, 9)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


# Plot Elbow Graph

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.xticks(list(k_values))

plt.savefig(
    "reports/elbow_method.png"
)

plt.show()


# ==============================
# 5. FINAL K-MEANS MODEL
# ==============================

# Abhi K = 5 use kar rahe hain
# Elbow graph dekhkar later change kar sakte hain

optimal_k = 4

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(
    X_scaled
)


# ==============================
# 6. SAVE CLUSTERED DATA
# ==============================

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    "data/processed/clustered_gps_data.csv",
    index=False
)


# ==============================
# 7. SAVE MODEL AND SCALER
# ==============================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    kmeans,
    "models/clustering_model.pkl"
)

joblib.dump(
    scaler,
    "models/clustering_scaler.pkl"
)


# ==============================
# 8. VISUALIZE CLUSTERS
# ==============================

plt.figure(figsize=(10, 7))

plt.scatter(
    df["lon"],
    df["lat"],
    c=df["cluster"],
    alpha=0.5
)

# Cluster centers
centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

plt.scatter(
    centers[:, 1],
    centers[:, 0],
    marker="X",
    s=200
)

plt.title("GPS Movement Clusters")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.savefig(
    "reports/gps_clusters.png"
)

plt.show()


# ==============================
# 9. CLUSTER SUMMARY
# ==============================

cluster_summary = df.groupby(
    "cluster"
).agg(
    latitude=("lat", "mean"),
    longitude=("lon", "mean"),
    records=("cluster", "count"),
    average_speed=("speed", "mean")
).reset_index()

print("\nCluster Summary:")
print(cluster_summary)


cluster_summary.to_csv(
    "data/processed/cluster_summary.csv",
    index=False
)

print("\nClustering Completed Successfully!")
print("Optimal K currently used:", optimal_k)
print("Clustered dataset saved successfully!")