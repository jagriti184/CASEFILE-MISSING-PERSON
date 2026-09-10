import pandas as pd
import folium
import os


# =========================================================
# 1. LOAD DATA
# =========================================================

gps_file = "data/processed/anomaly_gps_data.csv"
priority_file = "data/processed/search_priority.csv"

df = pd.read_csv(gps_file)
priority = pd.read_csv(priority_file)

print("=" * 60)
print("INTERACTIVE MAP GENERATION")
print("=" * 60)

print("GPS Data:", df.shape)
print("Priority Areas:", priority.shape)


# =========================================================
# 2. CREATE MAP CENTER
# =========================================================

center_lat = df["lat"].mean()
center_lon = df["lon"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12
)


# =========================================================
# 3. GPS MOVEMENT POINTS
# =========================================================

# Sample points so map doesn't become too heavy
plot_df = df.sample(
    min(3000, len(df)),
    random_state=42
)

for _, row in plot_df.iterrows():

    folium.CircleMarker(
        location=[
            row["lat"],
            row["lon"]
        ],

        radius=2,

        popup=(
            f"User: {row['user']}<br>"
            f"Trajectory: {row['tid']}<br>"
            f"Speed: {row['speed']:.2f}"
        ),

        fill=True
    ).add_to(m)


# =========================================================
# 4. ANOMALOUS POINTS
# =========================================================

anomalies = df[
    df["anomaly"] == -1
]

# Limit anomaly points
anomalies = anomalies.head(500)


for _, row in anomalies.iterrows():

    folium.Marker(
        location=[
            row["lat"],
            row["lon"]
        ],

        popup=(
            f"<b>Anomalous Movement</b><br>"
            f"User: {row['user']}<br>"
            f"Speed: {row['speed']:.2f}<br>"
            f"Area: Area_{row['cluster']}"
        ),

        icon=folium.Icon(
            icon="warning",
            prefix="fa"
        )

    ).add_to(m)


# =========================================================
# 5. PRIORITY AREAS
# =========================================================

for _, row in priority.iterrows():

    score = row["priority_score"]

    level = row["priority_level"]

    popup_text = f"""
    <b>Search Priority Area</b><br>
    Area: {row['area']}<br>
    Priority Score: {score:.2f}<br>
    Priority Level: {level}<br>
    Visits: {row['visit_count']}<br>
    Anomalies: {row['anomaly_count']}
    """

    folium.Circle(
        location=[
            row["latitude"],
            row["longitude"]
        ],

        radius=500,

        popup=folium.Popup(
            popup_text,
            max_width=300
        ),

        fill=True
    ).add_to(m)


# =========================================================
# 6. ADD PRIORITY MARKERS
# =========================================================

for _, row in priority.iterrows():

    folium.Marker(
        location=[
            row["latitude"],
            row["longitude"]
        ],

        popup=(
            f"<b>{row['area']}</b><br>"
            f"Priority Score: "
            f"{row['priority_score']:.2f}<br>"
            f"Priority: "
            f"{row['priority_level']}"
        ),

        icon=folium.Icon(
            icon="search",
            prefix="fa"
        )

    ).add_to(m)


# =========================================================
# 7. ADD LAYER CONTROL
# =========================================================

folium.LayerControl().add_to(m)


# =========================================================
# 8. SAVE MAP
# =========================================================

os.makedirs(
    "reports",
    exist_ok=True
)

output_file = (
    "reports/movement_investigation_map.html"
)

m.save(output_file)


# =========================================================
# 9. FINAL OUTPUT
# =========================================================

print("\n" + "=" * 60)

print("INTERACTIVE MAP CREATED SUCCESSFULLY!")

print("=" * 60)

print(
    "Map saved at:"
)

print(output_file)