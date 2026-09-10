import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# 1. LOAD DATA
# ==============================

df = pd.read_csv("data/processed/featured_gps_data.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Create reports folder
os.makedirs("reports", exist_ok=True)


# ==============================
# 2. SPEED DISTRIBUTION
# ==============================

plt.figure(figsize=(8, 5))

plt.hist(df["speed"], bins=50)

plt.title("Speed Distribution")
plt.xlabel("Speed")
plt.ylabel("Frequency")

plt.savefig("reports/speed_distribution.png")
plt.show()


# ==============================
# 3. MOVEMENT BY HOUR
# ==============================

hourly_movement = df.groupby("hour").size()

plt.figure(figsize=(8, 5))

plt.plot(hourly_movement.index, hourly_movement.values, marker="o")

plt.title("Movement Frequency by Hour")
plt.xlabel("Hour")
plt.ylabel("Number of GPS Records")

plt.savefig("reports/movement_by_hour.png")
plt.show()


# ==============================
# 4. MOVEMENT BY WEEKDAY
# ==============================

weekday_movement = df.groupby("weekday").size()

plt.figure(figsize=(10, 5))

plt.bar(
    weekday_movement.index,
    weekday_movement.values
)

plt.title("Movement Frequency by Weekday")
plt.xlabel("Day")
plt.ylabel("Number of GPS Records")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("reports/movement_by_weekday.png")
plt.show()


# ==============================
# 5. GPS LOCATION SCATTER PLOT
# ==============================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["lon"],
    df["lat"],
    alpha=0.3
)

plt.title("GPS Movement Pattern")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.savefig("reports/gps_movement_pattern.png")

plt.show()


print("\nEDA Completed Successfully!")
print("Graphs saved in reports folder.")