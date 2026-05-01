import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# files
INPUT_CSV = "india_mandis_geocoded.csv"
OUTPUT_CSV = "india_mandis_cleaned.csv"

# max allowed distance from state centroid (km)
MAX_STATE_DISTANCE_KM = 400

# load csv
df = pd.read_csv(INPUT_CSV)

# normalize headers aggressively
df.columns = [
    str(c).replace("\xa0", "").strip().lower()
    for c in df.columns
]

print("Columns after normalization:", df.columns.tolist())

# explicit rename based on your schema
if "market" not in df.columns:
    raise RuntimeError(f"'market' column not found. Found: {df.columns.tolist()}")

df = df.rename(columns={
    "market": "mandi",
    "latitude": "lat",
    "longitude": "lon"
})

# hard assertions
assert "mandi" in df.columns
assert "lat" in df.columns
assert "lon" in df.columns
assert "state" in df.columns
assert "district" in df.columns

# text normalization
df["state"] = df["state"].astype(str).str.title().str.strip()
df["district"] = df["district"].astype(str).str.title().str.strip()
df["mandi"] = (
    df["mandi"]
    .astype(str)
    .str.upper()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# india bounds filter
df = df[
    df["lat"].between(6, 37) &
    df["lon"].between(68, 97)
].copy()

# haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# state centroids
state_centroids = (
    df.groupby("state")[["lat", "lon"]]
    .mean()
    .reset_index()
    .rename(columns={"lat": "state_lat", "lon": "state_lon"})
)

df = df.merge(state_centroids, on="state", how="left")

# distance from state centroid
df["dist_from_state_km"] = df.apply(
    lambda r: haversine(
        r["lat"], r["lon"],
        r["state_lat"], r["state_lon"]
    ),
    axis=1
)

# drop mandis too far from their state
df = df[df["dist_from_state_km"] <= MAX_STATE_DISTANCE_KM].copy()

# flag approximate locations
coord_counts = (
    df.groupby(["lat", "lon"])
    .size()
    .reset_index(name="reuse_count")
)

df = df.merge(coord_counts, on=["lat", "lon"], how="left")

df["location_quality"] = np.where(
    df["reuse_count"] > 3,
    "approx",
    "exact"
)

# stable mandi id
df["mandi_id"] = (
    df["state"] + "|" +
    df["district"] + "|" +
    df["mandi"]
).apply(lambda x: abs(hash(x)) % (10**10))

# final selection
df_clean = df[[
    "mandi_id",
    "state",
    "district",
    "mandi",
    "lat",
    "lon",
    "location_quality",
    "dist_from_state_km"
]].sort_values(["state", "district", "mandi"])

# save
df_clean.to_csv(OUTPUT_CSV, index=False)

print("Done")
print("Rows:", len(df_clean))
print(df_clean["location_quality"].value_counts())
print("Saved to:", OUTPUT_CSV)