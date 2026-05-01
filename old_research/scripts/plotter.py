import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------- CONFIG ----------
CSV_FILE = "india_mandis_geocoded.csv"

# ---------- LOAD DATA ----------
df = pd.read_csv(CSV_FILE)

# Drop rows without coordinates
df = df.dropna(subset=["latitude", "longitude"])

lats = df["latitude"].astype(float)
lons = df["longitude"].astype(float)

print(f"Plotting {len(df)} mandi locations")

# ---------- MAP SETUP ----------
plt.figure(figsize=(10, 12))
ax = plt.axes(projection=ccrs.PlateCarree())

# India extent
ax.set_extent([68, 98, 6, 38], crs=ccrs.PlateCarree())

# Features
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.6)
ax.add_feature(cfeature.STATES, linewidth=0.4)
ax.add_feature(cfeature.LAND, facecolor="#f5f5f5")
ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")

# ---------- SCATTER ----------
ax.scatter(
    lons,
    lats,
    s=18,
    color="red",
    alpha=0.7,
    transform=ccrs.PlateCarree()
)

# ---------- TITLE ----------
ax.set_title(
    "APMC / Mandi Locations Across India",
    fontsize=14,
    pad=12
)

plt.savefig("india_mandis_map.png", dpi=300, bbox_inches="tight")
plt.show()