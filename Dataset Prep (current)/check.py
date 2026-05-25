"""
Pipeline diagnostic script.
Run:  python check.py
Prints enough info to pinpoint what went wrong at each step.
"""

import os, glob, re
import numpy as np
import pandas as pd
import warnings

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
RAW_CSV_GLOB  = os.path.join(RAW_DIR, "Punjab_Onion_*.csv")
COORD_CSV     = os.path.join(RAW_DIR, "Punjab_mandi_coordinates.csv")

SEP = "-" * 60

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ─────────────────────────────────────────────────────────────
# 1. Raw files
# ─────────────────────────────────────────────────────────────
section("1. RAW CSV FILES")
files = sorted(glob.glob(RAW_CSV_GLOB))
print(f"Files found: {len(files)}")
for f in files:
    df = pd.read_csv(f)
    print(f"  {os.path.basename(f):35s}  rows={len(df):6,}  cols={list(df.columns)}")

# ─────────────────────────────────────────────────────────────
# 2. Date parsing — look at raw values before any conversion
# ─────────────────────────────────────────────────────────────
section("2. DATE COLUMN — RAW SAMPLES")
all_frames = [pd.read_csv(f) for f in files]
raw = pd.concat(all_frames, ignore_index=True)
print(f"Total rows (no cleaning): {len(raw):,}")

# detect the date column
date_col = None
for c in raw.columns:
    if c.lower() in ("t", "date", "arrival_date", "price_date"):
        date_col = c
        break
if date_col is None:
    # fallback: first object column
    date_col = raw.select_dtypes("object").columns[0]

print(f"\nDate column detected : '{date_col}'")
print("First 10 raw values  :", raw[date_col].dropna().head(10).tolist())
print("Last  10 raw values  :", raw[date_col].dropna().tail(10).tolist())
print("Unique formats sample:", raw[date_col].dropna().sample(min(15, len(raw))).tolist())

# try both dayfirst=True and dayfirst=False
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    parsed_dmy = pd.to_datetime(raw[date_col], dayfirst=True,  errors="coerce")
    parsed_mdy = pd.to_datetime(raw[date_col], dayfirst=False, errors="coerce")
print(f"\npd.to_datetime dayfirst=True  -> NaT count: {parsed_dmy.isna().sum():,} / {len(raw):,}")
print(f"pd.to_datetime dayfirst=False -> NaT count: {parsed_mdy.isna().sum():,} / {len(raw):,}")

best = parsed_dmy if parsed_dmy.isna().sum() <= parsed_mdy.isna().sum() else parsed_mdy
print(f"\nBest parse (fewer NaTs):")
print(f"  Min  : {best.min()}")
print(f"  Max  : {best.max()}")
print(f"  25%  : {best.quantile(0.25)}")
print(f"  50%  : {best.quantile(0.50)}")
print(f"  75%  : {best.quantile(0.75)}")

# show unparseable date strings
bad_dates = raw.loc[best.isna(), date_col].dropna().unique()
print(f"\nUnparseable date strings ({len(bad_dates)} unique):")
print(bad_dates[:20])

# ─────────────────────────────────────────────────────────────
# 3. Price column
# ─────────────────────────────────────────────────────────────
section("3. PRICE COLUMNS")
price_cols = [c for c in raw.columns if any(x in c.lower() for x in ("price","modal","min","max","p_"))]
print(f"Price-like columns: {price_cols}")
for c in price_cols:
    s = pd.to_numeric(raw[c], errors="coerce")
    print(f"  {c:20s}  non-null={s.notna().sum():,}  NaN={s.isna().sum():,}  "
          f"min={s.min():.0f}  max={s.max():.0f}  zeros={( s==0).sum():,}")

# ─────────────────────────────────────────────────────────────
# 4. Market / mandi name column
# ─────────────────────────────────────────────────────────────
section("4. MARKET NAME COLUMN")
name_col = None
for c in raw.columns:
    if "market" in c.lower() or "mandi" in c.lower():
        name_col = c
        break
if name_col:
    print(f"Column: '{name_col}'")
    print(f"Unique values: {raw[name_col].nunique()}")
    print("Sample:", raw[name_col].dropna().unique()[:20].tolist())
else:
    print("No market/mandi column found — check column names:", raw.columns.tolist())

# ─────────────────────────────────────────────────────────────
# 5. Coordinate CSV
# ─────────────────────────────────────────────────────────────
section("5. COORDINATE CSV")
if os.path.exists(COORD_CSV):
    coord = pd.read_csv(COORD_CSV)
    print(f"Rows: {len(coord)}")
    print(f"Columns: {coord.columns.tolist()}")
    print(coord.head(5).to_string())
    print(f"\nLat range: {coord['latitude'].min():.3f} to {coord['latitude'].max():.3f}")
    print(f"Lon range: {coord['longitude'].min():.3f} to {coord['longitude'].max():.3f}")
    out_of_punjab = coord[
        (coord["latitude"]  < 29.5) | (coord["latitude"]  > 32.5) |
        (coord["longitude"] < 73.8) | (coord["longitude"] > 76.9)
    ]
    print(f"\nCoords outside Punjab bbox: {len(out_of_punjab)}")
    if len(out_of_punjab):
        print(out_of_punjab.to_string())
else:
    print(f"FILE NOT FOUND: {COORD_CSV}")

# ─────────────────────────────────────────────────────────────
# 6. Row-loss audit through each processed step
# ─────────────────────────────────────────────────────────────
section("6. ROW COUNT AT EACH PIPELINE STEP")
steps = [
    ("step1_concatenated.csv", "After concat + dedup"),
    ("step2_with_coords.csv",  "After fuzzy match"),
    ("step3_punjab_only.csv",  "After spatial filter"),
    ("final_training_dataset.csv", "Final (with ERA5)"),
]
prev = len(raw)
print(f"  {'Raw (no cleaning)':40s}  {prev:>8,}")
for fname, label in steps:
    path = os.path.join(PROCESSED_DIR, fname)
    if os.path.exists(path):
        n = len(pd.read_csv(path))
        print(f"  {label:40s}  {n:>8,}   (lost {prev-n:,})")
        prev = n
    else:
        print(f"  {label:40s}  FILE NOT FOUND")

# ─────────────────────────────────────────────────────────────
# 7. ERA5 files
# ─────────────────────────────────────────────────────────────
section("7. ERA5 MONTHLY FILES")
era5_dir = os.path.join(PROCESSED_DIR, "era5_monthly")
if os.path.exists(era5_dir):
    nc_files = sorted(glob.glob(os.path.join(era5_dir, "*.nc")))
    print(f"Files on disk: {len(nc_files)}")
    if nc_files:
        sizes = [(os.path.basename(f), round(os.path.getsize(f)/1e6, 1)) for f in nc_files]
        # flag suspiciously small files (< 0.1 MB = likely corrupt/empty)
        for name, mb in sizes:
            flag = "  ⚠ SUSPICIOUSLY SMALL" if mb < 0.1 else ""
            print(f"  {name}  {mb:.1f} MB{flag}")
        years_present  = sorted({f.split("_")[1] for f, _ in sizes})
        months_present = len(nc_files)
        print(f"\nMonths downloaded : {months_present}")
        print(f"Months expected   : check years × 12")
else:
    print(f"Directory not found: {era5_dir}")

# ─────────────────────────────────────────────────────────────
# 8. Final dataset — per-year and per-mandi row counts
# ─────────────────────────────────────────────────────────────
final_path = os.path.join(PROCESSED_DIR, "final_training_dataset.csv")
if os.path.exists(final_path):
    section("8. FINAL DATASET — BREAKDOWN")
    fin = pd.read_csv(final_path)
    fin["t"] = pd.to_datetime(fin["t"], errors="coerce")
    fin["year"] = fin["t"].dt.year

    print("Rows per year:")
    print(fin["year"].value_counts().sort_index().to_string())

    print("\nRows per mandi (bottom 20 — sparse mandis):")
    print(fin["market_name"].value_counts().tail(20).to_string())

    print("\nNaN counts in final dataset:")
    print(fin.isna().sum()[fin.isna().sum() > 0].to_string())

print(f"\n{SEP}\nDiagnostic complete.\n{SEP}\n")
