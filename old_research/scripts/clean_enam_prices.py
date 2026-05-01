import pandas as pd

# =====================================================
# LOAD RAW CSV
# =====================================================
INPUT_CSV = "enam_trade_details.csv"
OUTPUT_CSV = "enam_prices_cleaned.csv"

df = pd.read_csv(INPUT_CSV)

# =====================================================
# STANDARDIZE COLUMN NAMES (if needed)
# =====================================================
df.columns = [
    "state",
    "mandi",
    "commodity",
    "min_price",
    "modal_price",
    "max_price",
    "arrival_qty",
    "traded_qty",
    "unit",
    "date"
]

# =====================================================
# CLEAN NUMERIC COLUMNS
# =====================================================
num_cols = [
    "min_price",
    "modal_price",
    "max_price",
    "arrival_qty",
    "traded_qty"
]

for col in num_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =====================================================
# CLEAN DATE
# =====================================================
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")

# =====================================================
# NORMALIZE TEXT FIELDS
# =====================================================
df["state"] = df["state"].str.title().str.strip()
df["mandi"] = df["mandi"].str.upper().str.strip()
df["commodity"] = df["commodity"].str.upper().str.strip()

# =====================================================
# NORMALIZE UNIT
# =====================================================
df["unit"] = df["unit"].replace({"Qui": "Quintal"})

# =====================================================
# OPTIONAL: CONVERT QUINTAL → KG
# =====================================================
df["arrival_qty_kg"] = df["arrival_qty"] * 100
df["traded_qty_kg"] = df["traded_qty"] * 100

# =====================================================
# REMOVE INVALID ROWS
# =====================================================
df = df.dropna(subset=["modal_price", "date"])

# =====================================================
# SORT FOR TIME SERIES USE
# =====================================================
df = df.sort_values(["mandi", "date"]).reset_index(drop=True)

# =====================================================
# SAVE CLEAN DATA
# =====================================================
df.to_csv(OUTPUT_CSV, index=False)

print("✅ Cleaning complete")
print("Rows:", len(df))
print("Saved to:", OUTPUT_CSV)
