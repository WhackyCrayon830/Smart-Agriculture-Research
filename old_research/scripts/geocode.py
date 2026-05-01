import pandas as pd
import requests
import os
import time
from requests.exceptions import RequestException
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ================= CONFIG =================
INPUT_CSV   = "india_mandis.csv"
SUCCESS_CSV = "india_mandis_geocoded.csv"
FAILED_CSV  = "india_mandis_failed.csv"

API_KEY = "c8fd4bc5d89c480586a217b006324ed5"
GEOAPIFY_URL = "https://api.geoapify.com/v1/geocode/search"

REQUEST_DELAY = 0.3
TIMEOUT = 10

# -------- SELECT MODE & METHOD ----------
MODE   = "normal"        # "normal" | "retry_failed"
METHOD = 1               # 1 = Geoapify, 2 = Nominatim
# ========================================

# ---------- LOAD FILES ----------
df_all = pd.read_csv(INPUT_CSV)

df_success = pd.read_csv(SUCCESS_CSV) if os.path.exists(SUCCESS_CSV) else pd.DataFrame()
df_failed  = pd.read_csv(FAILED_CSV)  if os.path.exists(FAILED_CSV)  else pd.DataFrame()

# ---------- BUILD RESUME SETS ----------
success_keys = set(
    zip(df_success["state"], df_success["district"], df_success["market"])
) if not df_success.empty else set()

failed_keys = set(
    zip(df_failed["state"], df_failed["district"], df_failed["market"])
) if not df_failed.empty else set()

# ---------- SELECT DATASET TO RUN ----------
if MODE == "normal":
    # Never retry known rows
    known = success_keys | failed_keys
    df_run = df_all[
        ~df_all.apply(
            lambda r: (r["state"], r["district"], r["market"]) in known,
            axis=1
        )
    ]
    print(f"▶ NORMAL MODE — processing {len(df_run)} new rows")

elif MODE == "retry_failed":
    # Only retry failed rows that are not already successful
    df_run = df_failed[
        ~df_failed.apply(
            lambda r: (r["state"], r["district"], r["market"]) in success_keys,
            axis=1
        )
    ]
    print(f"▶ RETRY FAILED MODE — retrying {len(df_run)} rows")

else:
    raise ValueError("MODE must be 'normal' or 'retry_failed'")

# ---------- HELPERS ----------
def build_address(state, district, market):
    return f"{state}, {district}, {market} mandi"

# ---------- GEOAPIFY ----------
def geocode_geoapify(address):
    try:
        r = requests.get(
            GEOAPIFY_URL,
            params={"text": address, "apiKey": API_KEY, "limit": 1},
            timeout=TIMEOUT
        )
        r.raise_for_status()
        data = r.json()
        if data.get("features"):
            p = data["features"][0]["properties"]
            return p["lat"], p["lon"]
    except RequestException:
        pass
    return None, None

# ---------- NOMINATIM ----------
osm = Nominatim(user_agent="mandi_resume_geocoder")
osm_geocode = RateLimiter(osm.geocode, min_delay_seconds=1.2)

def geocode_nominatim(address):
    try:
        loc = osm_geocode(address)
        if loc:
            return loc.latitude, loc.longitude
    except:
        pass
    return None, None

# ---------- METHOD DISPATCH ----------
def geocode(address):
    if METHOD == 1:
        return geocode_geoapify(address)
    elif METHOD == 2:
        return geocode_nominatim(address)
    else:
        raise ValueError("METHOD must be 1 (Geoapify) or 2 (Nominatim)")

# ---------- MAIN LOOP ----------
for _, r in df_run.iterrows():
    state, district, market = r["state"], r["district"], r["market"]

    if not isinstance(market, str) or not market.strip():
        continue

    address = build_address(state, district, market)
    lat, lon = geocode(address)

    key = (state, district, market)

    # ---------- SUCCESS ----------
    if lat is not None and lon is not None:
        row = pd.DataFrame([[state, district, market, address, lat, lon]],
            columns=["state","district","market","query_address","latitude","longitude"]
        )

        row.to_csv(
            SUCCESS_CSV,
            mode="a",
            header=not os.path.exists(SUCCESS_CSV),
            index=False
        )

        # Remove from failed if retry succeeded
        if MODE == "retry_failed" and not df_failed.empty:
            df_failed = df_failed[
                ~(
                    (df_failed["state"] == state) &
                    (df_failed["district"] == district) &
                    (df_failed["market"] == market)
                )
            ]
            df_failed.to_csv(FAILED_CSV, index=False)

        success_keys.add(key)
        print(f"✔ SUCCESS [{METHOD}] → {address}")

    # ---------- FAILED ----------
    else:
        if MODE == "normal":
            row = pd.DataFrame([[state, district, market, address, "failed"]],
                columns=["state","district","market","query_address","reason"]
            )
            row.to_csv(
                FAILED_CSV,
                mode="a",
                header=not os.path.exists(FAILED_CSV),
                index=False
            )
            failed_keys.add(key)

        print(f"✖ FAILED [{METHOD}] → {address}")

    time.sleep(REQUEST_DELAY)

print("\n✅ DONE — resume-safe, no known rows reprocessed")
print(f"Mode   : {MODE}")
print(f"Method : {METHOD}")
print(f"✔ Success file: {SUCCESS_CSV}")
print(f"✖ Failed file : {FAILED_CSV}")
