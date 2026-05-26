"""
Fetches onion price data for Punjab from data.gov.in API,
paginating with offset. After every page it merges, sorts, and
saves the CSV so you always have an up-to-date file on disk.

Usage examples:
    # All available data
    python fetch_onion_data.py

    # Full year
    python fetch_onion_data.py --start 2022 --end 2022

    # Multiple years
    python fetch_onion_data.py --start 2020 --end 2023

    # Exact dates  (DD-MM-YYYY or YYYY-MM-DD both accepted)
    python fetch_onion_data.py --start 01-04-2021 --end 31-03-2022

    # Only a start (everything from that point onward)
    python fetch_onion_data.py --start 2023

Output:
    onion_punjab_<start>_to_<end>.csv  — merged, filtered & date-sorted dataset
    (updated after every page download)
"""

import argparse
import io
import sys
import time
from datetime import datetime

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
BASE_PARAMS = {
    "api-key": "579b464db66ec23bdd000001a05aa23878f24b706331f5d9cba84d49",
    "format":  "csv",
    "limit":   1000,
    "filters[State]":     "Punjab",
    "filters[Commodity]": "Onion",
}
DATE_COLUMN   = "Arrival_Date"
RETRY_LIMIT   = 3
RETRY_DELAY   = 5
REQUEST_PAUSE = 1
# ─────────────────────────────────────────────────────────────────────────────


# ── Date parsing helpers ──────────────────────────────────────────────────────

def parse_user_date(value: str, end_of_period: bool = False) -> pd.Timestamp:
    value = value.strip()
    if value.isdigit() and len(value) == 4:
        year = int(value)
        return pd.Timestamp(year=year, month=12, day=31) if end_of_period \
               else pd.Timestamp(year=year, month=1,  day=1)
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return pd.Timestamp(datetime.strptime(value, fmt))
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse date '{value}'. Use YYYY, DD-MM-YYYY, or YYYY-MM-DD."
    )


def build_output_filename(start, end) -> str:
    s = start.strftime("%Y%m%d") if start else "all"
    e = end.strftime("%Y%m%d")   if end   else "all"
    return f"onion_punjab_{s}_to_{e}.csv"


# ── API fetch ─────────────────────────────────────────────────────────────────

def fetch_page(session: requests.Session, offset: int) -> pd.DataFrame | None:
    params = {**BASE_PARAMS, "offset": offset}
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            print(f"  offset={offset:>6}  attempt={attempt}", end=" ... ", flush=True)
            resp = session.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            print(f"got {len(df)} rows", end="")
            return df
        except requests.exceptions.Timeout:
            print(f"timeout — retrying in {RETRY_DELAY}s")
        except requests.exceptions.HTTPError as exc:
            print(f"HTTP {exc.response.status_code} — {exc}")
            if exc.response.status_code in (400, 401, 403, 404):
                return None
        except requests.exceptions.RequestException as exc:
            print(f"request error: {exc}")
        except Exception as exc:
            print(f"unexpected error: {exc}")
        time.sleep(RETRY_DELAY)
    print(f"  Failed after {RETRY_LIMIT} attempts — stopping.")
    return None


# ── Save helper ───────────────────────────────────────────────────────────────

def save(combined: pd.DataFrame, date_col: str | None,
         start_date, end_date, out_file: str, page_num: int):
    """Dedup, filter, sort, and write the cumulative dataframe to disk."""

    combined = combined.drop_duplicates()

    if date_col and date_col in combined.columns:
        combined[date_col] = pd.to_datetime(
            combined[date_col], dayfirst=True, errors="coerce"
        )
        combined = combined.dropna(subset=[date_col])

        if start_date:
            combined = combined[combined[date_col] >= start_date]
        if end_date:
            combined = combined[combined[date_col] <= end_date]

        combined = combined.sort_values(date_col).reset_index(drop=True)

    combined.to_csv(out_file, index=False)

    date_info = ""
    if date_col and not combined.empty and date_col in combined.columns:
        date_info = (f"  |  {combined[date_col].min().date()}"
                     f" → {combined[date_col].max().date()}")

    print(f"  →  saved {len(combined)} rows to '{out_file}'{date_info}")

    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Punjab onion price data, saving after every page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start", metavar="DATE",
                        help="Start date: YYYY | DD-MM-YYYY | YYYY-MM-DD")
    parser.add_argument("--end",   metavar="DATE",
                        help="End date:   YYYY | DD-MM-YYYY | YYYY-MM-DD")
    args = parser.parse_args()

    start_date = end_date = None
    try:
        if args.start:
            start_date = parse_user_date(args.start, end_of_period=False)
        if args.end:
            end_date = parse_user_date(args.end,   end_of_period=True)
    except ValueError as exc:
        print(f"Error: {exc}"); sys.exit(1)

    if start_date and end_date and start_date > end_date:
        print("Error: --start must not be after --end."); sys.exit(1)

    out_file = build_output_filename(start_date, end_date)

    print("=" * 60)
    print("  Punjab Onion Price Fetcher  (incremental save mode)")
    print("=" * 60)
    if start_date or end_date:
        s_str = start_date.strftime("%d %b %Y") if start_date else "beginning"
        e_str = end_date.strftime("%d %b %Y")   if end_date   else "latest"
        print(f"  Date filter : {s_str}  →  {e_str}")
    else:
        print("  Date filter : none (all available data)")
    print(f"  Output file : {out_file}")
    print("=" * 60)
    print()

    cumulative   = pd.DataFrame()   # grows after every page
    date_col     = None              # resolved on first page
    offset       = 0
    page_num     = 0
    last_is_done = False

    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0 (data fetch script)"})

        while True:
            df = fetch_page(session, offset)

            # ── stopping conditions ───────────────────────────────────────
            if df is None:
                print("Stopping: unrecoverable fetch error.")
                break

            if df.empty:
                print(" — empty page, all records fetched.")
                last_is_done = True
                break

            page_num += 1
            is_last = len(df) < BASE_PARAMS["limit"]

            # ── resolve date column once ──────────────────────────────────
            if date_col is None:
                date_col = next(
                    (c for c in df.columns
                     if c.strip().lower() == DATE_COLUMN.lower()),
                    None,
                )
                if date_col is None:
                    print(f"\nWarning: date column '{DATE_COLUMN}' not found.")
                    print(f"Columns seen: {list(df.columns)}")
                    print("Saving without date filtering or sorting.\n")

            # ── accumulate & save ─────────────────────────────────────────
            cumulative = pd.concat([cumulative, df], ignore_index=True)
            cumulative = save(cumulative, date_col,
                              start_date, end_date, out_file, page_num)

            if is_last:
                print("Last page detected (partial page) — done.")
                last_is_done = True
                break

            offset += BASE_PARAMS["limit"]
            time.sleep(REQUEST_PAUSE)

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if cumulative.empty:
        print("  No data matched. File not written.")
    else:
        status = "Complete" if last_is_done else "Partial (stopped early)"
        print(f"  Status  : {status}")
        print(f"  Pages   : {page_num}")
        print(f"  Rows    : {len(cumulative)}")
        print(f"  Saved → : {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()