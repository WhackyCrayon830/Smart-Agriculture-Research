from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

# =====================================================
# CONFIG
# =====================================================
URL = "https://enam.gov.in/web/dashboard/trade-data"
OUTPUT_CSV = "enam_trade_details.csv"

# =====================================================
# CHROME SETUP
# =====================================================
options = Options()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 600)  # up to 10 minutes for manual steps

driver.get(URL)

print("\n================ MANUAL STEP REQUIRED ================")
print("1️⃣ Select FROM and TO dates")
print("2️⃣ Select commodity = POTATO")
print("3️⃣ Click the REFRESH button")
print("4️⃣ Wait until table rows appear")
print("=====================================================")
print("⏳ Script will resume automatically...\n")

time.sleep(10)  # Give user time to read instructions

# =====================================================
# WAIT UNTIL TABLE HAS AT LEAST ONE ROW
# =====================================================
wait.until(
    lambda d: len(
        d.find_elements(
            By.XPATH, "//table[contains(@class,'table')]//tbody/tr"
        )
    ) > 0
)

print("✅ Table data detected. Starting scrape...\n")

# =====================================================
# PAGINATION DROPDOWN
# =====================================================
page_select = Select(driver.find_element(By.ID, "min_max_no_of_list"))
total_pages = len(page_select.options)

print(f"📌 Total pages detected: {total_pages}")

# =====================================================
# PREPARE CSV (remove old file if exists)
# =====================================================
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

# =====================================================
# SCRAPE ALL PAGES (APPEND MODE)
# =====================================================
for page_idx in range(total_pages):
    print(f"📄 Scraping page {page_idx + 1}/{total_pages}")

    page_select.select_by_index(page_idx)
    time.sleep(0.5)

    table = driver.find_element(By.XPATH, "//table[contains(@class,'table')]")
    tbody = table.find_element(By.TAG_NAME, "tbody")
    rows = tbody.find_elements(By.TAG_NAME, "tr")

    page_rows = []

    for row in rows:
        cols = [c.text.strip() for c in row.find_elements(By.TAG_NAME, "td")]

        if len(cols) < 10:
            continue

        page_rows.append({
            "state": cols[0],
            "mandi": cols[1],
            "commodity": cols[2],
            "min_price": cols[3],
            "modal_price": cols[4],
            "max_price": cols[5],
            "arrival_qty": cols[6],
            "traded_qty": cols[7],
            "unit": cols[8],
            "date": cols[9]
        })

    # -------------------------------------------------
    # APPEND TO CSV AFTER EACH PAGE
    # -------------------------------------------------
    if page_rows:
        df_page = pd.DataFrame(page_rows)

        write_header = not os.path.exists(OUTPUT_CSV)

        df_page.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=write_header,
            index=False
        )

        print(f"   ✅ Saved {len(page_rows)} rows")

print(f"\n✅ DONE — data appended page-by-page")
print(f"💾 Final CSV: {OUTPUT_CSV}")

driver.quit()
