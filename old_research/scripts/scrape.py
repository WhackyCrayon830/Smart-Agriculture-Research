from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv
import os

OUTPUT_FILE = "india_mandis.csv"

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://agmarknet.gov.in/viewmarketprofileinputpublic")

time.sleep(3)  # React hydration (DO NOT reduce)

# ---------- HELPERS ----------
def open_dropdown(label_text):
    driver.execute_script(f"""
    for (let lbl of document.querySelectorAll('label')) {{
        if (lbl.innerText.includes('{label_text}')) {{
            lbl.closest('div.flex').querySelector('div.cursor-pointer').click();
            break;
        }}
    }}
    """)

def read_dropdown_options():
    return driver.execute_script("""
    let s = document.querySelector("input[placeholder='Search...']");
    if (!s) return [];
    let opts = s.parentElement.querySelectorAll("div.cursor-pointer");
    return Array.from(opts).map(o => o.innerText.trim()).filter(t => t);
    """)

def read_selected_value(label_text):
    return driver.execute_script(f"""
    for (let lbl of document.querySelectorAll('label')) {{
        if (lbl.innerText.includes('{label_text}')) {{
            return lbl.closest('div.flex').querySelector('div.cursor-pointer').innerText.trim();
        }}
    }}
    return '';
    """)

def select_option_by_index(idx):
    driver.execute_script(f"""
    let s = document.querySelector("input[placeholder='Search...']");
    if (!s) return;
    let opts = s.parentElement.querySelectorAll("div.cursor-pointer");
    if (opts.length > {idx}) opts[{idx}].click();
    """)

# ---------- CSV SETUP ----------
file_exists = os.path.exists(OUTPUT_FILE)

csv_file = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
writer = csv.writer(csv_file)

if not file_exists:
    writer.writerow(["state", "district", "market"])
    csv_file.flush()

# ---------- READ STATES ----------
open_dropdown("State")
time.sleep(0.4)
states = read_dropdown_options()
open_dropdown("State")

current_state = read_selected_value("State")
print(f"Initial state: {current_state}")

# ---------- PROCESS INITIAL STATE ----------
time.sleep(1.0)

open_dropdown("District")
time.sleep(0.25)
districts = read_dropdown_options()
open_dropdown("District")

current_district = read_selected_value("District")
print(f"  Initial district: {current_district}")

time.sleep(0.8)

open_dropdown("Market")
time.sleep(0.25)
markets = read_dropdown_options()
open_dropdown("Market")

if not markets:
    writer.writerow([current_state, current_district, ""])
else:
    for m in markets:
        writer.writerow([current_state, current_district, m])

csv_file.flush()
print(f"  ✔ Saved {current_state} / {current_district}")

# ---------- LOOP REMAINING STATES ----------
for si in range(1, len(states)):
    open_dropdown("State")
    time.sleep(0.4)
    select_option_by_index(si)
    time.sleep(1.0)

    state = states[si]
    print(f"\nSTATE [{si+1}/{len(states)}]: {state}")

    open_dropdown("District")
    time.sleep(0.25)
    districts = read_dropdown_options()
    open_dropdown("District")

    # ---------- FIRST DISTRICT ----------
    open_dropdown("District")
    time.sleep(0.25)
    select_option_by_index(0)
    time.sleep(0.8)

    district = districts[0]

    open_dropdown("Market")
    time.sleep(0.25)
    markets = read_dropdown_options()
    open_dropdown("Market")

    if not markets:
        writer.writerow([state, district, ""])
    else:
        for m in markets:
            writer.writerow([state, district, m])

    csv_file.flush()
    print(f"  ✔ Saved {state} / {district}")

    # ---------- REMAINING DISTRICTS ----------
    for di in range(1, len(districts)):
        open_dropdown("District")
        time.sleep(0.25)
        select_option_by_index(di)
        time.sleep(0.8)

        district = districts[di]
        print(f"  DISTRICT [{di+1}/{len(districts)}]: {district}")

        open_dropdown("Market")
        time.sleep(0.25)
        markets = read_dropdown_options()
        open_dropdown("Market")

        if not markets:
            writer.writerow([state, district, ""])
        else:
            for m in markets:
                writer.writerow([state, district, m])

        csv_file.flush()
        print(f"    ✔ Saved {state} / {district}")

# ---------- CLEANUP ----------
csv_file.close()
driver.quit()

print("\n✅ DONE — CSV UPDATED LIVE")
print(f"File: {OUTPUT_FILE}")
