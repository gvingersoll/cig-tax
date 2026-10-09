"""Person A: download the CDC Tax Burden on Tobacco data and build data/clean.csv.

Saves the raw download unmodified to data/raw/, pivots it to one row per state
and year (columns state, year, price, packs_pc, state_tax, revenue), and prints
California's rows for 2014-2019.
"""
import os
import urllib.request

import pandas as pd

# Keep $limit=50000: without it the portal returns only the first 1,000 rows.
URL = "https://data.cdc.gov/resource/7nwe-3aj9.csv?$limit=50000"
RAW_PATH = "data/raw/tax_burden_on_tobacco.csv"
CLEAN_PATH = "data/clean.csv"
EXPECTED_ROWS = 15300

MEASURES = {
    "Average Cost per pack": "price",
    "Cigarette Consumption (Pack Sales Per Capita)": "packs_pc",
    "State Tax per pack": "state_tax",
    "Gross Cigarette Tax Revenue": "revenue",
}

# ---- Download, saved unmodified ----
os.makedirs("data/raw", exist_ok=True)
urllib.request.urlretrieve(URL, RAW_PATH)

raw = pd.read_csv(RAW_PATH)
print(f"Raw rows: {len(raw):,}")
assert len(raw) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} raw rows, got {len(raw):,}"

# ---- Pivot to one row per state and year ----
keep = raw[raw["submeasuredesc"].isin(MEASURES)]
clean = (
    keep.pivot_table(
        index=["locationdesc", "year"],
        columns="submeasuredesc",
        values="data_value",
        aggfunc="first",
    )
    .rename(columns=MEASURES)
    .reset_index()
    .rename(columns={"locationdesc": "state"})
)
clean = clean[["state", "year", "price", "packs_pc", "state_tax", "revenue"]]
clean = clean.sort_values(["state", "year"]).reset_index(drop=True)

assert not clean.duplicated(["state", "year"]).any(), "Duplicate state-year rows"
print(f"Clean rows: {len(clean):,} ({clean['state'].nunique()} states, "
      f"{clean['year'].min()}-{clean['year'].max()})")

clean.to_csv(CLEAN_PATH, index=False)
print(f"Saved {CLEAN_PATH}")

# ---- California, 2014-2019 ----
ca = clean[(clean["state"] == "California") & clean["year"].between(2014, 2019)]
print("\nCalifornia, 2014-2019:")
print(ca.to_string(index=False))
