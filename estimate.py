"""Person B: pass-through and elasticity of California's Prop 56 tax.

Reads data/clean.csv (columns: state, year, price, packs_pc, state_tax, revenue).
Writes results/estimates.json and figs/packs.png.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# TODO: fill these in from the Decisions section of README.md (written by Person A).
# Price and pack sales use different year pairs.
PRICE_BEFORE, PRICE_AFTER = 2016, 2017
PACKS_BEFORE, PACKS_AFTER = 2016, 2018

TAX = 2.0  # Prop 56 increase, dollars per pack
CA = "California"
NON_STATES = {"United States", "Guam", "Puerto Rico", "Virgin Islands"}
CHART_YEARS = range(2005, 2020)
BASE_YEAR = 2016

assert None not in (PRICE_BEFORE, PRICE_AFTER, PACKS_BEFORE, PACKS_AFTER), (
    "Fill in the before/after years from README.md Decisions first."
)

df = pd.read_csv("data/clean.csv")
df = df[~df["state"].isin(NON_STATES)]

# ---- Control states: state_tax unchanged in every year 2015-2019 ----
window = df[(df["year"] >= 2015) & (df["year"] <= 2019)]
spread = window.groupby("state")["state_tax"].agg(lambda x: x.max() - x.min())
controls = [s for s in spread[spread < 1e-9].index if s != CA]

# Keep only controls with complete data in the years used and in the chart window
needed = df[df["state"].isin(controls)]
chart = needed[needed["year"].isin(CHART_YEARS)]
complete = chart.groupby("state")[["price", "packs_pc"]].apply(lambda g: g.notna().all().all())
controls = [s for s in controls if complete.get(s, False)]
print(f"{len(controls)} control states: {sorted(controls)}")


def value(state, year, col):
    return df[(df["state"] == state) & (df["year"] == year)][col].iloc[0]


def change(state, col, y0, y1, log=False):
    v0, v1 = value(state, y0, col), value(state, y1, col)
    return np.log(v1) - np.log(v0) if log else v1 - v0


# ---- Pass-through ----
dP_ca = change(CA, "price", PRICE_BEFORE, PRICE_AFTER)
dP_ctrl = np.mean([change(c, "price", PRICE_BEFORE, PRICE_AFTER) for c in controls])
passthrough_raw = dP_ca / TAX
passthrough = (dP_ca - dP_ctrl) / TAX

# ---- Elasticity ----
dlnQ_ca = change(CA, "packs_pc", PACKS_BEFORE, PACKS_AFTER, log=True)
dlnP_ca = change(CA, "price", PRICE_BEFORE, PRICE_AFTER, log=True)
dlnQ_ctrl = np.mean([change(c, "packs_pc", PACKS_BEFORE, PACKS_AFTER, log=True) for c in controls])
dlnP_ctrl = np.mean([change(c, "price", PRICE_BEFORE, PRICE_AFTER, log=True) for c in controls])
elasticity_naive = dlnQ_ca / dlnP_ca
elasticity = (dlnQ_ca - dlnQ_ctrl) / (dlnP_ca - dlnP_ctrl)

q0 = float(value(CA, PACKS_BEFORE, "packs_pc"))
p0 = float(value(CA, PRICE_BEFORE, "price"))

print(f"Price years {PRICE_BEFORE}->{PRICE_AFTER}, pack years {PACKS_BEFORE}->{PACKS_AFTER}")
print(f"CA price change: {dP_ca:.3f}   control mean price change: {dP_ctrl:.3f}")
print(f"CA dlnP: {dlnP_ca:.4f}   control mean dlnP: {dlnP_ctrl:.4f}")
print(f"CA dlnQ: {dlnQ_ca:.4f}   control mean dlnQ: {dlnQ_ctrl:.4f}")
print(f"Pass-through: raw {passthrough_raw:.3f}, DiD {passthrough:.3f}")
print(f"Elasticity:   naive {elasticity_naive:.3f}, DiD {elasticity:.3f}")
print(f"q0 = {q0:.3f} packs/person/yr, p0 = ${p0:.3f}")

os.makedirs("results", exist_ok=True)
with open("results/estimates.json", "w") as f:
    json.dump(
        {"q0": q0, "p0": p0, "passthrough": float(passthrough), "elasticity": float(elasticity)},
        f,
        indent=2,
    )

# ---- Chart: pack sales indexed to 2016 = 100 ----
pivot = df[df["year"].isin(CHART_YEARS)].pivot(index="year", columns="state", values="packs_pc")
index = 100 * pivot / pivot.loc[BASE_YEAR]
ca_idx = index[CA]
ctrl_idx = index[controls].mean(axis=1)  # index each state first, then average equally

gap = (ca_idx.loc[PACKS_AFTER] - ctrl_idx.loc[PACKS_AFTER])
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ca_idx.index, ca_idx.values, color="tab:red", lw=2, label="California")
ax.plot(ctrl_idx.index, ctrl_idx.values, color="tab:gray", lw=2, label=f"Control average ({len(controls)} states)")
ax.axvline(2017, color="black", ls="--", lw=1)
ax.text(2017.1, ax.get_ylim()[1], "Prop 56 (Apr 2017)", va="top", fontsize=9)
ax.set_title(
    f"After Prop 56, California pack sales fell {abs(gap):.0f} index points\n"
    f"more than in states with no tax change (by {PACKS_AFTER})"
)
ax.set_xlabel("Year")
ax.set_ylabel(f"Packs per capita ({BASE_YEAR} = 100)")
ax.legend()
fig.text(
    0.01, 0.005,
    f"Source: CDC, The Tax Burden on Tobacco, 1970-2019. Controls: {len(controls)} states "
    "with no state tax change 2015-19.",
    fontsize=7, ha="left",
)
os.makedirs("figs", exist_ok=True)
fig.savefig("figs/packs.png", dpi=150, bbox_inches="tight")
print("Saved results/estimates.json and figs/packs.png")
