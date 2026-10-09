data/clean.csv: one row per state and year; columns state, year, price, packs_pc, state_tax, revenue.

results/estimates.json: keys q0, p0, passthrough, elasticity (California, before the tax).

results/model.json: the numbers in C's table.


| | Person A: Data | Person B: Estimates | Person C: Model |
|---|---|---|---|
| Owns | `get_data.py`, `data/clean.csv` | `estimate.py`, `results/estimates.json`, `figs/packs.png` | `model.py`, `results/model.json` |
| Branch | `data` | `estimates` | `model` |
| Reviews | B's pull request | C's pull request | A's pull request |
| Writes | `answers/A.md` | `answers/B.md` | `answers/C.md` |


## Decisions
