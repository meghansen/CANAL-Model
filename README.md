# CANAL (Compartmental Amsterdam Neighborhood Agent-based modeL)

Agent-based model of influenza transmission in Amsterdam. Simulates disease spread through a synthetic population across four settings: households, schools, workplaces, and community.

---

## Requirements

- Python 3.9+

Install dependencies:

```bash
pip install numpy pandas scipy geopandas shapely sciris numba openpyxl
```

> **Note:** On the first run, Numba will compile the transmission kernels. This takes 30–60 seconds but only happens once.

---

## Data

All input files go in `./data/`. The following are required:

| File | Where to get it |
|------|-----------------|
| `WijkBuurtkaart_2020_v3/wijk_2020_v3.shp` | Download from [CBS](https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2020) and place the extracted folder directly in `./data/`. Shapefiles come as a set — keep all files (`.shp`, `.dbf`, `.prj`, `.shx`) together. |
| `population_counts_2021.xlsx` | AHTI/CBS population register. Columns: `WK_NAAM`, `Variable`, `Level`, `count`. |
| `hh_counts_2021.xlsx` | AHTI household composition data. Counts are number of *households*, not people. Columns: `WK_NAAM`, `Variable`, `Level`, `count`. |
| `amsterdam_travel_data.xlsx` | Age-specific commuting probabilities. Columns: `min_age`, `max_age`, `amsterdam`, `multiplier`. |
| `rate_params_amsterdam.xlsx` | State-transition rates. Columns: `state_i`, `state_j`, `rate_per_day`. |
| `prem-et-al_synthetic_contacts_2020.csv` | Prem et al. (2020) synthetic contact matrices. |

Your `./data/` folder should look like this:

```
data/
├── WijkBuurtkaart_2020_v3/
│   └── wijk_2020_v3.shp  (+ .dbf, .prj, .shx)
├── population_counts_2021.xlsx
├── hh_counts_2021.xlsx
├── amsterdam_travel_data.xlsx
├── rate_params_amsterdam.xlsx
└── prem-et-al_synthetic_contacts_2020.csv
```

---

## Running Simulations

```bash
python run_parameter_set.py \
    --betas 0.1 0.2 \
    --recovery_rates 0.05 0.1 \
    --ndays 100 \
    --replicates 5
```

This runs 5 replicates each of `(beta=0.1, recovery=0.05)` and `(beta=0.2, recovery=0.1)` for 100 days — 10 simulations in total. Betas and recovery rates are consumed as matched pairs, so the lists must be the same length.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--betas` | Yes | — | One or more transmission rates. |
| `--recovery_rates` | Yes | — | One or more recovery rates, paired with `--betas`. Mean infectious duration = 1 / recovery_rate days. |
| `--ndays` | No | 50 | Number of days per simulation. |
| `--replicates` | No | 1 | Independent replicates per parameter pair. Replicates differ only in their random seed, so variation between them reflects stochastic noise. |
| `--no-save-obj` | No | off | Skip saving the full `.obj` simulation file. Only CSVs are written. Useful for saving disk space when you only need the tabular outputs. |

### Population caching

On the **first run**, the synthetic population is built from the input data (takes several minutes) and saved to `./pop.obj`. All subsequent runs load this file directly, so only the first run is slow.

If you change any of the input data files or population parameters, delete `pop.obj` to force a rebuild.

---

## Outputs

Results are written to `./validation_outputs/`. For each (beta, recovery_rate, replicate):

| File | Description |
|------|-------------|
| `data_beta{b}_rec{r}_rep{k}.csv` | Per-agent epidemic and demographic data (see columns below). Always written. |
| `sim_beta{b}_rec{r}_rep{k}.obj` | Full simulation object. Only written unless `--no-save-obj` is passed. Load with `sc.load(path)` for access to the full state time-series and transmission chain. |
| `wijk_idx_to_naam.csv` | Wijk index-to-name mapping. Written once and shared across all runs. |

### `data_beta{b}_rec{r}_rep{k}.csv` columns

| Column | Description |
|--------|-------------|
| `id` | Agent index (0 to pop_size−1). |
| `age` | Age in years. |
| `wijk` | Wijk (neighbourhood) index. Join to `wijk_idx_to_naam.csv` for names. |
| `household` | Household ID. Agents sharing a value live in the same household. |
| `swstatus` | School/work status: 0=none, 1=primary school, 2=secondary school, 3=employed (firm), 4=self-employed. |
| `swindex` | School or firm ID. |
| `swwijk` | Wijk of the agent's school or workplace. |
| `exposure_day` | Day the agent was infected; `-1` = never infected. |
| `exposure_setting` | Setting of infection: 1=household, 2=school, 3=workplace, 4=community, 5=seeded. |
| `infector` | Agent index of the person who infected this agent; `-1` = unknown or seeded. |

### Loading a full simulation object

```python
import sciris as sc
import numpy as np

sim = sc.load("validation_outputs/sim_beta0.1_rec0.05_rep1.obj")

# Daily counts of susceptible / infectious / recovered
daily_counts = np.apply_along_axis(
    lambda col: np.bincount(col, minlength=3), axis=1, arr=sim.epidemic.result
)
# daily_counts shape: (ndays+1, 3) — columns are S, I, R counts per day
```

---

## Key Parameters

### Transmission rate (`--betas`)

Controls how readily the virus spreads per contact. Higher values produce larger, faster epidemics. The right value depends on the recovery rate and the setting modifiers below.

### Recovery rate (`--recovery_rates`)

The daily probability parameter for recovery. Mean infectious duration = `1 / recovery_rate` days. 

### Setting beta modifiers

Defined in `run_parameter_set.py` as `setting_beta_modifier = [4.0, 0.12, 1.5, 0.007]`:

| Index | Setting | Default | What it controls |
|-------|---------|---------|-----------------|
| 0 | Household | 4.0 | Amplifies transmission among household members (close, prolonged contact). |
| 1 | School | 0.12 | Transmission among pupils in the same school. |
| 2 | Workplace | 1.5 | Transmission among colleagues in the same firm. |
| 3 | Community | 0.007 | Cross-neighbourhood transmission via daily mobility. |

These are multiplied by `beta` to give the effective rate in each setting. To change them, edit `SIM_PARS` in `run_parameter_set.py`.

---

## File Overview

| File | Purpose |
|------|---------|
| `sim.py` | `Pop` class (builds population) and `Sim` class (runs epidemic). |
| `utils.py` | Transmission kernels, mobility model, firm assignment — called by `sim.py`. |
| `run_parameter_set.py` | Entry point. Parses arguments, loads/builds population, runs simulations, exports outputs. |
