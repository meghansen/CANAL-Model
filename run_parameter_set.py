"""
run_parameter_set.py
--------------------
Command-line entry point for running paired-parameter epidemic simulations
using the Amsterdam agent-based model (neopatat_amsterdam_clean).

For each (beta, recovery_rate) pair, the simulation is run `--replicates`
times. Results are saved per replicate as:
    validation_outputs/sim_beta{b}_rec{r}_rep{k}.obj     -- full Sim object
    validation_outputs/data_beta{b}_rec{r}_rep{k}.csv    -- per-agent epidemic + demographic data
    validation_outputs/wijk_idx_to_naam.csv              -- wijk index-to-name mapping (written once)

Pass --no-save-obj to skip saving the full .obj file and only write CSVs.

Usage
-----
    python run_parameter_set.py \
        --betas 0.1 0.2 \
        --recovery_rates 0.05 0.1 \
        --ndays 100 \
        --replicates 5

    # CSV output only (no .obj files):
    python run_parameter_set.py --betas 0.1 --recovery_rates 0.05 --no-save-obj

This runs 5 replicates each of (beta=0.1, recovery=0.05) and (beta=0.2,
recovery=0.1), for 100 simulation days each. The number of betas and
recovery_rates must match.

Population
----------
If a cached population object exists at ./pop.obj it is loaded directly,
avoiding the expensive (~minutes) population construction step. Otherwise
a new population is built and saved for future runs.
"""

import os
import argparse

import numpy as np
import pandas as pd
import sciris as sc

import neopatat_amsterdam_clean


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = "validation_outputs"
POP_CACHE  = "./pop.obj"

# Fixed simulation parameters (beta, recovery_rate, and ndays are set via CLI)
SIM_PARS = dict(
    datadir               = "./data",
    age_travel_fpath      = "data/amsterdam_travel_data.xlsx",
    # Per-setting beta multipliers: [household, school, workplace, community]
    setting_beta_modifier = [4.0, 0.12, 1.5, 0.007],
    n_initial_infections  = 100,
)

# Fixed population parameters
POP_PARS = sc.objdict(
    datadir   = "./data",
    place     = "Amsterdam",
    wp_zipf_a = 0.983,
    verbose   = 1,
)


# ---------------------------------------------------------------------------
# Population loading / construction
# ---------------------------------------------------------------------------

def load_or_build_population(pop_pars: sc.objdict, pop_cache: str):
    """
    Return a Pop object, loading from disk if a cached copy exists or
    building and saving a new one otherwise.

    Parameters
    ----------
    pop_pars : sc.objdict
        Parameters forwarded to neopatat_amsterdam_clean.Pop().
    pop_cache : str
        File path for the cached population object.

    Returns
    -------
    Pop
        Fully initialised synthetic population.
    """
    if os.path.exists(pop_cache):
        print(f"Loading cached population from '{pop_cache}'...")
        return sc.load(filename=pop_cache)

    print(f"No cached population found at '{pop_cache}'. Building new population...")
    popobj = neopatat_amsterdam_clean.Pop(pop_pars)
    sc.save(filename=pop_cache, obj=popobj)
    print(f"Population saved to '{pop_cache}'.")
    return popobj


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def export_simulation_data(simobj, data_path: str, wijk_path: str) -> None:
    """
    Export per-agent epidemic and demographic data from a completed Sim object
    to CSV files.

    Two files are written:

    data_path
        One row per agent. Columns:
            id, age, wijk, household, swstatus, swindex, swwijk,
            exposure_day, exposure_setting, infector
        exposure_day = -1 means the agent was never infected.
        exposure_setting codes: 1=household, 2=school, 3=workplace,
            4=community, 5=seeded/imported.

    wijk_path
        Wijk index-to-name mapping. One row per wijk. Columns: wijk_idx, wijk_name.
        Written only if the file does not already exist, since the mapping is
        identical across all replicates and parameter sets.

    Parameters
    ----------
    simobj : Sim
        Completed simulation object.
    data_path : str
        Output path for the per-agent CSV.
    wijk_path : str
        Output path for the wijk mapping CSV.
    """
    # Per-agent data: combine epidemic outcomes with demographic attributes
    df = pd.DataFrame({
        "id":               simobj.people.id,
        "age":              simobj.people.age,
        "wijk":             simobj.people.wijk,
        "household":        simobj.people.household,
        "swstatus":         simobj.people.swstatus,
        "swindex":          simobj.people.swindex,
        "swwijk":           simobj.people.swwijk,
        "exposure_day":     simobj.epidemic.exposure_day,
        "exposure_setting": simobj.epidemic.exposure_setting,
        "infector":         simobj.epidemic.infector,
    })
    df.to_csv(data_path, index=False)

    # Wijk mapping — write once; skip if already present from a previous replicate
    if not os.path.exists(wijk_path):
        pd.DataFrame({
            "wijk_idx":  range(len(simobj.people.wijk_idx_to_naam)),
            "wijk_name": simobj.people.wijk_idx_to_naam,
        }).to_csv(wijk_path, index=False)
        print(f"  Saved wijk mapping : {wijk_path}")


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_parameter_set(
    betas: list,
    recovery_rates: list,
    ndays: int,
    replicates: int,
    save_obj: bool,
) -> None:
    """
    Run multiple replicates for each (beta, recovery_rate) pair and save results.

    Parameters are consumed as matched pairs: the first beta is paired with
    the first recovery_rate, and so on. If the lists differ in length, only
    the shorter length of pairs is simulated and a warning is printed.

    Per replicate, the following are always written:
        data_beta{b}_rec{r}_rep{k}.csv  -- per-agent data (see export_simulation_data)

    Optionally (if save_obj=True):
        sim_beta{b}_rec{r}_rep{k}.obj   -- full serialised Sim object

    The wijk index-to-name mapping is written once to wijk_idx_to_naam.csv.

    Parameters
    ----------
    betas : list of float
        Baseline transmission rate values to sweep.
    recovery_rates : list of float
        Daily recovery rate values to sweep (must match length of betas).
    ndays : int
        Number of simulation days per replicate.
    replicates : int
        Number of independent replicates per (beta, recovery_rate) pair.
    save_obj : bool
        If True, save the full Sim object as a .obj file in addition to CSVs.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Warn if lists are unequal — zip will silently truncate to the shorter one
    if len(betas) != len(recovery_rates):
        print(
            f"Warning: length mismatch — {len(betas)} beta(s) vs "
            f"{len(recovery_rates)} recovery_rate(s). "
            f"Only {min(len(betas), len(recovery_rates))} pair(s) will be simulated."
        )

    # Build or load population once; reused across all pairs and replicates
    popobj = load_or_build_population(POP_PARS, POP_CACHE)

    wijk_path = os.path.join(OUTPUT_DIR, "wijk_idx_to_naam.csv")

    n_pairs = min(len(betas), len(recovery_rates))
    print(f"\nRunning {n_pairs} parameter pair(s) x {replicates} replicate(s) "
          f"= {n_pairs * replicates} simulation(s) of {ndays} days each.")

    # ---- Parameter sweep ------------------------------------------------
    for b, r in zip(betas, recovery_rates):

        for rep in range(1, replicates + 1):
            print(f"\n{'='*60}")
            print(f"  beta={b}  recovery_rate={r}  replicate={rep}/{replicates}")
            print(f"{'='*60}")

            # Assemble simulation parameters for this run
            sim_pars = sc.objdict(
                **SIM_PARS,
                ndays         = np.uint32(ndays),
                beta          = float(b),
                recovery_rate = float(r),
            )

            # Run simulation
            simobj = neopatat_amsterdam_clean.Sim(sim_pars, pop=popobj)
            simobj.simulate(death_bool=1)

            # ---- Save outputs -------------------------------------------
            base_name = f"beta{b}_rec{r}_rep{rep}"
            data_path = os.path.join(OUTPUT_DIR, f"data_{base_name}.csv")

            # Per-agent CSV (always written)
            export_simulation_data(simobj, data_path, wijk_path)
            print(f"  Saved agent data   : {data_path}")

            # Full simulation object (optional)
            if save_obj:
                sim_path = os.path.join(OUTPUT_DIR, f"sim_{base_name}.obj")
                sc.save(filename=sim_path, obj=simobj)
                print(f"  Saved sim object   : {sim_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run paired-parameter epidemic simulations for the Amsterdam ABM. "
            "Each --betas value is paired with the corresponding --recovery_rates value. "
            "Each pair is run --replicates times for --ndays simulation days."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        required=True,
        metavar="BETA",
        help="One or more baseline transmission rates.",
    )
    parser.add_argument(
        "--recovery_rates",
        nargs="+",
        type=float,
        required=True,
        metavar="RATE",
        help="One or more daily recovery rates (must match length of --betas).",
    )
    parser.add_argument(
        "--ndays",
        type=int,
        default=50,
        metavar="N",
        help="Number of days to simulate per replicate.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        metavar="K",
        help="Number of independent replicates per (beta, recovery_rate) pair.",
    )
    parser.add_argument(
        "--no-save-obj",
        action="store_false",
        dest="save_obj",
        help=(
            "Skip saving the full .obj simulation file. "
            "Only the per-agent CSV and wijk mapping are written. "
            "Use this to save disk space when the full object is not needed."
        ),
    )

    args = parser.parse_args()
    run_parameter_set(
        args.betas,
        args.recovery_rates,
        args.ndays,
        args.replicates,
        args.save_obj,
    )
