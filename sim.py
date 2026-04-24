"""
sim.py
------
Agent-based influenza simulation for Amsterdam.

Classes
-------
Sim  : Epidemic simulation engine. Runs a stochastic SEIR-like model over a
       pre-built synthetic population, with transmission in four settings
       (household, school, workplace, community).

Pop  : Synthetic population builder. Constructs a georeferenced Amsterdam
       population with realistic age structure, household composition,
       school enrolment, and workplace assignment.

Typical usage
-------------
    pars = sc.objdict(...)
    pop  = Pop(pars)
    sim  = Sim(pars, pop)
    sim.simulate()
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import re
import random

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import sciris as sc
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from . import utils


# ===========================================================================
# Sim
# ===========================================================================

class Sim:
    """
    Stochastic epidemic simulation over a synthetic Amsterdam population.

    The model tracks three discrete disease states per person:
        0 = susceptible
        1 = infectious
        2 = recovered

    Transmission is computed independently in four settings each day:
        household   — within shared household
        school      — primary (swstatus 1) and secondary (swstatus 2) schools
        workplace   — multi-person firms (swstatus 3)
        community   — cross-wijk via gravity-model visitor flux

    School and workplace transmission are suppressed on weekends.

    Parameters
    ----------
    pars : sc.objdict
        Simulation parameters. Required keys:
            beta                  (float)  — baseline transmission rate
            recovery_rate         (float)  — daily recovery probability parameter
            ndays                 (int)    — number of simulation days
            n_initial_infections  (int)    — seeds on day 0
            setting_beta_modifier (list)   — per-setting beta multipliers
                                             [household, school, workplace, community]
            datadir               (str)    — root data directory
            age_travel_fpath      (str)    — path to age-commuting Excel file
    pop : Pop
        Fully initialised population object.
    """

    def __init__(self, pars: sc.objdict, pop: "Pop"):

        # -- Attach population arrays -----------------------------------------
        self.people            = pop.people
        self.dist_matrix       = pop.dist_matrix
        self.visitor_flux_prob = pop.visitor_flux_prob

        # -- Pre-compute place sizes ------------------------------------------
        # Used to normalise within-place contact rates by occupancy.

        # household_n[h] = number of residents in household h
        hh_id, hh_count    = np.unique(self.people.household, return_counts=True)
        self.household_n   = np.zeros(hh_id.max() + 1, dtype=np.uint32)
        self.household_n[hh_id] = hh_count

        # swindex_n[s] = number of people assigned to school/workplace s
        sw_id, sw_count    = np.unique(self.people.swindex, return_counts=True)
        self.swindex_n     = np.zeros(sw_id.max() + 1, dtype=np.uint32)
        self.swindex_n[sw_id] = sw_count

        # -- External data ----------------------------------------------------
        # Age × age contact matrices for all four settings (home/school/work/other)
        self.contact_mat   = utils.get_contact_matrix(pars.datadir, "NLD")

        # Age-specific daily commuting probability (used in community transmission)
        self.age_commute_f = utils.read_age_commute_dist(pars.age_travel_fpath)

        # State-transition rates (currently loaded but not wired into the loop)
        self._load_state_transition_rates(pars)

        # -- Epidemic state ---------------------------------------------------
        self.epidemic = sc.objdict()

        self.epidemic.beta                  = pars.beta
        self.epidemic.recovery_rate         = pars.recovery_rate
        self.epidemic.setting_beta_modifier = pars.setting_beta_modifier
        self.epidemic.ndays                 = pars.ndays
        self.epidemic.n_initial_infections  = pars.n_initial_infections

        n = self.people.pop_size

        # Disease state: 0 = susceptible, 1 = infectious, 2 = recovered
        self.epidemic.curr_state       = np.zeros(n, dtype=np.uint8)
        # Day on which each person's current state began
        self.epidemic.curr_state_st    = np.zeros(n, dtype=np.int32)
        # Setting of exposure: 1=household, 2=school, 3=workplace, 4=community, 5=seeded
        self.epidemic.exposure_setting = np.zeros(n, dtype=np.int32)
        # Day of exposure; -1 = never exposed
        self.epidemic.exposure_day     = np.full(n, -1, dtype=np.int32)
        # Full state time-series: result[t, i] = state of person i on day t
        self.epidemic.result           = np.zeros((pars.ndays + 1, n), dtype=np.uint8)
        # Infector index for each person; -1 = unknown or seeded
        self.epidemic.infector         = np.full(n, -1, dtype=np.int32)

        # Simulation clock (integer days)
        self.t = np.uint32(0)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def simulate(self, death_bool: int = 1) -> None:
        """
        Run the epidemic for self.epidemic.ndays days.

        Each day proceeds in order:
            1. Seed infections (day 0 only).
            2. Recovery step.
            3. Transmission in each setting.
            4. Record state snapshot.

        Parameters
        ----------
        death_bool : int
            Accepted for API compatibility; death is not yet implemented.
        """
        print("Starting simulation...")
        print(f"  Day {self.t}: {np.unique(self.epidemic.curr_state, return_counts=True)}")

        while self.t < self.epidemic.ndays + 1:

            # Day-0 seeding
            if self.t == 0 and self.epidemic.n_initial_infections > 0:
                self._initialize_infections()

            # Recovery must precede transmission so that individuals who recover
            # today cannot simultaneously infect others on the same day.
            self._recovery()

            # Transmission in each setting (school/workplace skipped on weekends)
            self._transmission("household")
            self._transmission("school")
            self._transmission("workplace")
            self._transmission("community")

            # Record full population state for this time step
            self.epidemic.result[self.t, :] = self.epidemic.curr_state

            print(f"  Day {self.t}: {np.unique(self.epidemic.curr_state, return_counts=True)}")
            self.t += 1

    # -----------------------------------------------------------------------
    # Private simulation steps
    # -----------------------------------------------------------------------

    def _transmission(self, setting: str) -> None:
        """
        Compute transmission events for one setting on the current day.

        School and workplace are skipped on weekends. The simulation clock t
        is assumed to start on a Monday (t % 7 == 0), so t % 7 in {5, 6}
        are weekend days.

        Parameters
        ----------
        setting : str
            One of "household", "school", "workplace", "community".
        """
        # Weekend check: skip school and workplace on days 5 and 6 of each week
        if setting in ("school", "workplace") and self.t % 7 >= 5:
            return

        # -- Identify infectious individuals, filtered by setting attendance --
        infectious_inds = self.people.id[self.epidemic.curr_state == 1]

        if setting == "school":
            # swstatus 1 = primary, 2 = secondary school
            mask = (
                (self.people.swstatus[infectious_inds] >= 1) &
                (self.people.swstatus[infectious_inds] < 3)
            )
            infectious_inds = infectious_inds[mask]

        elif setting == "workplace":
            # swstatus 3 = employed in a multi-person firm
            infectious_inds = infectious_inds[
                self.people.swstatus[infectious_inds] == 3
            ]

        if infectious_inds.size == 0:
            return  # no infectious individuals in this setting today

        # -- Identify susceptible individuals, filtered by setting attendance --
        susceptible_inds = self.people.id[self.epidemic.curr_state == 0]

        if setting == "school":
            mask = (
                (self.people.swstatus[susceptible_inds] >= 1) &
                (self.people.swstatus[susceptible_inds] < 3)
            )
            susceptible_inds = susceptible_inds[mask]

        elif setting == "workplace":
            susceptible_inds = susceptible_inds[
                self.people.swstatus[susceptible_inds] == 3
            ]

        # -- Resolve place IDs, contact matrix, and setting-adjusted beta ----
        SETTING_IDX = {"household": 0, "school": 1, "workplace": 2, "community": 3}
        s_idx = SETTING_IDX[setting]

        if setting == "household":
            infectious_places  = self.people.household[infectious_inds]
            susceptible_places = self.people.household[susceptible_inds]
            place_n            = self.household_n
            # Household contacts: uniform mixing, no age structure applied
            setting_contact_mat = np.ones_like(self.contact_mat[0])

        elif setting == "community":
            # Place = wijk; cross-wijk exposure via gravity-model visitor flux
            infectious_places   = self.people.wijk[infectious_inds]
            susceptible_places  = self.people.wijk[susceptible_inds]
            setting_contact_mat = self.contact_mat[3]

        else:
            # School or workplace: place = swindex (school/firm ID)
            infectious_places  = self.people.swindex[infectious_inds]
            susceptible_places = self.people.swindex[susceptible_inds]
            place_n            = self.swindex_n
            setting_contact_mat = self.contact_mat[1 if setting == "school" else 2]

        infectious_age  = self.people.age[infectious_inds]
        beta_setting    = self.epidemic.beta * self.epidemic.setting_beta_modifier[s_idx]

        # -- Compute new exposures --------------------------------------------
        if setting == "community":
            # Community uses the cross-wijk kernel with commuting probability
            infectious_age_commute_f = self.age_commute_f[infectious_age]
            susceptible_age          = self.people.age[susceptible_inds]

            exposed_boolean, infector = utils.compute_rand_transmission(
                infectious_inds, infectious_places, infectious_age,
                susceptible_places, susceptible_age,
                setting_contact_mat, infectious_age_commute_f,
                self.visitor_flux_prob, beta_setting
            )

        else:
            # Within-place kernel: pre-filter to susceptibles sharing a place
            # with at least one infectious person (major speed-up)
            included_sus_mask = utils.filter_susceptibles(
                infectious_places, infectious_age, susceptible_places
            )
            susceptible_inds   = susceptible_inds[included_sus_mask > 0]
            if susceptible_inds.size == 0:
                return
            susceptible_places = susceptible_places[included_sus_mask > 0]
            susceptible_age    = self.people.age[susceptible_inds]

            exposed_boolean, infector = utils.compute_transmission(
                infectious_inds, infectious_places, infectious_age,
                susceptible_places, susceptible_age,
                setting_contact_mat, place_n, beta_setting
            )

        # -- Record outcomes --------------------------------------------------
        self.epidemic.infector[susceptible_inds] = infector

        exposed_persons = susceptible_inds[exposed_boolean > 0]
        self._exposed(exposed_persons)

        # Tag the exposure setting (1-indexed to match the legend in docstring)
        self.epidemic.exposure_setting[exposed_persons] = s_idx + 1

    def _recovery(self) -> None:
        """
        Attempt stochastic recovery for all currently infectious individuals.

        Recovery probability uses a cumulative exponential hazard so that
        the per-day recovery chance increases with time since infection:

            P(recovered by day t | infected on day t0) = 1 - exp(-r * (t - t0))

        where r = epidemic.recovery_rate.
        """
        infectious = self.people.id[self.epidemic.curr_state == 1]
        if infectious.size == 0:
            return

        duration   = self.t - self.epidemic.curr_state_st[infectious]
        recovery_p = 1 - np.exp(-self.epidemic.recovery_rate * duration)
        recovering = infectious[np.random.random(recovery_p.size) < recovery_p]

        if recovering.size > 0:
            self.epidemic.curr_state[recovering]    = 2
            self.epidemic.curr_state_st[recovering] = self.t

    def _exposed(self, exposed_persons: np.ndarray) -> None:
        """
        Transition individuals from susceptible (0) to infectious (1) and
        record the day of exposure.

        Parameters
        ----------
        exposed_persons : np.ndarray
            Person indices to transition.
        """
        self.epidemic.curr_state[exposed_persons]    = 1
        self.epidemic.curr_state_st[exposed_persons] = self.t
        self.epidemic.exposure_day[exposed_persons]  = self.t

    def _initialize_infections(self) -> None:
        """
        Seed n_initial_infections randomly chosen susceptible individuals on
        day 0, representing external importations. Tagged with exposure_setting = 5.
        """
        candidates      = self.people.id[self.epidemic.curr_state == 0]
        exposed_persons = np.random.choice(
            candidates, self.epidemic.n_initial_infections, replace=False
        )
        self.epidemic.exposure_setting[exposed_persons] = 5  # 5 = seeded / imported
        self._exposed(exposed_persons)

    def _load_state_transition_rates(self, pars: sc.objdict) -> None:
        """
        Load the state-transition rate matrix from Excel.

        NOTE: This matrix is loaded but not yet used in the simulation loop —
        the model currently uses epidemic.recovery_rate directly. Preserved for
        future extension to multi-compartment models (e.g., SEIRD).

        Note on attribute name: 'state_transtion_rates' (sic) preserves the
        original misspelling to avoid breaking any downstream code.
        """
        df = (
            pd.read_excel(pars.datadir + "/rate_params_amsterdam.xlsx")
            .set_index(["state_i", "state_j"])
        )
        self.state_transtion_rates = np.zeros((5, 5), dtype=np.float32)
        for (i, j) in df.index.unique():
            self.state_transtion_rates[i, j] = df.loc[(i, j), "rate_per_day"]


# ===========================================================================
# Pop
# ===========================================================================

class Pop:
    """
    Synthetic Amsterdam population.

    Construction order (each step depends on the previous):
        1. setup_age()         — assigns age and wijk index to every person
        2. get_dist_and_area() — builds inter-wijk distance matrix and area vector
        3. grav_mod()          — computes gravity-model visitor-flux probabilities
        4. setup_households()  — assigns household IDs from empirical composition data
        5. get_hh_coord()      — samples a coordinate for each household  [TODO]
        6. setup_schools()     — enrols children in primary/secondary schools
        7. setup_workplaces()  — assigns working-age adults to firms

    swstatus codes
    --------------
    0 = not in school or multi-person firm
    1 = primary school (age 4–11)
    2 = secondary school (age 12–18)
    3 = employed in a multi-person firm
    4 = self-employed / single-person firm

    Parameters
    ----------
    pars : sc.objdict
        Population parameters. Required keys:
            verbose       (int)  — verbosity level
            place         (str)  — city name (used for logging)
            datadir       (str)  — root data directory
            age_travel_fpath (str) — path to age-commuting Excel file
            wp_zipf_a     (float)— Zipf exponent for firm-size distribution
    """

    def __init__(self, pars: sc.objdict):

        if pars.verbose > 0:
            print(f"Creating population for '{pars.place.capitalize()}'...")

        self.people = sc.objdict()

        # 1. Age and wijk assignment
        self.setup_age(pars)

        # 2. Spatial structure: inter-wijk distances and areas
        self.dist_matrix, self.wijk_area = utils.get_dist_and_area()

        # 3. Gravity-model visitor flux
        #    visitor_flux_prob[i, j] = P(person from wijk i visits wijk j)
        _, self.people.popsize_by_wijk = np.unique(self.people.wijk, return_counts=True)
        self.visitor_flux_prob = utils.grav_mod(
            self.dist_matrix, self.people.popsize_by_wijk
        )

        # 4. Household assignment
        self.setup_households(pars)

        # 5. Household coordinates  [TODO: implement get_hh_coord]

        # 6 & 7. School and workplace assignment
        # swstatus and swindex are initialised to 0; setup_* methods fill them.
        self.people.swstatus = np.zeros(self.people.pop_size, dtype=np.uint8)
        self.people.swindex  = np.zeros(self.people.pop_size, dtype=np.uint32)
        self.people.swwijk   = np.zeros(self.people.pop_size, dtype=np.uint32)
        self.setup_schools()
        self.setup_workplaces(pars.wp_zipf_a)

        print("Population construction complete.")

    # -----------------------------------------------------------------------
    # Setup methods
    # -----------------------------------------------------------------------

    def setup_age(self, pars: sc.objdict) -> None:
        """
        Assign age and wijk index to every individual using age-band counts
        from the AHTI/CBS population register.

        Ages are sampled uniformly within each 5-year band. The 95+ band is
        treated as 95–109. Individuals are stored in contiguous wijk blocks.
        """
        # Age band labels from the register
        age_headers = [
            "0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
            "40-44","45-49","50-54","55-59","60-64","65-69","70-74",
            "75-79","80-84","85-89","90-94","95-99","100+"
        ]

        # Load and clean population counts
        pop_counts = (
            pd.read_excel(
                pars.datadir + "/population_counts_2021.xlsx",
                sheet_name="WK_NAAM"
            )
            .pipe(lambda df: df[df["WK_NAAM"].notna()])
            .pipe(lambda df: df[df["Variable"] == "age"])
        )
        pop_counts["count"] = pop_counts["count"].fillna(0)

        # Initialise people arrays
        self.people.pop_size = np.uint32(pop_counts["count"].to_numpy().sum())
        self.people.id       = np.arange(self.people.pop_size, dtype=np.uint32)
        self.people.age      = np.zeros(self.people.pop_size, dtype=np.uint16)
        self.people.wijk     = np.zeros(self.people.pop_size, dtype=np.uint16)

        wijk_idx_to_naam = []  # built as list to avoid O(n²) np.append
        prev_id = 0

        for wijk_idx, wijk_name in enumerate(pop_counts["WK_NAAM"].unique()):
            wijk_idx_to_naam.append(wijk_name)

            wijk_counts     = pop_counts[pop_counts["WK_NAAM"] == wijk_name]
            age_demography  = wijk_counts["count"].to_numpy().astype(np.int32)
            age_categories  = wijk_counts["Level"].to_list()

            for age_level, N in zip(age_categories, age_demography):
                curr_id = prev_id + N

                # Parse age band lower bound; cap upper bound at 110 for 95+
                min_age = int(re.search(r"^[0-9]+", age_level).group())
                max_age = min_age + 5 if min_age < 95 else 110

                # Assign ages uniformly within the band
                age_range = np.arange(min_age, max_age, dtype=np.int32)
                self.people.age[prev_id:curr_id]  = np.random.choice(age_range, N, replace=True)
                self.people.wijk[prev_id:curr_id] = wijk_idx

                prev_id = curr_id

        self.people.wijk_idx_to_naam = np.array(wijk_idx_to_naam)

    def setup_workplaces(self, wp_zipf_a: float) -> None:
        """
        Assign working-age individuals (19–65) to firms drawn from a
        Zipf (power-law) firm-size distribution.

        Single-employee firms receive swstatus = 4 (self-employed).
        Multi-person firm employees receive swstatus = 3.

        Firm IDs are appended after any school IDs already in swindex so
        that school and firm index spaces do not overlap.

        Parameters
        ----------
        wp_zipf_a : float
            Zipf exponent controlling the firm-size distribution.
        """
        # Select working-age individuals (19 < age <= 65)
        employed_idxes = self.people.id[
            (self.people.age > 18) & (self.people.age <= 65)
        ]
        employed_n    = employed_idxes.size
        employed_wijk = self.people.wijk[employed_idxes]

        # Estimate number of firms from the Zipf distribution
        firms_n = (
            (wp_zipf_a - 1) / wp_zipf_a
            * (((1 / employed_n) ** wp_zipf_a - 1)
               / ((1 / employed_n) ** wp_zipf_a - (1 / employed_n)))
        )
        firms_n = np.uint32(firms_n)

        employed_firms, employed_firms_location = utils.assign_firms(
            wp_zipf_a, firms_n, employed_wijk, self.people.wijk, self.dist_matrix
        )

        # Self-employed (firm ID < 0)
        self_employed_mask = employed_firms < 0
        self.people.swstatus[employed_idxes[self_employed_mask]] = 4

        n_self_employed = self_employed_mask.sum()
        print(f"\n{n_self_employed:,} of {employed_n:,} employed are in single-person firms.")

        # Multi-person firm employees (firm ID > 0)
        multi_mask = employed_firms > 0
        firm_employees     = employed_idxes[multi_mask]
        firm_locations     = employed_firms_location[multi_mask]
        firm_ids           = employed_firms[multi_mask]

        # Offset firm IDs to avoid collision with school IDs in swindex
        id_offset = self.people.swindex.max()

        self.people.swstatus[firm_employees] = 3
        self.people.swindex[firm_employees]  = firm_ids + id_offset
        self.people.swwijk[firm_employees]   = firm_locations

    def setup_schools(self) -> None:
        """
        Enrol children in primary and secondary schools.

        Primary schools (ages 4–11): school sizes drawn from a normal
        distribution (mean 263, SD 151) calibrated to Amsterdam data.
        Schools are assigned per wijk; children choose a school randomly
        within their wijk.

        Secondary schools (ages 12–18): a fixed mean size of 558 pupils
        (empirically derived). Schools are placed across the city weighted
        by wijk population; each child is assigned the nearest school within
        an expanding distance threshold.
        """
        # ---- Primary schools ------------------------------------------------
        PRIMARY_SIZE_MEAN = 263   # mean pupils per primary school
        PRIMARY_SIZE_SD   = 151   # standard deviation
        PRIMARY_MIN_SIZE  = 50    # discard schools smaller than this

        n_school_so_far = 1  # running school ID counter (1-based)

        for wijk_idx, wijk_name in enumerate(self.people.wijk_idx_to_naam):

            primary_kids = self.people.id[
                (self.people.wijk == wijk_idx) &
                (self.people.age > 3) &
                (self.people.age < 12)
            ]
            self.people.swstatus[primary_kids] = 1

            n_kids = len(primary_kids)
            if n_kids == 0:
                continue

            # Sample school sizes until cumulative capacity exceeds n_kids
            size_sample = np.random.normal(PRIMARY_SIZE_MEAN, PRIMARY_SIZE_SD, 50)
            size_sample = size_sample[size_sample > PRIMARY_MIN_SIZE]

            n_schools = (n_kids > np.cumsum(size_sample)).argmin()
            size_sample = size_sample[:n_schools + 1]
            # Last school absorbs any remaining children
            size_sample[n_schools] = n_kids - size_sample[:n_schools].sum()

            # Assign each child to a school proportional to school size
            school_ids = np.arange(n_schools + 1) + n_school_so_far
            probs      = size_sample / size_sample.sum()
            sampled    = np.random.choice(school_ids, n_kids, p=probs)

            self.people.swindex[primary_kids] = sampled
            self.people.swwijk[primary_kids]  = wijk_idx

            n_school_so_far += n_schools + 1

        # ---- Secondary schools ----------------------------------------------
        SECONDARY_SIZE = 558  # mean pupils per secondary school

        secondary_kids = self.people.id[
            (self.people.age >= 12) & (self.people.age <= 18)
        ]
        self.people.swstatus[secondary_kids] = 2

        n_secondary_kids  = len(secondary_kids)
        n_schools_needed  = round(n_secondary_kids / SECONDARY_SIZE)
        n_primary_schools = int(self.people.swindex.max())

        # Place secondary schools across wijken, weighted by population size
        pop_probs       = self.people.popsize_by_wijk / self.people.popsize_by_wijk.sum()
        school_wijken   = np.random.choice(
            np.unique(self.people.wijk), n_schools_needed, p=pop_probs
        )
        # Secondary school IDs follow on from primary school IDs
        secondary_ids   = np.arange(n_schools_needed) + n_primary_schools + 1

        # Assign each child to the nearest accessible school, expanding radius
        search_radii = [3_000, 6_000, 9_000, 20_000]  # metres

        for kid_idx in secondary_kids:
            kid_wijk  = self.people.wijk[kid_idx]
            dist_to_schools = self.dist_matrix[kid_wijk, school_wijken]

            for max_dist in search_radii:
                nearby = secondary_ids[dist_to_schools < max_dist]
                if nearby.size == 0:
                    continue

                chosen_id = np.random.choice(nearby)
                self.people.swindex[kid_idx] = chosen_id
                # Store the wijk of the chosen school
                self.people.swwijk[kid_idx]  = school_wijken[
                    np.where(secondary_ids == chosen_id)[0]
                ]
                break

    def setup_households(self, pars: sc.objdict) -> None:
        """
        Assign individuals to households using empirical household-composition
        counts from the AHTI dataset.

        The Excel file contains counts of HOUSEHOLDS (not people) for each
        combination of:
            hh_type         — 1-person / 2-person / Single Parent / Couple w children
            hh_nr_children  — 0 / 1 / 2 / 3+ children per family household
            hh_nr_seniors   — 0 / 1 / 2 / 3+ seniors (age ≥ 65) per household

        Processing per wijk
        --------------------
        1. Build a 'child count deck': a shuffled list with one entry per
           family household specifying how many children it should receive.
        2. Build a 'senior deck': similarly for senior counts across all
           household types.
        3. Fill households in order (1-person, 2-person, single-parent,
           couple-with-children) drawing from age-stratified pools.
        4. Place any unhoused individuals in single-person households.

        Age thresholds
        --------------
        CHILD_MAX_AGE = 17   (age ≤ 17 are children)
        SENIOR_AGE    = 65   (age ≥ 65 are seniors)
        """
        SENIOR_AGE    = 65
        CHILD_MAX_AGE = 17

        # -- Load and clean household composition data -----------------------
        hh_df = (
            pd.read_excel(pars.datadir + "/hh_counts_2021.xlsx", sheet_name="WK_NAAM")
            .pipe(lambda df: df[df["WK_NAAM"].notna()])
            .pipe(lambda df: df[df["WK_NAAM"] != "Unknown"])
        )
        hh_df["count"] = hh_df["count"].fillna(0).astype(np.int32)
        self.people.hh_df = hh_df

        # Power-law weights for randomly expanding "3+ children" counts
        # to specific values in [3, 10]; heavily weights smaller sizes
        pow_weights  = 1.0 / np.power(np.arange(3, 11), 4).astype(float)
        pow_weights /= pow_weights.sum()

        # ---- Helper closures -----------------------------------------------

        def get_counts(variable: str, wijk_name: str) -> dict:
            """Return {Level: count} dict for a given variable and wijk."""
            sub = hh_df[(hh_df["Variable"] == variable) & (hh_df["WK_NAAM"] == wijk_name)]
            return sub.set_index("Level")["count"].to_dict()

        def pop_from(pool: list, n: int) -> list:
            """
            Remove and return up to n items from the front of pool (in-place).
            Returns fewer than n items if the pool is exhausted.
            """
            n     = min(n, len(pool))
            taken = pool[:n]
            del pool[:n]
            return taken

        def make_child_count_deck(nr_children_counts: dict, n_family_hh: int) -> list:
            """
            Build a shuffled deck of per-household child counts with length
            n_family_hh. The "3+ children" entries are expanded to specific
            values drawn from a power-law distribution over [3, 10].
            """
            n1  = int(nr_children_counts.get("1 child",      0))
            n2  = int(nr_children_counts.get("2 childrens",  0))
            n3p = int(nr_children_counts.get("3+ childrens", 0))

            deck = (
                [1] * n1 +
                [2] * n2 +
                list(np.random.choice(np.arange(3, 11), size=n3p, p=pow_weights).astype(int))
            )
            np.random.shuffle(deck)

            # Pad with 1-child entries if the deck is shorter than n_family_hh
            if len(deck) < n_family_hh:
                deck += [1] * (n_family_hh - len(deck))

            return deck

        def make_senior_deck(nr_seniors_counts: dict, n_hh: int) -> list:
            """
            Build a shuffled deck of per-household senior counts with length n_hh.
            "3+ seniors" is treated as exactly 3.
            """
            n0  = int(nr_seniors_counts.get("0 seniors",  0))
            n1  = int(nr_seniors_counts.get("1 senior",   0))
            n2  = int(nr_seniors_counts.get("2 seniors",  0))
            n3p = int(nr_seniors_counts.get("3+ seniors", 0))

            deck = [0] * n0 + [1] * n1 + [2] * n2 + [3] * n3p

            # Trim or pad to exactly n_hh entries
            if len(deck) > n_hh:
                deck = deck[:n_hh]
            elif len(deck) < n_hh:
                deck += [0] * (n_hh - len(deck))

            np.random.shuffle(deck)
            return deck

        # ---- Main per-wijk loop --------------------------------------------
        self.people.household = np.zeros(self.people.pop_size, dtype=np.uint32)
        n_hh_so_far = 1  # household IDs start at 1 (0 = unassigned sentinel)

        for wijk_idx, wijk_name in enumerate(self.people.wijk_idx_to_naam):

            people_in_wijk = self.people.id[self.people.wijk == wijk_idx]
            if len(people_in_wijk) == 0:
                continue

            if pars.verbose > 1:
                print(f"  {wijk_name}: {len(people_in_wijk):,} people")

            # Load household-type and composition counts for this wijk
            hh_type_counts     = get_counts("hh_type",        wijk_name)
            nr_children_counts = get_counts("hh_nr_children", wijk_name)
            nr_seniors_counts  = get_counts("hh_nr_seniors",  wijk_name)

            n_hh_1p  = int(hh_type_counts.get("1 person",          0))
            n_hh_2p  = int(hh_type_counts.get("2 person",          0))
            n_hh_sp  = int(hh_type_counts.get("Single Parent",      0))
            n_hh_cwc = int(hh_type_counts.get("Couple w children",  0))
            n_hh_family = n_hh_sp + n_hh_cwc
            n_hh_total  = n_hh_1p + n_hh_2p + n_hh_family

            # Build per-household composition decks
            child_deck  = make_child_count_deck(nr_children_counts, n_hh_family)
            senior_deck = make_senior_deck(nr_seniors_counts, n_hh_total)
            senior_deck_iter = iter(senior_deck)

            # Shuffle age-stratified pools for this wijk
            age = self.people.age
            unhoused_children = list(people_in_wijk[age[people_in_wijk] <= CHILD_MAX_AGE])
            unhoused_seniors  = list(people_in_wijk[age[people_in_wijk] >= SENIOR_AGE])
            unhoused_adults   = list(people_in_wijk[
                (age[people_in_wijk] > CHILD_MAX_AGE) &
                (age[people_in_wijk] < SENIOR_AGE)
            ])
            np.random.shuffle(unhoused_children)
            np.random.shuffle(unhoused_seniors)
            np.random.shuffle(unhoused_adults)

            # ---- 1-person households ----
            for _ in range(n_hh_1p):
                n_s = min(next(senior_deck_iter, 0), 1)
                if n_s == 1 and unhoused_seniors:
                    members = pop_from(unhoused_seniors, 1)
                elif unhoused_adults:
                    members = pop_from(unhoused_adults, 1)
                elif unhoused_seniors:
                    members = pop_from(unhoused_seniors, 1)
                else:
                    break  # no one left to house
                self.people.household[members] = n_hh_so_far
                n_hh_so_far += 1

            # ---- 2-person households ----
            for _ in range(n_hh_2p):
                n_s = min(next(senior_deck_iter, 0), 2)
                n_a = 2 - n_s
                members = pop_from(unhoused_seniors, n_s) + pop_from(unhoused_adults, n_a)

                # Top up to 2 members if either pool ran short
                shortfall = 2 - len(members)
                if shortfall > 0:
                    members += pop_from(unhoused_adults,  shortfall)
                shortfall = 2 - len(members)
                if shortfall > 0:
                    members += pop_from(unhoused_seniors, shortfall)

                if len(members) < 2:
                    # Pools exhausted; return partial draw and stop
                    for m in members:
                        self.people.household[m] = 0
                    break

                self.people.household[members] = n_hh_so_far
                n_hh_so_far += 1

            # ---- Single-parent households ----
            child_deck_iter = iter(child_deck)
            for _ in range(n_hh_sp):
                if not unhoused_adults:
                    break
                n_ch = max(1, min(next(child_deck_iter, 1), len(unhoused_children)))
                if n_ch == 0:
                    break
                n_s = min(next(senior_deck_iter, 0), 1)
                parent   = (pop_from(unhoused_seniors, 1) if n_s == 1 and unhoused_seniors
                            else pop_from(unhoused_adults, 1))
                children = pop_from(unhoused_children, n_ch)
                self.people.household[parent + children] = n_hh_so_far
                n_hh_so_far += 1

            # ---- Couple-with-children households ----
            for _ in range(n_hh_cwc):
                if len(unhoused_adults) + len(unhoused_seniors) < 2:
                    break
                n_ch = max(1, min(next(child_deck_iter, 1), len(unhoused_children)))
                if n_ch == 0:
                    break
                n_s  = min(next(senior_deck_iter, 0), 2)
                n_a  = 2 - n_s
                parents = pop_from(unhoused_seniors, n_s) + pop_from(unhoused_adults, n_a)

                # Top up to 2 parents if needed
                shortfall = 2 - len(parents)
                if shortfall > 0:
                    parents += pop_from(unhoused_adults,  shortfall)
                shortfall = 2 - len(parents)
                if shortfall > 0:
                    parents += pop_from(unhoused_seniors, shortfall)

                children = pop_from(unhoused_children, n_ch)
                self.people.household[parents + children] = n_hh_so_far
                n_hh_so_far += 1

            # ---- Leftover pass: house anyone not yet assigned ----
            unhoused = list(people_in_wijk[self.people.household[people_in_wijk] == 0])
            if unhoused:
                print(f"  Warning: {len(unhoused)} unhoused in {wijk_name} "
                      f"— placing in 1-person households.")
            for person in unhoused:
                self.people.household[person] = n_hh_so_far
                n_hh_so_far += 1

            # Sanity check
            still_unhoused = int(np.sum(self.people.household[people_in_wijk] == 0))
            if still_unhoused > 0:
                print(f"  ERROR: {still_unhoused} still unhoused in {wijk_name} "
                      f"after leftover pass.")
