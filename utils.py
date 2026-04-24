"""
utils.py
--------
Utility functions for the Amsterdam influenza ABM.

Contents
--------
- read_age_commute_dist   : Load age-specific commuting probability vector.
- compute_visitor_flux_prob: Schläpfer-style visitor flux (unused; kept for reference).
- grav_mod                : Gravity-model visitor-flux matrix (used in simulation).
- weighted_choice         : Numba-accelerated weighted random draw.
- compute_rand_transmission: Community (cross-wijk) transmission kernel.
- filter_susceptibles     : Pre-filter susceptibles to those sharing a place with an
                            infectious person (speeds up within-place transmission).
- compute_transmission    : Within-place transmission kernel.
- get_contact_matrix      : Load Prem et al. age-structured contact matrices.
- choose_workplaces       : Assign workers to firms with distance preference.
- assign_firms            : Build Zipf-distributed firm structure and assign employees.
- minobj_zipf_fn          : Objective function for Zipf-A parameter fitting.
- fit_zipf_A              : Fit the Zipf scale parameter A given N and alpha.
- get_dist_and_area       : Build inter-wijk centroid distance matrix and area vector.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.stats as st
from scipy.optimize import minimize
from numba import jit, prange


# ---------------------------------------------------------------------------
# Age-structured commuting
# ---------------------------------------------------------------------------

def read_age_commute_dist(age_travel_fpath: str) -> np.ndarray:
    """
    Load age-specific commuting probability and return a per-age vector.

    The Excel file contains rows indexed by (min_age, max_age) with columns
    for each city and an optional 'multiplier' column. For Amsterdam ('amsterdam'),
    the raw fraction is multiplied by the multiplier to obtain the final
    commuting probability used in community transmission.

    Parameters
    ----------
    age_travel_fpath : str
        Path to the Excel file with columns: min_age, max_age, amsterdam, multiplier.

    Returns
    -------
    age_commute_f : np.ndarray, shape (200,), dtype float32
        age_commute_f[a] = probability that a person of age a commutes on a given day.
    """
    age_travel_df = pd.read_excel(age_travel_fpath).set_index(["min_age", "max_age"])

    age_commute_f = np.zeros(200, dtype=np.float32)
    for (min_age, max_age) in age_travel_df.index.unique():
        n_ages   = max_age - min_age + 1
        raw_frac = age_travel_df.loc[(min_age, max_age), "amsterdam"] / n_ages
        mult     = age_travel_df.loc[(min_age, max_age), "multiplier"]
        age_commute_f[min_age : max_age + 1] = raw_frac * mult

    return age_commute_f


# ---------------------------------------------------------------------------
# Mobility / visitor flux
# ---------------------------------------------------------------------------

@jit(nopython=True, parallel=True, fastmath=True)
def compute_visitor_flux_prob(
    dist_matrix: np.ndarray,
    popgrid: np.ndarray,
    A: float,
    fhome: float = 1.0,
    T: float = 1.0,
) -> np.ndarray:
    """
    Compute inter-wijk visitor-flux probability matrix using the Schläpfer et al.
    (2021, Nature) scaling model.

    NOTE: This function is *not* currently used by the simulation — the simpler
    gravity model (grav_mod) is used instead. It is retained here for reference
    and potential future use.

    Parameters
    ----------
    dist_matrix : np.ndarray, shape (n, n)
        Pairwise centroid distances between wijken (metres).
    popgrid : np.ndarray, shape (n,)
        Population count per wijk.
    A : float
        Wijk area (km²).
    fhome : float
        Minimum visit frequency (assumed home = 1 visit/day).
    T : float
        Time period (days).

    Returns
    -------
    Q : np.ndarray, shape (n, n), dtype float32
        Q[i, j] = probability that a person from wijk i visits wijk j.
        Rows sum to 1.
    """
    n = dist_matrix.shape[0]
    Q = np.zeros((n, n), dtype=np.float32)

    popgrid_exp = popgrid ** 0.6  # sub-linear scaling with population

    for i in prange(n):
        # Squared distance decay
        d_ij = dist_matrix[i, :] ** 2
        Q[i, :] = popgrid_exp / d_ij
        # Normalise to a probability distribution
        Q[i, :] = Q[i, :] / np.sum(Q[i, :])

    return Q


@jit(nopython=True, parallel=True, fastmath=True)
def grav_mod(dist_matrix: np.ndarray, popgrid: np.ndarray) -> np.ndarray:
    """
    Compute inter-wijk visitor-flux probability using a standard gravity model.

    This is the mobility model actually used in the simulation. The attraction of
    wijk j from wijk i scales as pop(j)^0.6 / dist(i,j)^1.3.

    Parameters
    ----------
    dist_matrix : np.ndarray, shape (n, n), dtype float32
        Pairwise centroid distances between wijken (metres).
    popgrid : np.ndarray, shape (n,)
        Population count per wijk.

    Returns
    -------
    Q : np.ndarray, shape (n, n), dtype float32
        Q[i, j] = probability that a person from wijk i is found in wijk j.
        Rows sum to 1.
    """
    n = dist_matrix.shape[0]
    Q = np.zeros((n, n), dtype=np.float32)

    popgrid_exp = popgrid ** 0.6  # sub-linear population scaling

    for i in prange(n):
        d_ij   = dist_matrix[i, :] ** 1.3  # distance decay exponent
        Q[i,:] = popgrid_exp / d_ij
        Q[i,:] = Q[i,:] / np.sum(Q[i,:])   # normalise to probability

    return Q


# ---------------------------------------------------------------------------
# Stochastic helpers
# ---------------------------------------------------------------------------

@jit(nopython=True)
def weighted_choice(probs: np.ndarray) -> int:
    """
    Draw a single integer index proportional to probs using the inverse-CDF
    method. Faster than np.random.choice for small arrays inside JIT loops.

    Parameters
    ----------
    probs : np.ndarray
        Probability weights; must sum to 1.

    Returns
    -------
    int
        Sampled index in [0, len(probs)).
    """
    cdf = np.cumsum(probs)
    r   = np.random.random()
    return np.searchsorted(cdf, r)


# ---------------------------------------------------------------------------
# Transmission kernels
# ---------------------------------------------------------------------------

@jit(nopython=True, parallel=True, fastmath=True)
def compute_rand_transmission(
    infectious_inds: np.ndarray,
    infectious_places: np.ndarray,
    infectious_age: np.ndarray,
    susceptible_places: np.ndarray,
    susceptible_age: np.ndarray,
    setting_contact_mat: np.ndarray,
    infectious_age_commute_f: np.ndarray,
    visitor_flux_prob: np.ndarray,
    beta: float,
):
    """
    Community (cross-wijk) transmission kernel.

    For each susceptible individual s in wijk w_s, the force of infection
    from all currently infectious individuals is accumulated using the
    gravity-model visitor-flux probability and age-structured contact rates.
    A Poisson-based probability of at least one transmission event is then
    computed and a stochastic draw determines exposure.

    Parameters
    ----------
    infectious_inds : np.ndarray, shape (n_inf,)
        Person indices of currently infectious individuals.
    infectious_places : np.ndarray, shape (n_inf,)
        Wijk index of each infectious person.
    infectious_age : np.ndarray, shape (n_inf,)
        Age of each infectious person.
    susceptible_places : np.ndarray, shape (n_sus,)
        Wijk index of each susceptible person.
    susceptible_age : np.ndarray, shape (n_sus,)
        Age of each susceptible person.
    setting_contact_mat : np.ndarray, shape (max_age, max_age)
        Age-structured contact rates for the community setting.
    infectious_age_commute_f : np.ndarray, shape (n_inf,)
        Age-specific commuting probability for each infectious person.
    visitor_flux_prob : np.ndarray, shape (n_wijk, n_wijk)
        visitor_flux_prob[i, j] = probability a person from i is in j.
    beta : float
        Effective transmission rate (setting-adjusted).

    Returns
    -------
    exposed_boolean : np.ndarray, shape (n_sus,), dtype uint8
        1 if susceptible s was exposed, 0 otherwise.
    infector : np.ndarray, shape (n_sus,), dtype int32
        Index of the infector for each newly exposed person; -1 if not exposed.
    """
    n_sus = susceptible_age.size
    exposed_boolean = np.zeros(n_sus, dtype=np.uint8)
    infector        = np.zeros(n_sus, dtype=np.int32) - 1  # default: no infector

    for s in prange(n_sus):
        sus_age   = susceptible_age[s]
        sus_place = susceptible_places[s]

        # Poisson rate: sum over all infectious individuals of
        #   beta * contact_rate(sus_age, inf_age) * p(inf visits sus_place) * commute_f
        poisson_mu = (
            -beta
            * setting_contact_mat[sus_age][infectious_age]
            * visitor_flux_prob[sus_place][infectious_places]
            * infectious_age_commute_f
        )
        # Probability of at least one transmission event (1 - exp(-lambda))
        prob = 1 - np.exp(poisson_mu.sum())

        if np.random.random() < prob:
            exposed_boolean[s] = 1
            # Identify infector: sample proportional to contact * flux weight
            weight = (
                setting_contact_mat[sus_age][infectious_age]
                * visitor_flux_prob[sus_place][infectious_places]
            )
            weight  = weight / np.sum(weight)
            infector[s] = infectious_inds[weighted_choice(weight)]

    return exposed_boolean, infector


@jit(nopython=True, parallel=True)
def filter_susceptibles(
    infectious_places: np.ndarray,
    infectious_age: np.ndarray,
    susceptible_places: np.ndarray,
) -> np.ndarray:
    """
    Return a boolean mask selecting susceptibles who share a place with at
    least one infectious person.

    This pre-filtering step avoids evaluating transmission for susceptibles
    in places with no infectious individuals, substantially reducing the
    number of inner-loop iterations in compute_transmission.

    Parameters
    ----------
    infectious_places : np.ndarray, shape (n_inf,)
        Place IDs of infectious individuals.
    infectious_age : np.ndarray, shape (n_inf,)
        Ages of infectious individuals (accepted for API compatibility; unused).
    susceptible_places : np.ndarray, shape (n_sus,)
        Place IDs of susceptible individuals.

    Returns
    -------
    included_sus_mask : np.ndarray, shape (n_sus,), dtype uint8
        1 if the susceptible shares a place with an infectious person, else 0.
    """
    included_sus_mask       = np.zeros(susceptible_places.size, dtype=np.uint8)
    unique_infectious_places = np.unique(infectious_places)

    for p in prange(unique_infectious_places.size):
        place = unique_infectious_places[p]
        included_sus_mask[susceptible_places == place] = 1

    return included_sus_mask


@jit(nopython=True, parallel=True, fastmath=True)
def compute_transmission(
    infectious_inds: np.ndarray,
    infectious_places: np.ndarray,
    infectious_age: np.ndarray,
    susceptible_places: np.ndarray,
    susceptible_age: np.ndarray,
    setting_contact_mat: np.ndarray,
    place_n: np.ndarray,
    beta: float,
):
    """
    Within-place transmission kernel (household, school, workplace).

    For each susceptible person s, only infectious individuals in the *same*
    place contribute to the force of infection. The per-contact rate is
    normalised by place size (1 / place_n) so that larger places dilute
    transmission.

    Parameters
    ----------
    infectious_inds : np.ndarray, shape (n_inf,)
        Person indices of currently infectious individuals.
    infectious_places : np.ndarray, shape (n_inf,)
        Place ID of each infectious person.
    infectious_age : np.ndarray, shape (n_inf,)
        Age of each infectious person.
    susceptible_places : np.ndarray, shape (n_sus,)
        Place ID of each susceptible person (pre-filtered by filter_susceptibles).
    susceptible_age : np.ndarray, shape (n_sus,)
        Age of each susceptible person.
    setting_contact_mat : np.ndarray, shape (max_age, max_age)
        Age-structured contact rates for this setting.
    place_n : np.ndarray, shape (max_place_id + 1,)
        Number of people assigned to each place ID.
    beta : float
        Effective transmission rate (setting-adjusted).

    Returns
    -------
    exposed_boolean : np.ndarray, shape (n_sus,), dtype uint8
        1 if susceptible s was exposed, 0 otherwise.
    infector : np.ndarray, shape (n_sus,), dtype int32
        Index of the infector for each newly exposed person; -1 if not exposed.
    """
    n_sus           = susceptible_age.size
    exposed_boolean = np.zeros(n_sus, dtype=np.uint8)
    infector        = np.zeros(n_sus, dtype=np.int32) - 1

    for i in prange(n_sus):
        sus_place = susceptible_places[i]
        sus_age   = susceptible_age[i]

        # Restrict to infectious individuals in the same place
        same_place_mask          = infectious_places == sus_place
        inf_age_at_sus_place     = infectious_age[same_place_mask]
        inf_inds_at_sus_place    = infectious_inds[same_place_mask]

        if inf_age_at_sus_place.size == 0:
            continue

        # Force of infection: beta * contact_rate / place_size
        poisson_mu = (
            -beta
            * setting_contact_mat[sus_age][inf_age_at_sus_place]
            * (1.0 / place_n[sus_place])
        )
        prob = 1 - np.exp(poisson_mu.sum())

        if np.random.random() < prob:
            exposed_boolean[i] = 1
            # Sample infector weighted by contact rate
            weight = setting_contact_mat[sus_age][inf_age_at_sus_place]
            weight = weight / np.sum(weight)
            infector[i] = inf_inds_at_sus_place[weighted_choice(weight)]

    return exposed_boolean, infector


# ---------------------------------------------------------------------------
# Contact matrices
# ---------------------------------------------------------------------------

def get_contact_matrix(datadir: str, country: str) -> np.ndarray:
    """
    Load Prem et al. (2020) synthetic contact matrices and expand them from
    5-year age bins to single-year resolution.

    Returns a (4, 100, 100) array indexed as:
        [setting, age_contactor, age_contactee]
    where settings are: 0=home, 1=school, 2=work, 3=community.

    Parameters
    ----------
    datadir : str
        Root data directory. The contact matrix CSV is expected at
        {datadir}/prem-et-al_synthetic_contacts_2020.csv.
    country : str
        ISO3C country code (e.g., "NLD").

    Returns
    -------
    contact_mat : np.ndarray, shape (4, 100, 100), dtype float32
        contact_mat[s, a, b] = mean daily contacts between a person aged a
        and a person aged b in setting s.
    """
    contact_data = (
        pd.read_csv(os.path.join(datadir, "prem-et-al_synthetic_contacts_2020.csv"))
        .set_index(["iso3c", "setting", "location_contact"])
        .sort_index()
    )

    contact_mat = np.zeros((4, 100, 100), dtype=np.float32)
    settings    = ["home", "school", "work", "others"]

    for i, location in enumerate(settings):
        # Extract country × setting contact data and pivot to matrix
        raw = (
            contact_data
            .loc[(country, "overall", location)]
            .reset_index()[["age_contactor", "age_contactee", "mean_number_of_contacts"]]
        )
        cm_5yr = (
            raw.pivot(index="age_contactor", columns="age_contactee",
                      values="mean_number_of_contacts")
            .to_numpy()
            .astype(np.float32)
        )

        # Expand from 5-year bins to single-year resolution
        n_bins = cm_5yr.shape[0]
        for bx in range(n_bins):
            min_x, max_x = bx * 5, bx * 5 + 5
            for by in range(cm_5yr.shape[1]):
                min_y, max_y = by * 5, by * 5 + 5
                contact_mat[i, min_x:max_x, min_y:max_y] = cm_5yr[bx, by]

    return contact_mat


# ---------------------------------------------------------------------------
# Firm / workplace assignment
# ---------------------------------------------------------------------------

@jit(nopython=True, parallel=True, fastmath=True)
def choose_workplaces(
    employees_loc: np.ndarray,
    workplace_locs: np.ndarray,
    dist_matrix: np.ndarray,
    min_firm_size: int,
    max_firm_size: int = 1000,
) -> np.ndarray:
    """
    Assign each employee to a workplace with a preference for nearby firms,
    expanding the search radius in steps until a suitable firm is found.

    Firms that have not yet reached min_firm_size are preferred so that
    firms fill up before accepting additional workers above the minimum.

    Parameters
    ----------
    employees_loc : np.ndarray, shape (n_employees,)
        Wijk index of each employee.
    workplace_locs : np.ndarray, shape (n_firms,)
        Wijk index of each candidate firm.
    dist_matrix : np.ndarray, shape (n_wijk, n_wijk)
        Inter-wijk distance matrix (metres).
    min_firm_size : int
        Target minimum firm size; firms below this are preferred.
    max_firm_size : int
        Absolute cap on firm size.

    Returns
    -------
    chosen_wp_idx : np.ndarray, shape (n_employees,), dtype int32
        1-based firm index for each employee; -1 if no firm could be assigned.
    """
    n               = employees_loc.size
    n_firms         = workplace_locs.size
    # 1-based firm indices so that 0 can serve as "unassigned"
    potential_wp_idx   = np.arange(n_firms) + 1
    potential_wp_count = np.zeros(n_firms, dtype=np.uint32)
    chosen_wp_idx      = np.zeros(n, dtype=np.int32) - 1  # -1 = unassigned

    # Progressively expand maximum commute distance until a firm is found
    search_radii = [5_000, 10_000, 20_000]  # metres

    for s in np.arange(n):
        emp_wijk   = employees_loc[s]
        dist_to_wp = dist_matrix[emp_wijk, :][workplace_locs]

        for max_dist in search_radii:
            nearby_wp = potential_wp_idx[dist_to_wp <= max_dist]

            # Prefer under-filled firms; fall back to all nearby if none
            underfilled = nearby_wp[potential_wp_count[nearby_wp - 1] < min_firm_size]
            candidates  = underfilled if underfilled.size > 0 else nearby_wp

            if candidates.size > 0:
                chosen              = np.random.choice(candidates)
                chosen_wp_idx[s]    = chosen
                potential_wp_count[chosen - 1] += 1
                break  # stop expanding radius once assigned

    return chosen_wp_idx


def assign_firms(
    wp_zipf_a: float,
    firms_n: int,
    employed_wijk: np.ndarray,
    wijk: np.ndarray,
    dist_matrix: np.ndarray,
) -> tuple:
    """
    Build a Zipf-distributed set of firms and assign employed individuals to them.

    Firm sizes are drawn from power-law bins [1, 2), [2, 4), [4, 8), ..., each
    containing a Zipf-law fraction of the total firms_n firms. Employees are
    then distributed across firms using choose_workplaces.

    Single-employee "firms" receive a firm ID of -1 (handled as self-employed
    in setup_workplaces).

    Parameters
    ----------
    wp_zipf_a : float
        Zipf exponent alpha controlling the firm-size distribution.
    firms_n : int
        Total number of firms to create.
    employed_wijk : np.ndarray, shape (n_employed,)
        Wijk of each employed individual.
    wijk : np.ndarray, shape (pop_size,)
        Wijk of every individual (used to weight firm location sampling).
    dist_matrix : np.ndarray, shape (n_wijk, n_wijk)
        Inter-wijk distance matrix.

    Returns
    -------
    employed_firms : np.ndarray, shape (n_employed,), dtype int32
        Firm ID for each employee; -1 = single-person / self-employed.
    employed_firms_location : np.ndarray, shape (n_employed,), dtype int32
        Wijk of the assigned firm for each employee.
    """
    employed_N              = employed_wijk.size
    employed_firms          = np.zeros(employed_N, dtype=np.int32)
    employed_firms_location = np.zeros(employed_N, dtype=np.int32)
    employed_idx            = np.arange(employed_N)

    # Wijk-level population distribution for firm location sampling
    pop_count_by_wijk = np.bincount(wijk)
    popgrid_p         = pop_count_by_wijk / pop_count_by_wijk.sum()
    potential_locs    = np.uint32(np.arange(wijk.max() + 1))

    curr_assigned_firms_n   = np.int32(0)
    potential_max_firm_size = np.power(2, np.arange(10))  # [1, 2, 4, 8, ..., 512]
    curr_firm_id            = 0

    # ---- Process each firm-size bin [2^(i-1), 2^i) ---- #
    for i, max_firm_size in enumerate(potential_max_firm_size):
        if i == 0:
            # Skip the [0, 1) bin — firms need at least 1 employee
            continue

        min_firm_size = potential_max_firm_size[i - 1]

        # Fraction of total firms expected in this size range (Zipf CDF difference)
        size_firms_p = (
            (1 / min_firm_size + 1) ** wp_zipf_a
            - (1 / max_firm_size + 1) ** wp_zipf_a
        )
        size_firms_n = np.int32(np.around(size_firms_p * firms_n))

        curr_assigned_firms_n += size_firms_n
        if firms_n - curr_assigned_firms_n < 0:
            # We've planned more firms than the total; truncate and stop binning
            curr_assigned_firms_n -= size_firms_n
            max_firm_size          = potential_max_firm_size[i - 1]
            break

        # Draw firm sizes uniformly within this bin
        workplace_sizes = np.uint32(
            np.random.choice(np.arange(min_firm_size, max_firm_size), size_firms_n)
        )

        # Randomly select employees not yet assigned
        n_needed           = workplace_sizes.sum()
        potential_employees = np.random.choice(
            employed_idx[employed_firms == 0], n_needed, replace=False
        )

        if min_firm_size == 1:
            # Single-employee firms: mark as self-employed
            employed_firms[potential_employees] = -1
            continue

        # Sample firm locations weighted by wijk population
        workplace_locs        = np.random.choice(potential_locs, size=size_firms_n,
                                                  p=popgrid_p, replace=True)
        potential_employees_loc = employed_wijk[potential_employees]

        chosen_workplace_idx = choose_workplaces(
            potential_employees_loc, workplace_locs,
            dist_matrix, min_firm_size, max_firm_size=max_firm_size
        )

        # Drop anyone who couldn't be placed (radius exhausted)
        placed_mask         = chosen_workplace_idx > 0
        potential_employees = potential_employees[placed_mask]
        chosen_workplace_idx = chosen_workplace_idx[placed_mask]

        employed_firms_location[potential_employees] = workplace_locs[chosen_workplace_idx - 1]

        # Assign globally-unique firm IDs
        unique_wp, unique_wp_counts = np.unique(chosen_workplace_idx, return_counts=True)
        employed_firms[potential_employees] = chosen_workplace_idx + curr_firm_id

        print("  Firms [{:4d}–{:4d} employees]: {:,} firms, mean size {:.1f}".format(
            int(min_firm_size), int(max_firm_size),
            unique_wp.size, unique_wp_counts.mean()
        ))
        curr_firm_id += int(unique_wp.max())

    # ---- Final bin: remaining firms_n - curr_assigned_firms_n firms ---- #
    size_firms_n         = firms_n - curr_assigned_firms_n
    potential_employees  = employed_idx[employed_firms == 0]

    workplace_locs         = np.random.choice(potential_locs, size=size_firms_n,
                                               p=popgrid_p, replace=True)
    potential_employees_loc = employed_wijk[potential_employees]

    chosen_workplace_idx = choose_workplaces(
        potential_employees_loc, workplace_locs,
        dist_matrix, max_firm_size, max_firm_size=max_firm_size * 2
    )

    placed_mask          = chosen_workplace_idx > 0
    potential_employees  = potential_employees[placed_mask]
    chosen_workplace_idx = chosen_workplace_idx[placed_mask]

    employed_firms_location[potential_employees] = workplace_locs[chosen_workplace_idx - 1]

    unique_wp, unique_wp_counts = np.unique(chosen_workplace_idx, return_counts=True)
    employed_firms[potential_employees] = chosen_workplace_idx + curr_firm_id

    print("  Firms [final bin]: {:,} firms, mean size {:.1f}".format(
        unique_wp.size, unique_wp_counts.mean()
    ))

    # ---- Clean up unassigned and single-employee firms ---- #
    # Any employee still at 0 becomes self-employed
    employed_firms[employed_firms == 0] = -1

    # Firms with only 1 employee are also marked as self-employed
    unique_ids, id_counts = np.unique(employed_firms, return_counts=True)
    employed_firms[np.isin(employed_firms, unique_ids[id_counts == 1])] = -1

    # ---- Re-index firm IDs to be contiguous starting at 1 ---- #
    valid_ids   = np.unique(employed_firms[employed_firms > 0])
    new_ids     = np.uint32(np.arange(valid_ids.size)) + 1
    mapping_arr = np.zeros(valid_ids.max() + 1, dtype=np.uint32)
    mapping_arr[valid_ids] = new_ids
    employed_firms[employed_firms > 0] = mapping_arr[employed_firms[employed_firms > 0]]

    all_ids, all_counts = np.unique(employed_firms, return_counts=True)
    print("Average workplace size (all employed): {:.2f}".format(all_counts.mean()))

    return employed_firms, employed_firms_location


# ---------------------------------------------------------------------------
# Zipf parameter estimation
# ---------------------------------------------------------------------------

def _minobj_zipf_fn(A: float, N: int, alpha: float, s0: float) -> float:
    """
    Objective for fitting the Zipf scale parameter A.

    The expected number of firms with size >= s0 under a Zipf distribution
    with exponent alpha and scale A is compared to the observed count N.

    Parameters
    ----------
    A : float
        Scale parameter (maximum firm size).
    N : int
        Observed number of firms.
    alpha : float
        Zipf exponent.
    s0 : float
        Minimum firm size.

    Returns
    -------
    float
        Absolute difference between model expectation and N.
    """
    n = (alpha - 1) / alpha * (
        ((s0 / A) ** alpha - 1) / ((s0 / A) ** alpha - (s0 / A))
    )
    return abs(N - n)


def fit_zipf_A(N: int, alpha: float, s0: float = 1, x0: float = 1000) -> float:
    """
    Fit the Zipf scale parameter A given the number of firms N, exponent alpha,
    and minimum firm size s0.

    Parameters
    ----------
    N : int
        Target number of firms.
    alpha : float
        Zipf exponent.
    s0 : float
        Minimum firm size (default 1).
    x0 : float
        Initial guess for A (default 1000).

    Returns
    -------
    float
        Fitted value of A.

    Raises
    ------
    RuntimeError
        If the Nelder-Mead optimisation fails to converge.
    """
    result = minimize(
        _minobj_zipf_fn, x0=x0,
        args=(N, alpha, s0),
        method="Nelder-Mead",
        bounds=[(1e-6, 1e9)]
    )
    if result.success:
        return float(result.x[0])
    raise RuntimeError(f"fit_zipf_A failed to converge: {result}")


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def get_dist_and_area() -> tuple:
    """
    Build the inter-wijk centroid distance matrix and area vector from the
    CBS Wijk- en Buurtkaart 2020 shapefile and the household-composition
    spreadsheet.

    The shapefile path and Excel path are hard-coded relative to the project root.

    Returns
    -------
    dist_matrix : np.ndarray, shape (n_wijk, n_wijk), dtype float32
        Pairwise distances between wijk centroids (CRS units, typically metres).
        Diagonal is set to 500 m to avoid zero self-distance.
    area_vector : np.ndarray, shape (n_wijk,), dtype float32
        Area of each wijk in km².
    """
    shp_path = "data/WijkBuurtkaart_2020_v3/wijk_2020_v3.shp"
    hh_path  = "data/hh_counts_2021.xlsx"

    gdf = gpd.read_file(shp_path)
    # Fix encoding artifact in one wijk name
    gdf.loc[gdf["WK_NAAM"] == "ChassÃ©buurt", "WK_NAAM"] = "Chassébuurt"

    # Household data defines the set of wijken we model
    hh_df = (
        pd.read_excel(hh_path, sheet_name="WK_NAAM")
        .pipe(lambda df: df[df["WK_NAAM"].notna()])
        .pipe(lambda df: df[df["WK_NAAM"] != "Unknown"])
    )
    wijk_names = hh_df["WK_NAAM"].unique()
    n_wijk     = len(wijk_names)

    # Compute centroids and areas
    centroids   = [gdf[gdf["WK_NAAM"] == w].unary_union.centroid for w in wijk_names]
    area_vector = np.array(
        [gdf[gdf["WK_NAAM"] == w].unary_union.area / 1e6 for w in wijk_names],
        dtype=np.float32
    )

    # Build pairwise distance matrix
    dist_matrix = np.zeros((n_wijk, n_wijk), dtype=np.float32)
    for idx, centroid in enumerate(centroids):
        dist_matrix[idx, :] = centroid.distance(centroids)
        dist_matrix[idx, idx] = 500.0  # small non-zero self-distance

    return dist_matrix, area_vector
