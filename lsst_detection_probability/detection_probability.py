import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np

from variant_orbits import variant_orbit_ephemerides
from scheduling import get_LSST_schedule

import sys
sys.path.append("../neocp")
from lsst_neocp import find_first_file


def filter_tracklets(df, min_obs=2, min_arc=1, max_time=90):
    init = SkyCoord(ra=df["RA_deg"].iloc[0], dec=df["Dec_deg"].iloc[0], unit="deg")
    final = SkyCoord(ra=df["RA_deg"].iloc[-1], dec=df["Dec_deg"].iloc[-1], unit="deg")

    return np.logical_and.reduce((len(df) >= min_obs,
                                  init.separation(final).to(u.arcsecond).value > min_arc,
                                  df["mjd_utc"].diff().min() * 1440 < max_time))


def get_detection_probabilities(night_start, path="../neocp/neo/", detection_window=15, min_nights=3):
    """Get the probability that LSST will detect each object that was observed in a particular night

    Parameters
    ----------
    night_start : `int`
        Night of the initial observations
    path : `str`, optional
        Path the observation file, by default "../neocp/neo/"
    detection_window : `int`, optional
        How many days in the detection window, by default 15
    min_nights : `int`, optional
        Minimum number of nights on which observations need to have occurred, by default 3

    Returns
    -------
    probs : `list`
        Estimated probability that each object will be detected by LSST alone
    will_be_detected : `list`
        Truth of whether each object will be detected by LSST alone
    """
    # create a list of nights in the detection window and get schedule for them
    night_list = list(range(night_start, night_start + detection_window))
    schedule = get_LSST_schedule(night=(night_start, night_start + detection_window - 1))

    # offset the schedule by one row and re-merge to get the previous night column
    shifted = schedule.shift()
    shifted = shifted.drop("observationStartMJD", axis=1)
    shifted = shifted.rename(columns={"fieldRA": "previousFieldRA", "fieldDec": "previousFieldDec",
                                      "night": "previousNight"})
    full_schedule = pd.merge(schedule, shifted, left_index=True, right_index=True)

    # calculate the length of each night in days
    night_lengths = np.zeros(detection_window)
    for i, night in enumerate(night_list):
        mask = full_schedule["night"] == night

        # ignore nights that have no observations (bad weather/downtime)
        if not full_schedule[mask].empty:
            night_lengths[i] = full_schedule[mask].iloc[-1]["observationStartMJD"]\
                - full_schedule[mask].iloc[0]["observationStartMJD"]

    # get the first visit from each night
    night_transition = full_schedule["night"] != full_schedule["previousNight"]
    first_visit_times = full_schedule[night_transition]["observationStartMJD"].values

    file = find_first_file(night_list)
    visit_file = pd.read_hdf(path + f"filtered_visit_{file:03d}.h5")

    # get the objects from the night
    sorted_obs = visit_file[visit_file["night"] == night_start].sort_values(["ObjID", "FieldMJD"])
    unique_objs = sorted_obs["hex_id"].unique()

    # get the truth
    unique_findable_neo_hex_ids = np.load("../neocp/unique_findable_neo_hex_ids_linked.npy",
                                          allow_pickle=True)
    will_be_detected = np.isin(unique_objs, unique_findable_neo_hex_ids)

    print("Everything is prepped and ready for probability calculations")

    # calculate detection probabilities
    probs = np.zeros(len(unique_objs))
    for i, hex_id in enumerate(unique_objs):
        probs[i] = probability_from_id(hex_id, sorted_obs, distances=np.logspace(-1, 1.1, 50) * u.AU,
                                       radial_velocities=np.linspace(-100, 100, 10) * u.km / u.s,
                                       first_visit_times=first_visit_times, full_schedule=full_schedule,
                                       night_lengths=night_lengths, night_list=night_list,
                                       detection_window=detection_window, min_nights=min_nights)
        print(hex_id, probs[i])

    return probs, will_be_detected


def probability_from_id(hex_id, sorted_obs, distances, radial_velocities, first_visit_times,
                        full_schedule, night_lengths, night_list, detection_window=15, min_nights=3):
    """Get the probability of an object with a particular ID of being detected by LSST alone given
    observations on a single night.

    Parameters
    ----------
    hex_id : `str`
        ID of the object (in hex format)
    sorted_obs : `pandas DataFrame`
        DataFrame of the sorted observations from the initial night
    distances : `list`
        List of distances to consider
    radial_velocities : `list`
        List of radial velocities to consider
    first_visit_times : `list`
        Times at which each night has its first visit
    full_schedule : `pandas DataFrame`
        Full schedule of visits for the entire detection window
    night_lengths : `list`
        Length of each night of observations in days
    night_list : `list`
        List of the nights in the detection window
    detection_window : `int`, optional
        Length of the detection window in days, by default 15
    min_nights : `int`, optional
        Minimum number of nights required for a detection, by default 3

    Returns
    -------
    probs : `list`
        Estimated probability that the object will be detected by LSST alone
    """
    # get the matching rows and ephemerides for start of each night
    rows = sorted_obs[sorted_obs["hex_id"] == hex_id]
    start_orbits = variant_orbit_ephemerides(ra=rows.iloc[0]["AstRA(deg)"] * u.deg,
                                             dec=rows.iloc[0]["AstDec(deg)"] * u.deg,
                                             ra_end=rows.iloc[-1]["AstRA(deg)"] * u.deg,
                                             dec_end=rows.iloc[-1]["AstDec(deg)"] * u.deg,
                                             delta_t=(rows.iloc[-1]["FieldMJD"]
                                                      - rows.iloc[0]["FieldMJD"]) * u.day,
                                             obstime=Time(rows.iloc[0]["FieldMJD"], format="mjd"),
                                             distances=[1] * u.AU,
                                             radial_velocities=[2] * u.km / u.s,
                                             eph_times=Time(first_visit_times, format="mjd"))

    # create some nominal field size
    FIELD_SIZE = 2.1 * 5

    # mask the schedule to things that can be reached on each night
    masked_schedules = [pd.DataFrame() for i in range(len(night_list))]
    for j in range(len(start_orbits)):
        delta_ra = start_orbits.loc[j]["vRAcosDec"] / np.cos(start_orbits.loc[j]["Dec_deg"] * u.deg)\
            * night_lengths[j]
        delta_dec = start_orbits.loc[j]["vDec"] * night_lengths[j]

        ra_lims = sorted([start_orbits.loc[j]["RA_deg"], start_orbits.loc[j]["RA_deg"] + delta_ra.value])
        ra_lims = [ra_lims[0] - FIELD_SIZE, ra_lims[-1] + FIELD_SIZE]
        dec_lims = sorted([start_orbits.loc[j]["Dec_deg"], start_orbits.loc[j]["Dec_deg"] + delta_dec])
        dec_lims = [dec_lims[0] - FIELD_SIZE, dec_lims[-1] + FIELD_SIZE]

        night = (start_orbits.loc[j]["mjd_utc"] - 0.5).astype(int) - 59638

        mask = full_schedule["night"] == night
        within_lims = np.logical_and.reduce((full_schedule[mask]["fieldRA"] > ra_lims[0],
                                             full_schedule[mask]["fieldRA"] < ra_lims[1],
                                             full_schedule[mask]["fieldDec"] > dec_lims[0],
                                             full_schedule[mask]["fieldDec"] < dec_lims[1]))
        masked_schedules[night_list.index(night)] = full_schedule[mask][within_lims]
    # combine into a single reachable schedule
    reachable_schedule = pd.concat(masked_schedules)

    # get the orbits for the entire reachable schedule with the grid of distances and RVs
    orbits = variant_orbit_ephemerides(ra=rows.iloc[0]["AstRA(deg)"] * u.deg,
                                       dec=rows.iloc[0]["AstDec(deg)"] * u.deg,
                                       ra_end=rows.iloc[-1]["AstRA(deg)"] * u.deg,
                                       dec_end=rows.iloc[-1]["AstDec(deg)"] * u.deg,
                                       delta_t=(rows.iloc[-1]["FieldMJD"] - rows.iloc[0]["FieldMJD"]) * u.day,
                                       obstime=Time(rows.iloc[0]["FieldMJD"], format="mjd"),
                                       distances=distances,
                                       radial_velocities=radial_velocities,
                                       eph_times=Time(reachable_schedule["observationStartMJD"].values,
                                                      format="mjd"))
    orbits["orbit_id"] = orbits["orbit_id"].astype(int)

    # merge the orbits with the schedule
    joined_table = pd.merge(orbits, reachable_schedule, left_on="mjd_utc", right_on="observationStartMJD")

    # mask those that are within the field (2.1 degrees)
    in_current_field = np.sqrt((joined_table["fieldRA"]
                                - joined_table["RA_deg"])**2
                               + (joined_table["fieldDec"]
                                  - joined_table["Dec_deg"])**2) <= 2.1
    joined_table["observed"] = in_current_field.astype(int)

    # remove any nights that don't match requirements (min_obs, min_arc, max_time)
    df = joined_table[joined_table["observed"] != 0]
    mask = df.groupby(["orbit_id", "night"]).apply(filter_tracklets)
    df_multiindex = df.set_index(["orbit_id", "night"]).sort_index()
    filtered_obs = df_multiindex.loc[mask[mask].index].reset_index()

    # decide whether each orbit is findable
    N_ORB = len(distances) * len(radial_velocities)
    findable = np.repeat(False, N_ORB)
    for orbit_id in range(N_ORB):
        this_orbit = filtered_obs[filtered_obs["orbit_id"] == orbit_id]

        # if the orbit actually exists (if obs exist)
        if not this_orbit.empty:
            # check how many nights it is observed on and require the min nights
            unique_nights = np.sort(this_orbit["night"].unique())
            if len(unique_nights) >= min_nights:
                # start a count of how many nights have happened in the detection window
                init = unique_nights[0]
                counter = 1
                for night in unique_nights[1:]:
                    # if you're still within the detection window
                    if night - init <= detection_window:
                        # add to the counter
                        counter += 1

                        # if you've reached the min nights then mark it as findable
                        if counter == min_nights:
                            findable[orbit_id] = True
                            break

                    # otherwise reset the counter and start at this night
                    else:
                        counter = 1
                        init = night

    # return the fraction of orbits that are findable
    return findable.astype(int).sum() / N_ORB
