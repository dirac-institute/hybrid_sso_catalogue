import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.collections import PatchCollection
import time

import thor
from thor.backend import PYOORB
backend = PYOORB()

from variant_orbits import variant_orbit_ephemerides
from scheduling import get_LSST_schedule
from magnitudes import convert_colour_mags

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
    unique_objs : `list`
        List of unique hex ids that have digest2 > 65 that were observed on `night_start`
    """
    # create a list of nights in the detection window and get schedule for them
    night_list = list(range(night_start, night_start + detection_window))
    schedule = get_LSST_schedule(night=(night_start, night_start + detection_window - 1))

    # offset the schedule by one row and re-merge to get the previous night column
    shifted = schedule.shift()
    shifted = shifted.drop("observationStartMJD", axis=1)
    shifted = shifted.rename(columns={"night": "previousNight"})
    full_schedule = pd.merge(schedule, shifted["previousNight"], left_index=True, right_index=True)

    # calculate the length of each night in days
    night_lengths = np.zeros(detection_window)
    for i, night in enumerate(night_list):
        mask = full_schedule["night"] == night

        # ignore nights that have no observations (bad weather/downtime)
        if not full_schedule[mask].empty:
            night_lengths[i] = full_schedule[mask].iloc[-1]["observationStartMJD"]\
                - full_schedule[mask].iloc[0]["observationStartMJD"]

    # get the first/last visit from each night
    night_transition = full_schedule["night"] != full_schedule["previousNight"]
    first_visit_times = full_schedule[night_transition]["observationStartMJD"].values

    last_times_ind = np.array(list(full_schedule[night_transition].index[1:]) + [len(full_schedule)]) - 1
    last_visit_times = full_schedule.loc[last_times_ind]["observationStartMJD"].values

    file = find_first_file(night_list)
    visit_file = pd.read_hdf(path + f"filtered_visit_scores_{file:03d}.h5")

    # get the objects from the night
    obs_mask = np.logical_and(visit_file["night"] == night_start, visit_file["scores"] >= 65)
    sorted_obs = visit_file[obs_mask].sort_values(["ObjID", "FieldMJD"])
    unique_objs = sorted_obs.index.unique()

    print("Everything is prepped and ready for probability calculations")

    # calculate detection probabilities
    probs = np.zeros(len(unique_objs))
    for i, hex_id in enumerate(unique_objs):
        start = time.time()
        probs[i] = probability_from_id(hex_id, sorted_obs, distances=np.logspace(-1, 1, 50) * u.AU,
                                       radial_velocities=np.linspace(-100, 100, 20) * u.km / u.s,
                                       first_visit_times=first_visit_times, full_schedule=full_schedule,
                                       night_lengths=night_lengths, night_list=night_list,
                                       detection_window=detection_window, min_nights=min_nights)
        print(f"{i}/{len(unique_objs)}: {time.time() - start:1.2f}, {hex_id}, {probs[i]:1.3f}")

    return probs, unique_objs


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
    rows = sorted_obs.loc[hex_id]
    reachable_schedule = get_reachable_schedule(rows, first_visit_times, night_list,
                                                night_lengths, full_schedule)

    v_mags = [convert_colour_mags(r["MaginFilter"],
                                  in_colour=r["filter"], out_colour="V") for _, r in rows.iterrows()]
    apparent_mag = np.mean(v_mags)

    # get the orbits for the entire reachable schedule with the grid of distances and RVs
    orbits = variant_orbit_ephemerides(ra=rows.iloc[0]["AstRA(deg)"] * u.deg,
                                       dec=rows.iloc[0]["AstDec(deg)"] * u.deg,
                                       ra_end=rows.iloc[-1]["AstRA(deg)"] * u.deg,
                                       dec_end=rows.iloc[-1]["AstDec(deg)"] * u.deg,
                                       delta_t=(rows.iloc[-1]["FieldMJD"] - rows.iloc[0]["FieldMJD"]) * u.day,
                                       obstime=Time(rows.iloc[0]["FieldMJD"], format="mjd"),
                                       distances=distances,
                                       radial_velocities=radial_velocities,
                                       apparent_mag=apparent_mag,
                                       eph_times=Time(reachable_schedule["observationStartMJD"].values,
                                                      format="mjd"),
                                       only_neos=True)
    orbits["orbit_id"] = orbits["orbit_id"].astype(int)
    orbit_ids = orbits["orbit_id"].unique()

    # merge the orbits with the schedule
    joined_table = pd.merge(orbits, reachable_schedule, left_on="mjd_utc", right_on="observationStartMJD")
    
    # compute filter magnitudes
    mag_in_filter = np.ones(len(joined_table)) * np.inf
    for filter_letter in "ugrizy":
        filter_mask = joined_table["filter"] == filter_letter
        if not filter_mask.any():
            continue
        mag_in_filter[filter_mask] = convert_colour_mags(joined_table[filter_mask]["VMag"],
                                                         out_colour=filter_letter,
                                                         in_colour="V", convention="LSST",
                                                         asteroid_type="C")
    joined_table["mag_in_filter"] = mag_in_filter

    # mask those that are within the field (2.1 degrees)
    in_current_field = np.sqrt((joined_table["fieldRA"]
                                - joined_table["RA_deg"])**2
                               + (joined_table["fieldDec"]
                                  - joined_table["Dec_deg"])**2) <= 2.1

    bright_enough = joined_table["mag_in_filter"] < joined_table["fiveSigmaDepth"]

    joined_table["observed"] = (in_current_field & bright_enough).astype(int)

    # remove any nights that don't match requirements (min_obs, min_arc, max_time)
    df = joined_table[joined_table["observed"] != 0]
    mask = df.groupby(["orbit_id", "night"]).apply(filter_tracklets)
    df_multiindex = df.set_index(["orbit_id", "night"]).sort_index()
    filtered_obs = df_multiindex.loc[mask[mask].index].reset_index()

    # decide whether each orbit is findable
    N_ORB = len(orbit_ids)
    findable = np.repeat(False, N_ORB)
    for i, orbit_id in enumerate(orbit_ids):
        this_orbit = filtered_obs[filtered_obs["orbit_id"] == orbit_id]

        # if the orbit actually exists (if it hasn't been filtered out)
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
                            findable[i] = True
                            break

                    # otherwise reset the counter and start at this night
                    else:
                        counter = 1
                        init = night

    # return the fraction of orbits that are findable
    return findable.astype(int).sum() / N_ORB


def get_reachable_schedule(rows, first_visit_times, night_list, night_lengths, full_schedule):
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
    return pd.concat(masked_schedules)


def first_last_pos_from_id(hex_id, sorted_obs, s3m_cart, distances, radial_velocities,
                           first_visit_times, last_visit_times):
    rows = sorted_obs.loc[hex_id]

    eph_times = Time(np.sort(np.concatenate([first_visit_times, last_visit_times])), format="mjd")

    orbits = variant_orbit_ephemerides(ra=rows.iloc[0]["AstRA(deg)"] * u.deg,
                                       dec=rows.iloc[0]["AstDec(deg)"] * u.deg,
                                       ra_end=rows.iloc[-1]["AstRA(deg)"] * u.deg,
                                       dec_end=rows.iloc[-1]["AstDec(deg)"] * u.deg,
                                       delta_t=(rows.iloc[-1]["FieldMJD"] - rows.iloc[0]["FieldMJD"]) * u.day,
                                       obstime=Time(rows.iloc[0]["FieldMJD"], format="mjd"),
                                       distances=distances,
                                       radial_velocities=radial_velocities,
                                       eph_times=eph_times,
                                       only_neos=True)
    orbits["orbit_id"] = orbits["orbit_id"].astype(int)

    item = s3m_cart[s3m_cart["hex_id"] == hex_id]
    orb_class = thor.Orbits(orbits=np.atleast_2d(np.concatenate(([item["x"], item["y"], item["z"]],
                                                                 [item["vx"], item["vy"], item["vz"]]))).T,
                            epochs=Time(item["t_0"], format="mjd"))
    truth = backend.generateEphemeris(orbits=orb_class, observers={"I11": Time(eph_times, format="mjd")})

    return orbits, truth


def plot_LSST_schedule_with_orbits(schedule, reachable_schedule, orbits, truth, night,hex_id,
                                   colour_by="distance", lims="full_schedule", field_radius=2.1, s=10,
                                   filter_mask="all", show_mag_labels=False,
                                   fig=None, ax=None, show=True, ax_labels=True, cbar=True):
    """Plot LSST schedule up using the dataframe containing fields. Each is assumed to be a circle for
    simplicity.

    Parameters
    ----------
    df : `pandas DataFrame`
        DataFrame of fields (see `get_LSST_schedule`)
    """
    # create the figure with equal aspect ratio
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_aspect("equal")

    # check that there were observations in this night
    orbits["night"] = (orbits["mjd_utc"] - 0.5).astype(int) - 59638
    mask = orbits["night"] == night
    if not np.any(mask):
        if lims == "full_schedule":
            ax.set_xlim(schedule["fieldRA"].min() - 3,
                        schedule["fieldRA"].max() + 3)
            ax.set_ylim(schedule["fieldDec"].min() - 3,
                        schedule["fieldDec"].max() + 3)
        print("Warning: No observations in this night")
        return None, None

    # plot each schedule with difference colours and widths
    for table, colour, lw in zip([schedule, reachable_schedule], ["black", "tab:green"], [1, 2]):
        # ensure you only get the current night
        table_mask = table["night"] == night

        # filter table (pun intended)
        if filter_mask != "all":
            table_mask &= table["filter"] == filter_mask

        ra_field = table["fieldRA"][table_mask]
        dec_field = table["fieldDec"][table_mask]
        patches = [plt.Circle(center, field_radius) for center in np.transpose([ra_field, dec_field])]
        coll = PatchCollection(patches, edgecolors=colour, facecolors="none", linewidths=lw)
        ax.add_collection(coll)

        # if we're doing the full schedule
        if colour == "black":
            ax.annotate(f"{len(table[table_mask])} fields", xy=(0.98, 0.98), xycoords="axes fraction",
                        ha="right", va="top", fontsize=20)

            obs_dfs = [pd.read_hdf(f"../neocp/neo/filtered_visit_scores_{i:03d}.h5").sort_values("FieldMJD")[["FieldMJD", "night", "MaginFilter", "filter"]]
               for i in [0, 1]]
            all_obs = pd.concat(obs_dfs)
            all_obs.reset_index(inplace=True)
            nightly_obs = all_obs[all_obs["night"] == night]

            if not nightly_obs.empty:
                det_times = nightly_obs[nightly_obs["hex_id"] == hex_id]["FieldMJD"].values

                if len(det_times) != 0:
                    ids = [(thing - table[table_mask]["observationStartMJD"][table[table_mask]["observationStartMJD"] <= thing]).idxmin() for thing in det_times]
                    det_fields = table.loc[ids]

                    ra_field = det_fields["fieldRA"]
                    dec_field = det_fields["fieldDec"]
                    patches = [plt.Circle(center, field_radius * 0.8) for center in np.transpose([ra_field, dec_field])]
                    coll = PatchCollection(patches, edgecolors="#13f2a8", facecolors="none", linewidths=2)
                    ax.add_collection(coll)

                    ax.annotate(f"{len(det_times)} observations", xy=(0.98, 0.93), xycoords="axes fraction",
                                ha="right", va="top", fontsize=20, color="#13f2a8")

            if show_mag_labels:

                print("Previous magnitudes:", night)
                prev_mags = all_obs[np.logical_and(all_obs["hex_id"] == hex_id, all_obs["night"] <= night)].copy()
                prev_v_band = [convert_colour_mags(r["MaginFilter"], in_colour=r["filter"], out_colour="V") for i, r in prev_mags.iterrows()]
                prev_mags["VMag"] = prev_v_band
                print(prev_mags)


                # build a dictionary of magnitude labels based on field position
                mag_labels = {}
                for _, visit in table[table_mask].iterrows():
                    # create tuple of position for dict key
                    xy = (visit["fieldRA"], visit["fieldDec"])

                    v_mag = convert_colour_mags(visit["fiveSigmaDepth"], out_colour="V", in_colour=visit["filter"])

                    # append or create each dict item
                    if xy in mag_labels:
                        mag_labels[xy] += f'\n{visit["filter"]}{visit["fiveSigmaDepth"]:.2f},v{v_mag:.2f}'
                    else:
                        mag_labels[xy] = f'{visit["filter"]}{visit["fiveSigmaDepth"]:.2f},v{v_mag:.2f}'

                # go through each unique field position and add an annotation
                for xy, label in mag_labels.items():
                    ax.annotate(label, xy=xy, ha="center", va="center", fontsize=8)

    # if colouring by orbit then just use a plain old colourbar
    if colour_by == "orbit":
        ax.scatter(orbits["RA_deg"][mask], orbits["Dec_deg"][mask],
                   s=s, alpha=1, c=orbits["orbit_id"][mask])
        scatter = ax.scatter(truth["RA_deg"][mask], truth["Dec_deg"][mask], s=s * 10, c="tab:red")
    # if distance then use a log scale for the colourbar
    elif colour_by == "distance":
        log_dist_from_earth = np.log10(orbits["delta_au"])

        boundaries = np.arange(-1, 1.1 + 0.2, 0.2)
        norm = BoundaryNorm(boundaries, plt.cm.plasma_r.N, clip=True)

        for orb in orbits[mask]["orbit_id"].unique():
            more_mask = orbits[mask]["orbit_id"] == orb
            ax.plot(orbits["RA_deg"][mask][more_mask], orbits["Dec_deg"][mask][more_mask],
                    color=plt.cm.plasma_r(norm(log_dist_from_earth[mask][more_mask].iloc[0])))

        scatter = ax.scatter(orbits["RA_deg"][mask], orbits["Dec_deg"][mask], s=s,
                             c=log_dist_from_earth[mask], norm=norm, cmap="plasma_r")

        if cbar:
            fig.colorbar(scatter, label="Log Geocentric Distance [AU]")

        scatter = ax.scatter(truth["RA_deg"][mask], truth["Dec_deg"][mask], s=s, c="#13f2a8", marker="x")
        ax.plot(truth["RA_deg"][mask], truth["Dec_deg"][mask], color="#13f2a8")
    else:
        raise ValueError("Invalid value for colour_by")

    # if limited by the schedule then adjust the limits
    if lims in ["schedule", "reachable"]:
        table = schedule if lims == "schedule" else reachable_schedule
        table_mask = table["night"] == night
        if filter_mask != "all":
            table_mask &= table["filter"] == filter_mask

        if not table[table_mask].empty:
            ax.set_xlim(table[table_mask]["fieldRA"].min() - 3,
                        table[table_mask]["fieldRA"].max() + 3)
            ax.set_ylim(table[table_mask]["fieldDec"].min() - 3,
                        table[table_mask]["fieldDec"].max() + 3)
    elif lims == "full_schedule":
        ax.set_xlim(schedule["fieldRA"].min() - 3,
                    schedule["fieldRA"].max() + 3)
        ax.set_ylim(schedule["fieldDec"].min() - 3,
                    schedule["fieldDec"].max() + 3)
    elif lims == "orbits":
        ax.set_xlim(orbits["RA_deg"][mask].min() - 3,
                    orbits["RA_deg"][mask].max() + 3)
        ax.set_ylim(orbits["Dec_deg"][mask].min() - 3,
                    orbits["Dec_deg"][mask].max() + 3)
    else:
        raise ValueError("Invalid input for lims")

    # label the axes, add a grid, show the plot
    if ax_labels:
        ax.set_xlabel("Right Ascension [deg]")
        ax.set_ylabel("Declination [deg]")
    ax.grid()

    if show:
        plt.show()

    return fig, ax
