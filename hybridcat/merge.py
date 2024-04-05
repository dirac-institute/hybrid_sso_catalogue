from scipy.spatial import cKDTree
import numpy as np
from multiprocessing import Pool


def merge_magnitude_bin(sim, real, min_mag, max_mag, k=100, d_max=0.1, output_folder="output/"):
    """Merge the simulated solar system catalogue with the real MPCORB data for a certain magnitude bin.

    Parameters
    ----------
    sim : `pandas DataFrame`
        Dataframe with the simulated catalogue (must contain x,y,z,vx,vy,vz and H)
    real : `pandas DataFrame`
        Dataframe with the real catalogue (must contain x,y,z,vx,vy,vz and H)
    min_mag : `float`
        Minimum magnitude to consider for this merge
    max_mag : `float`
        Maximum magnitude to consider for this merge
    k : `int`, optional
        Maximum number of neighbours to find, by default 100
    d_max : `float`, optional
        Maximum distance within which to find neighbours in AU, by default 0.1
    output_folder : `str`, optional
        Path to the folder in which output will be stored, by default "output/"

    Returns
    -------
    taken_ids : `float/array`
        An array of the ids that have been replaced by the real objects in this magnitude bin
    """
    print(f"Merging bin {min_mag}-{max_mag}")
    real_xyz = np.array([real["x"].values, real["y"].values, real["z"].values]).T
    sim_xyz = np.array([sim["x"].values, sim["y"].values, sim["z"].values]).T

    v_sim = np.array([sim.vx.values, sim.vy.values, sim.vz.values])
    v_real = np.array([real.vx.values, real.vy.values, real.vz.values]).T

    # get the matching simulated data and build a K-D Tree
    sim_mag_mask = np.logical_and(sim.H >= min_mag, sim.H < max_mag)
    sim_id = sim.id[sim_mag_mask].values
    tree = cKDTree(sim_xyz[sim_mag_mask])

    # get the matching real data from MPCORB
    real_mag_mask = np.logical_and(real.H >= min_mag, real.H <= max_mag)
    real_objects = real_xyz[real_mag_mask]
    real_velocities = v_real[real_mag_mask]

    # keep track of objects already assigned
    taken = []

    # iterate over every object in the real catalogue
    for obj, vel in zip(real_objects, real_velocities):

        # find the nearest k neighbours within d_max and mask further neighbours
        distances, inds = tree.query(obj, k=k, distance_upper_bound=d_max)
        distances, inds = distances[np.isfinite(distances)], inds[np.isfinite(distances)]

        # get only the options which haven't yet been assigned
        unassigned_inds = np.setdiff1d(inds, taken, assume_unique=True)

        # if there are many matching object
        if len(unassigned_inds) > 0:
            # find the closest velocity of the bunch and assign it
            best = np.sum((v_sim[:, unassigned_inds] - vel[:, np.newaxis])**2, axis=0).argmin()
            taken.append(unassigned_inds[best])

        # if only one then just immediately assign it
        elif len(unassigned_inds) == 1:
            taken.append(unassigned_inds[0])

    np.save(f"{output_folder}matched_{min_mag}_{max_mag}.npy", np.array(sim_id[taken]))
            
    return np.array(sim_id[taken])

def merge_catalogues(mpcorb, s3m, output_folder="output/", H_bins=np.arange(-2, 28 + 1), n_workers=48,
                     k=100, d_max=0.1):
    """Merge mpcorb and s3m!

    Parameters
    ----------
    mpcorb : `Pandas Dataframe`
        MPCORB catalogue
    s3m : `Pandas DataFrame`
        S3m catalogue
    output_folder : `str`, optional
        Path to the output folder, by default "output/"
    H_bins : `list`, optional
        Magnitude bins, by default np.arange(-2, 28 + 1)
    n_workers : `int`, optional
        How many workers to use, by default 48
    k : `int`, optional
        Number of neighbours to consider in matching, by default 100
    d_max : `float`, optional
        Maximum distance a neighbour can occur at in AU, by default 0.1

    Returns
    -------
    matched_ids : `list`
        The IDs of S3m objects that can be replaced by MPCORB ones
    """
    def args(H_bins):
        for left, right in zip(H_bins[:-1], H_bins[1:]):
            yield s3m, mpcorb, left, right, k, d_max, output_folder

    with Pool(n_workers) as pool:
        results = list(pool.starmap(merge_magnitude_bin, args(H_bins)))
    return results
