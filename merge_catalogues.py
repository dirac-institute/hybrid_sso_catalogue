from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
from dask.distributed import Client, progress
import dask


def get_catalogues(s3m_path="catalogues/s3m_propagated.h5",
                   mpcorb_path="catalogues/mpcorb_propagated.h5"):
    
    s3m = pd.read_hdf(s3m_path, key="df")
    mpcorb = pd.read_hdf(mpcorb_path, key="df")
    return s3m, mpcorb


@dask.delayed
def merge_catalogues(sim, real, min_mag, max_mag, k=100, d_max=0.1):
    """Merge the simulated solar system catalogue with the real MPCORB data for a
    certain magnitude bin.
    
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
        
    k : `int`
        Maximum number of neighbours to find
        
    d_max : `float`
        Maxmimum distance within which to find neighbours
        
    Returns
    -------
    taken_ids : `float/array`
        An array of the ids that have been replaced by the real objects in this
        magnitude bin
        
    no_match_count : `int`
        A count of the number of systems that had no matches (and thus must be
        added directly to the simulated catalogue)
    """
    real_xyz = np.array([real["x"].values, real["y"].values, real["z"].values]).T
    sim_xyz = np.array([sim["x"].values, sim["y"].values, sim["z"].values]).T
    
    v_sim = np.array([sim.vx.values, sim.vy.values, sim.vz.values])
    v_real = np.array([real.vx.values, real.vy.values, real.vz.values])
    
    # get the matching simulated data and build a K-D Tree
    sim_mag_mask = np.logical_and(sim.H >= min_mag, sim.H < max_mag)
    sim_id = sim.id[sim_mag_mask].values
    tree = cKDTree(sim_xyz[sim_mag_mask])
    
    # get the matching real data from MPCORB
    real_mag_mask = np.logical_and(real.H >= min_mag, real.H <= max_mag)
    real_objects = real_xyz[real_mag_mask]
    
    # keep track of objects already assigned and a count of how many had no matches
    taken = []
    no_match_count = 0
    
    # iterate over every object in the real catalogue
    for obj in real_objects:
        
        # find the nearest k neighbours within d_max and mask further neighbours
        distances, inds = tree.query(obj, k=k, distance_upper_bound=d_max)
        distances, inds = distances[np.isfinite(distances)], inds[np.isfinite(distances)]
        
        # get only the options which haven't yet been assigned
        unassigned_inds = np.setdiff1d(inds, taken, assume_unique=True)

        # if there are many matching object
        if len(unassigned_inds) > 0:
            # find the closest velocity of the bunch and assign it
            best = np.sum((v_sim[:, unassigned_inds] - v_real[:, unassigned_inds])**2, axis=0).argmin()
            taken.append(unassigned_inds[best])
        
        # if only one then just immediately assign it
        elif len(unassigned_inds) == 1:
            taken.append(unassigned_inds[0])
        
        # otherwise then there was no match
        else:
            no_match_count += 1
            
    np.save("output/matched_{}_{}.npy".format(min_mag, max_mag), np.array(sim_id[taken]))
            
    return np.array(sim_id[taken])


if __name__ == "__main__":
    # start the Dask client
    client = Client(n_workers=48, threads_per_worker=1, memory_limit='16GB')
    
    # get the catalogues
    s3m, mpcorb = get_catalogues()
    
    # loop over magnitude bins and add them to the Dask pool thing
    H_bins = np.arange(-2, 28 + 1)
    output = []
    for i in range(len(H_bins) - 2):
        output.append(merge_catalogues(s3m, mpcorb, i, i + 1))
        
    results = client.compute(output)
    progress(*results)
    np.save("output/all.npy", results)