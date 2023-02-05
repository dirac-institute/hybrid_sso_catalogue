import pandas as pd
import h5py as h5
import numpy as np

all_ids = np.array([])
all_nfs = np.array([])

for i in range(3637):
    # get this night's values
    this_ids = np.load(f"difi_MBAs_ids_{i:04d}.npy")
    this_nfs = np.load(f"difi_MBAs_ids_{i:04d}.npy")

    # combine it with the previous nights'
    all_ids = np.concatenate([all_ids, this_ids])
    all_nfs = np.concatenate([all_nfs, this_nfs])

    # sort everything by the night on which it was found
    order = np.argsort(all_nfs)
    all_ids = all_ids[order]
    all_nfs = all_nfs[order]

    # get only the unique ids and the corresponding nights
    all_ids, idx = np.unique(all_ids, return_index=True)
    all_nfs = all_nfs[idx]

    print(i)

np.save("all_obj_ids.npy", all_ids)
np.save("all_night_found.npy", all_nfs)