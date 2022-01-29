import numpy as np
import h5py as h5
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
from os.path import isfile


def create_mpcorb_from_json(in_path="../catalogues/mpcorb_extended.json",
                            out_path="../catalogues/mpcorb_initial.h5", force=False):

    if not isfile(out_path) or force:
        mpcorb_df = pd.read_json(in_path)

        # only select pertinent columns
        mpcorb_df = mpcorb_df[["H", "G", "Principal_desig", "Epoch",
                            "M", "Peri", "Node", "i", "e", "a"]]

        # adjust to more similar column names
        mpcorb_df.rename(columns={"Principal_desig": "des", "Epoch": "t_0",
                                "M": "mean_anom", "Peri": "argperi", "Node": "Omega"},
                        inplace=True)

        # adjust to modified JD
        mpcorb_df.t_0 = mpcorb_df.t_0 - 2400000.5
        mpcorb_df.to_hdf(out_path, mode="w", key="df")
    else:
        mpcorb_df = pd.read_hdf(out_path, key="df")
    return mpcorb_df


def create_s3m_from_files(in_path="../catalogues/original_data/s3m/", out_path="../catalogues/initial/s3m.h5"):
    names_s3m = ['id', 'format', 'q', 'e', 'i', 'Omega', 'argperi', 't_p',
                'H', 't_0', 'INDEX', 'N_PAR', 'MOID', 'COMPCODE']

    files_s3m = ['S0', 'S1_00', 'S1_01', 'S1_02', 'S1_03', 'S1_04', 'S1_05',
                'S1_06', 'S1_07', 'S1_08', 'S1_09', 'S1_10', 'S1_11', 'S1_12',
                'S1_13', "SL", "St5", "ST"]

    dfs = [None for i in range(len(files_s3m))]

    for i in range(len(files_s3m)):
        print(files_s3m[i])
        dfs[i] = pd.read_csv(s3m_data_path + "{}.s3m".format(files_s3m[i]),
                            comment="!", delim_whitespace=True,
                            header=None, names=names_s3m, skiprows=2)

    s3m_df = pd.concat(dfs)
    # make sure the indices are unique
    s3m_df.set_index("id", inplace=True)

    # drop anything with e = 1.0 because openorb can't do it
    s3m_df.drop(labels=s3m_df.index[s3m_df.e == 1.0], axis=0, inplace=True)

    # save to hdf5
    s3m_df.to_hdf(s3m_h5_path, key="df", mode="w")