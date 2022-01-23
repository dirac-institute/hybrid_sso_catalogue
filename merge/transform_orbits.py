import numpy as np
import h5py as h5
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt

import pickle
pickle.HIGHEST_PROTOCOL = 4

import pandas as pd
import os.path

import pyoorb as oo


def df_to_orbits(df, coord_system="KEP"):
    """Convert a DataFrame to orbits that can be used by OpenOrb

    Parameters
    ----------
    df : `Pandas DataFrame`
        The dataframe that you want to convert
    coord_system : `str`, optional
        Which coordinate system `df` uses ("CART", "COM" or "KEP"), by default "KEP"

    Returns
    -------
    orbits : `np.array`
        Orbits for OpenOrb

    Raises
    ------
    ValueError
        Raised when bad coordinate system inputted
    """
    if coord_system == "KEP":
        orbits = np.array(
            np.array([
                np.linspace(0, len(df) - 1, len(df)),
                df.a,
                df.e,
                np.deg2rad(df.i),
                np.deg2rad(df.Omega),
                np.deg2rad(df.arg_peri),
                np.deg2rad(df.mean_anom),
                np.repeat(3, len(df)).astype(int),  # keplerian input
                df.epoch,
                np.repeat(3, len(df)).astype(int),  # TT timescale
                df.H,
                df.G
            ]).transpose(),
        dtype=np.double, order='F')
    elif coord_system == "COM":
        orbits = np.array(
            np.array([
                np.linspace(0, len(df) - 1, len(df)),
                df.q,
                df.e,
                np.deg2rad(df.i),
                np.deg2rad(df.Omega),
                np.deg2rad(df.argperi),
                df.t_p,
                np.repeat(2, len(df)).astype(int),  # cometary input
                df.t_0,
                np.repeat(3, len(df)).astype(int),  # TT timescale
                df.H,
                np.repeat(0.15, len(df))
            ]).transpose(),
        dtype=np.double, order='F')
    elif coord_system == "CART":
        orbits = np.array(
            np.array([
                np.linspace(0, len(df) - 1, len(df)),
                df.x,
                df.y,
                df.z,
                df.vx,
                df.vy,
                df.vz,
                np.repeat(1, len(df)).astype(int),  # cartesian input
                df.t,
                np.repeat(3, len(df)).astype(int),  # TT timescale
                df.H,
                df.g
            ]).transpose(),
        dtype=np.double, order='F')
    else:
        raise ValueError("Invalid coordinate system")
    return orbits


def transform_catalogue(df, coord_system, final_coord_system):
    assert oo.pyoorb.oorb_init() == 0

    orbits = df_to_orbits(df=df, coord_system=coord_system)

    final_etype = 1 if final_coord_system == "CART" else 2 if final_coord_system == "COM" else 3

    transformed_orbits, error = oo.pyoorb.oorb_element_transformation(in_orbits=orbits,
                                                                      in_element_type=final_etype)
    assert error == 0

    columns = None
    if final_coord_system == "CART":
        columns = ["id", "x", "y", "z", "vx", "vy", "vz", "coords", "epoch", "time_type", "H", "g"]
    elif final_coord_system == "COM":
        columns = ["id", "q", "e", "i", "Omega", "argperi", "t_p", "coords", "epoch", "time_type", "H", "g"]
    elif final_coord_system == "KEP":
        columns = ["id", "a", "e", "i", "Omega", "argperi", "mean_anom", "coords", "epoch", "time_type",
                   "H", "g"]
    else:
        raise ValueError("Invalid coordinate system")

    # convert orbits back to a df
    transformed_df = pd.DataFrame(transformed_orbits, columns=columns)

    # drop the useless columns
    transformed_df.drop(labels=["coords", "time_type"], axis=1)

    # add the designations back in
    transformed_df["des"] = df.index.values

    return transformed_df


def propagate_catalogues(orbits, until_when, dynmodel="2"):
    assert oo.pyoorb.oorb_init() == 0

    