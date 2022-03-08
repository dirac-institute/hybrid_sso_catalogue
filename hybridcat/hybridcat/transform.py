import numpy as np
import pandas as pd
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
                np.deg2rad(df.argperi),
                np.deg2rad(df.mean_anom),
                np.repeat(3, len(df)).astype(int),  # keplerian input
                df.t_0,
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
                df.t_0,
                np.repeat(3, len(df)).astype(int),  # TT timescale
                df.H,
                df.g
            ]).transpose(),
        dtype=np.double, order='F')
    else:
        raise ValueError("Invalid coordinate system")
    return orbits


def columns_from_coords(coords):
    """Get df column names from coordinates

    Parameters
    ----------
    coords : `str`
        One of CART, COM and KEP

    Returns
    -------
    columns : `list`
        The column names

    Raises
    ------
    ValueError
        Bad coordinate system inputted
    """
    columns = None
    if coords == "CART":
        columns = ["id", "x", "y", "z", "vx", "vy", "vz", "coords", "t_0", "time_type", "H", "g"]
    elif coords == "COM":
        columns = ["id", "q", "e", "i", "Omega", "argperi", "t_p", "coords", "t_0", "time_type", "H", "g"]
    elif coords == "KEP":
        columns = ["id", "a", "e", "i", "Omega", "argperi", "mean_anom", "coords", "t_0", "time_type",
                   "H", "g"]
    else:
        raise ValueError("Bad coordinate system")
    return columns


def transform_catalogue(df, current_coords, transformed_coords, initialise=True):
    """Transform a catalogue from coordinate system to another

    Parameters
    ----------
    df : `Pandas DataFrame`
        DataFrame contained the catalogue
    current_coords : `str`
        Current coordinate system (one of CART, COM, KEP)
    transformed_coords : `str`
        Coordinate system that you want to transform to (one of CART, COM, KEP)

    Returns
    -------
    transformed_df : `Pandas DataFrame`
        The transformed catalogue

    Raises
    ------
    ValueError
        Bad coordinate system entered
    """
    # initialise openorb and make sure it isn't broken
    if initialise:
        assert oo.pyoorb.oorb_init() == 0

    # convert the dataframe to orbit array
    orbits = df_to_orbits(df=df, coord_system=current_coords)

    # ensure coordinate systems are okay
    if transformed_coords not in ["CART", "COM", "KEP"]:
        raise ValueError("Bad coordinate system")

    # perform the transformation
    final_etype = 1 if transformed_coords == "CART" else 2 if transformed_coords == "COM" else 3
    transformed_orbits, error = oo.pyoorb.oorb_element_transformation(in_orbits=orbits,
                                                                      in_element_type=final_etype)
    assert error == 0

    # convert orbits back to a df
    transformed_df = pd.DataFrame(transformed_orbits, columns=columns_from_coords(transformed_coords))

    # drop the useless columns
    transformed_df.drop(labels=["coords", "time_type"], axis=1)

    # add the designations back in
    transformed_df.set_index(df.index.values, inplace=True)

    return transformed_df


def propagate_catalogues(df, until_when, coords="CART", dynmodel="2", initialise=True):
    """Propagate a catalogue until a certain time

    Parameters
    ----------
    df : `Pandas DataFrame`
        DataFrame contained the catalogue
    until_when : `float`
        Modified Julian date for when to evolve until
    coords : `str`, optional
        Coordinate system (one of CART, COM, KEP), by default "CART"
    dynmodel : `str`, optional
        Which model to use (basically 2=fast, N=accurate), by default "2"
    initialise : `boolean`, optional
        Whether to initialise openorb, by default True

    Returns
    -------
    propagated_df : `Pandas DataFrame`
        The propagated catalogue

    Raises
    ------
    ValueError
        Bad dynmodel entered
    """
    # initialise openorb and make sure it isn't broken
    if initialise:
        assert oo.pyoorb.oorb_init() == 0

    # convert the dataframe to orbit array
    orbits = df_to_orbits(df=df, coord_system=coords)

    # ensure coordinate systems are okay
    if dynmodel not in ["2", "N"]:
        raise ValueError("Bad dynmodel system")

    # perform the transformation
    propagated_orbits, error = oo.pyoorb.oorb_propagation(in_orbits=orbits,
                                                          in_epoch=np.array([until_when, 3],
                                                                            dtype=np.double, order='F'),
                                                          in_dynmodel=dynmodel)
    assert error == 0

    # convert orbits back to a df
    propagated_df = pd.DataFrame(propagated_orbits, columns=columns_from_coords(coords))

    # drop the useless columns
    propagated_df.drop(labels=["coords", "time_type"], axis=1)

    # add the designations back in
    propagated_df.set_index(df.index.values, inplace=True)

    return propagated_df

    