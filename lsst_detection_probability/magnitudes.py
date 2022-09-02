import numpy as np


def absolute_magnitude(m, d_ast_sun, d_ast_earth, d_earth_sun=None, phase_angle=None, G=0.15):
    """Get the absolute magnitude of an asteroid (following MPC equation
    https://minorplanetcenter.net/iau/ECS/MPCArchive/1985/MPC_19851227.pdf)

    Parameters
    ----------
    m : `float/array`
        Apparent magnitude
    d_ast_sun : `float/array`
        Distance from the asteroid to the Sun in AU
    d_ast_earth : `float/array`
        Distance from the asteroid to the Earth in AU
    d_earth_sun : `float/array`
        Distance from the Earth to the Sun in AU, optional. Must be supplied if `phase_angle` is None.
    phase_angle : `float/array`
        Phase angle in radians, optional. Must be supplied if `d_earth_sun` is None.
    G : `float/array`
        Slope parameter, optional. By default, 0.15.

    Returns
    -------
    H : `float/array`
        Absolute magnitude
    """
    if phase_angle is None and d_earth_sun is None:
        raise ValueError("Either `phase_angle` or `d_earth_sun` must be provided")
    elif phase_angle is None:
        phase_angle = np.arccos((d_ast_sun**2 + d_ast_earth**2 - d_earth_sun**2)
                                / (2 * d_ast_sun * d_ast_earth))
    return m - 5 * np.log10(d_ast_sun * d_ast_earth)\
        + 2.5 * np.log10((1 - G) * phi(1, phase_angle) + G * phi(2, phase_angle))


def apparent_magnitude(H, d_ast_sun, d_ast_earth, d_earth_sun=None, phase_angle=None, G=0.15):
    """Get the apparent magnitude of an asteroid (following MPC equation
    https://minorplanetcenter.net/iau/ECS/MPCArchive/1985/MPC_19851227.pdf)

    Parameters
    ----------
    H : `float/array`
        Absolute magnitude
    d_ast_sun : `float/array`
        Distance from the asteroid to the Sun in AU
    d_ast_earth : `float/array`
        Distance from the asteroid to the Earth in AU
    d_earth_sun : `float/array`
        Distance from the Earth to the Sun in AU, optional. Must be supplied if `phase_angle` is None.
    phase_angle : `float/array`
        Phase angle in radians, optional. Must be supplied if `d_earth_sun` is None.
    G : `float/array`
        Slope parameter, optional. By default, 0.15.

    Returns
    -------
    m : `float/array`
        Apparent magnitude
    """
    if phase_angle is None and d_earth_sun is None:
        raise ValueError("Either `phase_angle` or `d_earth_sun` must be provided")
    elif phase_angle is None:
        phase_angle = np.arccos((d_ast_sun**2 + d_ast_earth**2 - d_earth_sun**2)
                                / (2 * d_ast_sun * d_ast_earth))
    return H + 5 * np.log10(d_ast_sun * d_ast_earth)\
        - 2.5 * np.log10((1 - G) * phi(1, phase_angle) + G * phi(2, phase_angle))


def phi(ind, phase_angle):
    """Phase functions for absolute magnitude

    Parameters
    ----------
    ind : `int`
        Either 1 or 2, which phase function to use
    phase_angle : `float/array`
        Phase angle in radians
    """
    coeffs = [
        [-3.33, -1.87],
        [0.63, 1.22]
    ]
    return np.exp(coeffs[0][ind - 1] * np.tan(phase_angle / 2)**(coeffs[1][ind - 1]))


def convert_colour_mags(mag, out_colour, in_colour="V", convention="LSST", asteroid_type="C"):
    """Convert between different colours in magnitudes.

    MPC convention is here: https://minorplanetcenter.net/iau/info/BandConversion.txt
    LSST convention is in Table 1 of this: http://faculty.washington.edu/ivezic/Publications/Jones2018.pdf

    Parameters
    ----------
    mag : `float/array`
        Input magnitude
    out_colour : `str`
        Desired output colour, one of (V, u, g, r, i, z, y)
    in_colour : `str`
        Colour of the input magnitude, one of (V, u, g, r, i, z, y), by default "V"
    convention : `str`, optional
        Which convention of colours to use, either MPC or LSST, by default "LSST"
    asteroid_type : `str`, optional
        What type of asteroid to use, only used for LSST convention, either C or S, by default "C"

    Returns
    -------
    out_mag : `float/array`
        Magnitude in output colour
    """
    if convention == "MPC":
        colours = {
            "V": 0,
            "u": +2.5,
            "g": -0.35,
            "r": +0.14,
            "i": +0.32,
            "z": +0.26,
            "y": +0.32
        }
    elif convention == "LSST":
        if asteroid_type == "C":
            colours = {
                "V": 0,
                "u": +1.53,
                "g": +0.28,
                "r": -0.18,
                "i": -0.29,
                "z": -0.30,
                "y": -0.30
            }
        elif asteroid_type == "S":
            colours = {
                "V": 0,
                "u": +1.82,
                "g": +0.37,
                "r": -0.26,
                "i": -0.46,
                "z": -0.40,
                "y": -0.41
            }
        else:
            raise ValueError(f"Invalid asteroid type: {asteroid_type}")
    else:
        raise ValueError(f"Invalid convention: {convention}")

    return mag + (colours[out_colour] - colours[in_colour])
