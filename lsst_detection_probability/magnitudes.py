import numpy as np


def absolute_magnitude(m, G, phase_angle, d_ast_sun, d_ast_earth):
    """Get the absolute magnitude of an asteroid (following MPC equation
    https://minorplanetcenter.net/iau/ECS/MPCArchive/1985/MPC_19851227.pdf)

    Parameters
    ----------
    m : `float/array`
        Apparent magnitude
    G : `float/array`
        Slope parameter
    phase_angle : `float/array`
        Phase angle in radians
    d_ast_sun : `float/array`
        Distance from the asteroid to the Sun in AU
    d_ast_earth : `float/array`
        Distance from the asteroid to the Earth in AU

    Returns
    -------
    H : `float/array`
        Absolute magnitude
    """
    return m - 5 * np.log10(d_ast_sun * d_ast_earth)\
        + 2.5 * np.log10((1 - G) * phi(1, phase_angle) + G * phi(2, phase_angle))


def apparent_magnitude(H, G, phase_angle, d_ast_sun, d_ast_earth):
    """Get the apparent magnitude of an asteroid (following MPC equation
    https://minorplanetcenter.net/iau/ECS/MPCArchive/1985/MPC_19851227.pdf)

    Parameters
    ----------
    H : `float/array`
        Absolute magnitude
    G : `float/array`
        Slope parameter
    phase_angle : `float/array`
        Phase angle in radians
    d_ast_sun : `float/array`
        Distance from the asteroid to the Sun in AU
    d_ast_earth : `float/array`
        Distance from the asteroid to the Earth in AU

    Returns
    -------
    m : `float/array`
        Apparent magnitude
    """
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
