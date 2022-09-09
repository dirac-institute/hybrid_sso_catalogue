import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, get_sun
from astropy.visualization import quantity_support
quantity_support()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
pd.set_option("display.max_columns", None)

from magnitudes import absolute_magnitude

import thor
from thor.constants import Constants
from thor.backend import PYOORB
backend = PYOORB()

import urllib.request
import json

plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.6 * fs,
          'ytick.labelsize': 0.6 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)


def variant_orbit_ephemerides(ra, dec, ra_end, dec_end, delta_t, obstime, distances, radial_velocities,
                              sigma_ra=0.1 * u.arcsecond, sigma_dec=0.1 * u.arcsecond, apparent_mag=None,
                              eph_times=None, coords="heliocentriceclipticiau76", location="Gemini South",
                              obs_code="I11", pm_ra_cosdec=None, pm_dec=None, only_neos=False, verbose=False,
                              num_jobs="auto", chunk_size=100):
    """Generate ephemerides for a series of variant orbits for an observed object without constraints on its
    distance and radial velocity.

    Parameters
    ----------
    ra : `float`
        Right ascension at initial observation (with Astropy units)
    dec : `float`
        Declination at initial observation (with Astropy units)
    obstime : `Astropy Time object`
        Time of initial observation
    ra_end : `float`
        Right ascension at final observation (with Astropy units)
    dec_end : `float`
        Declination at final observation (with Astropy units)
    delta_t : `Astropy Time object`
        Time between observations
    pm_ra_cosdec : `float`, optional
        Directly calculated change in dot(ra) * cos(dec) (replaces ra_end, dec_end, delta_t), by default None
    pm_dec : `float`, optional
        Directly calculated change in dot(dec) (replaces ra_end, dec_end, delta_t), by default None
    distances : `float/array`
        Array of possible distances to use
    radial_velocities : `float/array`
        Array of possible radial velocities to use
    sigma_ra : `float`, optional
        Observation error in RA, by default 0.1*u.arcsecond
    sigma_dec : `float`, optional
        Observation error in Dec, by default 0.1*u.arcsecond
    apparent_mag : `float`, optional
        Apparent magnitude of object in V band
    eph_times : `Astropy Time object/array`, optional
        Array of times at which to produce ephemerides, by default 1 day later
    coords : `str`, optional
        Coordinate system to use, by default "heliocentriceclipticiau76"
    location : `str`, optional
        Location of observations (str from astropy Earth locations), by default "Gemini South"a
    obs_code : `str`, optional
        As `location` for the 3 character code, by default "I11"
    only_neos : `bool`, optional
        Whether to restrict orbits to only those that match an NEO (q < 1.3), by default False
    verbose: `bool`, optional
        Whether to print some debugging messages, by default False

    Returns
    -------
    df : `pandas DataFrame`
        Dataframe containing ephemerides for each distance-radial velocity combination

    """

    # create a grid from the distances and radial velocities
    D, RV = np.meshgrid(distances, radial_velocities)
    size = len(distances) * len(radial_velocities)

    # use astropy to get information based on a site name
    obs_loc = EarthLocation.of_site(location)

    # need a list with units rather than list of things each with units
    obsgeoloc = [x.to(u.m).value for x in obs_loc.geocentric] * u.m

    # get the observer position in cartesian GCRS coordinates for THOR
    observer_position = SkyCoord(x=obsgeoloc[0],
                                 y=obsgeoloc[1],
                                 z=obsgeoloc[2],
                                 obstime=obstime,
                                 frame="gcrs",
                                 representation_type="cartesian").transform_to(coords).cartesian.xyz
    observer_positions = np.tile(observer_position.to(u.AU).value, (size, 1))

    # if proper motions are not provided
    if pm_ra_cosdec is None and pm_dec is None:
        # add some dispersion to the ra/dec's with the given sigmas (or just repeat if not are given)
        if sigma_ra.value == 0.0:
            ra = np.repeat(ra.value, repeats=size) * ra.unit
            ra_end = np.repeat(ra_end.value, repeats=size) * ra_end.unit
        else:
            ra = np.random.normal(ra.value, scale=sigma_ra.to(ra.unit).value, size=size) * ra.unit
            ra_end = np.random.normal(ra_end.value, scale=sigma_ra.to(ra_end.unit).value, size=size) * ra_end.unit

        if sigma_dec.value == 0.0:
            dec = np.repeat(dec.value, repeats=size) * dec.unit
            dec_end = np.repeat(dec_end.value, repeats=size) * dec_end.unit
        else:
            dec = np.random.normal(dec.value, scale=sigma_dec.to(dec.unit).value, size=size) * dec.unit
            dec_end = np.random.normal(dec_end.value, scale=sigma_dec.to(dec_end.unit).value, size=size) * dec_end.unit

        # convert them to Skycoords
        start = SkyCoord(ra=ra, dec=dec, frame="icrs")
        end = SkyCoord(ra=ra_end, dec=dec_end, frame="icrs")

        # calculate the offset in ra/dec and convert to a proper motion
        delta_ra_cosdec, delta_dec = start.spherical_offsets_to(end)
        pm_ra_cosdec, pm_dec = delta_ra_cosdec / delta_t, delta_dec / delta_t

        if verbose:
            print(ra, dec, ra_end, dec_end, delta_t, pm_ra_cosdec, pm_dec)

    # put it all together into a single astropy SkyCoord in GCRS (using loc/time from above)
    coord = SkyCoord(ra=ra,
                     dec=dec,
                     pm_ra_cosdec=pm_ra_cosdec,
                     pm_dec=pm_dec,
                     distance=D.ravel(),
                     radial_velocity=RV.ravel(),
                     frame="gcrs",
                     obsgeoloc=obsgeoloc,
                     obstime=obstime)

    # convert to ecliptic
    ecl = coord.transform_to(coords)

    # translate astropy into what THOR wants
    orbits = np.atleast_2d(np.concatenate((ecl.cartesian.xyz.to(u.AU).value,
                                           ecl.velocity.d_xyz.to(u.AU / u.day).value))).T
    t0 = np.repeat(ecl.obstime.mjd, size)

    # use THOR to account for light travel time
    corrected_orbits, lt = thor.addLightTime(orbits=orbits, t0=t0, observer_positions=observer_positions,
                                             lt_tol=1e-10, mu=Constants.MU, max_iter=1000, tol=1e-15)
    lt = np.nan_to_num(lt, nan=0.0)
    corrected_t0 = t0 - lt

    if apparent_mag is None:
        H = None
    else:
        d_ast_sun = ecl.distance.to(u.AU).value
        d_ast_earth = D.ravel().to(u.AU).value
        d_earth_sun = get_sun(time=Time(corrected_t0, format="mjd")).distance.to(u.AU).value
        H = absolute_magnitude(m=apparent_mag,
                               d_ast_sun=d_ast_sun, d_ast_earth=d_ast_earth, d_earth_sun=d_earth_sun)

    # create a THOR orbits class
    orbits_class = thor.Orbits(orbits=corrected_orbits, epochs=Time(corrected_t0, format="mjd"), H=H)

    # if you only want NEO orbits then mask anything with a perihelion above 1.3 AU
    if only_neos:
        perihelion = orbits_class.keplerian[:, 0] * (1 - orbits_class.keplerian[:, 1])
        orbits_class = thor.Orbits(orbits=corrected_orbits[perihelion < 1.3],
                                   epochs=Time(corrected_t0[perihelion < 1.3], format="mjd"),
                                   H=H[perihelion < 1.3])

    # default to one day after the observation
    if eph_times is None:
        eph_times = np.atleast_1d(obstime + 1)

    # use pyoorb (through THOR) to get the emphemeris at the supplied times
    df = backend.generateEphemeris(orbits=orbits_class, observers = {obs_code: eph_times},
                                   num_jobs=num_jobs, chunk_size=chunk_size)

    return df


def create_scout_comparison_plot(day, time, des="P21vBEn", distances=None, radial_velocities=None,
                                 obs_code="F52", location="Haleakala Observatories", **kwargs):

    print("Loading in observation data from Scout...")

    # load in the Scout observation data about this object in MPC format
    with urllib.request.urlopen(f"https://ssd-api.jpl.nasa.gov/scout.api?tdes={des}&file=mpc") as url:
        file = json.loads(url.read().decode())["fileMPC"]

        scout_times = []
        scout_coords = []

        for line in file.split("\n"):
            if line.rstrip() != "":
                date = line[15:32].split()
                obstime = Time(date[0] + "-" + date[1] + "-" + date[2].split(".")[0])\
                    + float("0." + date[2].split(".")[1])
                scout_times.append(obstime)

                scout_coords.append(SkyCoord(ra=line[32:44], dec=line[44:56],
                                             unit=(u.hourangle, u.deg), frame="icrs"))

        # NOTE: this currently just naively takes the first and last observation
        scout_delta_t = (scout_times[-1] - scout_times[0]).to(u.day)

    print("Computing and loading ephemerides from Scout...")

    # get the ephemeris information from Scout
    url_str = f"https://ssd-api.jpl.nasa.gov/scout.api?tdes={des}&eph-start={day}T{time}&obs-code={obs_code}"
    with urllib.request.urlopen(url_str) as url:
        data = json.loads(url.read().decode())

        # nominal comparison orbit (I think this is the median in Scout)
        nominal = SkyCoord(ra=float(data["eph"][0]["median"]["ra"]),
                           dec=float(data["eph"][0]["median"]["dec"]), unit="deg", frame="icrs")

        # ephemeris (taking only the ra/dec/magnitude)
        scout_eph = np.array(data["eph"][0]["data"])[:, [0, 1, 4]].astype(float)

    print("Computing orbits using Tom's variant orbit code...")
    if distances is None:
        distances = np.logspace(np.log10(6e-2), np.log10(2), 500) * u.AU
    if radial_velocities is None:
        radial_velocities = np.linspace(-30, 30, 20) * u.km / u.s
    D, RV = np.meshgrid(distances, radial_velocities)

    df = variant_orbit_ephemerides(ra=scout_coords[0].ra,
                                   dec=scout_coords[0].dec,
                                   ra_end=scout_coords[-1].ra,
                                   dec_end=scout_coords[-1].dec,
                                   delta_t=scout_delta_t,
                                   obstime=scout_times[0],
                                   distances=distances,
                                   radial_velocities=radial_velocities,
                                   eph_times=np.atleast_1d(Time(f"{day} {time}")),
                                   obs_code=obs_code,
                                   location=location,
                                   **kwargs)

    print("Done. Plotting!")

    # create a figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # scatter the variant orbits created with this code
    scatter = ax.scatter((df["RA_deg"] - nominal.ra.to(u.deg).value) * np.cos(nominal.dec.to(u.deg)) * (u.deg).to(u.arcminute),
                         (df["Dec_deg"] - nominal.dec.value) * (u.deg).to(u.arcminute),
                         s=5, c=D.ravel(), norm=LogNorm(), cmap="Reds_r")

    first_xlims = ax.get_xlim()
    first_ylims = ax.get_ylim()

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Assumed Distance [AU]")

    # scatter the orbits using Scout
    scout = ax.scatter((scout_eph.T[0] - nominal.ra.value) * np.cos(nominal.dec.to(u.deg)) * (u.deg).to(u.arcminute),
                       (scout_eph.T[1] - nominal.dec.value) * (u.deg).to(u.arcminute),
                       s=10, c=scout_eph.T[2], cmap="rainbow", label="Scout Data")

    second_cbar = fig.colorbar(scout, ax=ax)
    second_cbar.set_label("V-band Magnitude")

    ax.set_xlabel("Relative RA cos(dec) [arcminutes]")
    ax.set_ylabel("Relative Declination [arcminutes]")

    ax.set_xlim(min(first_xlims[0], ax.get_xlim()[0]), max(first_xlims[1], ax.get_xlim()[1]))
    ax.set_ylim(min(first_ylims[0], ax.get_ylim()[0]), max(first_ylims[1], ax.get_ylim()[1]))

    ax.set_xlim(reversed(ax.get_xlim()))

    ax.annotate(f"RV range used: [{RV.min()}, {RV.max()}]", xy=(0.95, 0.05),
                xycoords="axes fraction", ha="right", fontsize=0.8*fs)

    ax.set_title(des, fontsize=fs)

    ax.legend(loc="upper left", handletextpad=0, markerscale=5)

    ax.grid()

    plt.show()

    return df, scout_eph, nominal
