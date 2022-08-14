import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

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


def get_LSST_schedule(night, fields=["fieldRA", "fieldDec", "observationStartMJD",
                                     "flush_by_mjd", "visitExposureTime", "numExposures",
                                     "visitTime"]):
    """Get the schedule for LSST (where it will point at what time)

    Parameters
    ----------
    night : `int`
        Which night you want the schedule for
    fields : `list`, optional
        Fields you want from the database, by default ["fieldRA", "fieldDec", "observationStartMJD",
        "flush_by_mjd", "visitExposureTime", "numExposures", "visitTime"]

    Returns
    -------
    df : `pandas DataFrame`
        DataFrame containing individual fields
    """
    con = sqlite3.connect('/epyc/projects/jpl_survey_sim/10yrs/opsims/march_start_v2.1_10yrs.db')
    cur = con.cursor()

    res = cur.execute(f"select {','.join(fields)} from observations where night={night}")
    con.close()

    df = pd.DataFrame(res.fetchall(), columns=fields)
    return df


def plot_LSST_schedule(df):
    """Plot LSST schedule up using the dataframe containing fields. Each is assumed to be a circle of radius
    2.1 degrees for simplicity.

    Parameters
    ----------
    df : `pandas DataFrame`
        DataFrame of fields (see `get_LSST_schedule`)
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for _, row in df.iterrows():
        ax.add_patch(plt.Circle(xy=(row["fieldRA"], row["fieldDec"]),
                                radius=2.1, fc="none", ec="grey"))

    scatter = ax.scatter(df["fieldRA"], df["fieldDec"],
                         marker="H", cmap="plasma",
                         c=(df["observationStartMJD"] - df["observationStartMJD"].iloc[0])*24)

    fig.colorbar(scatter, label="Time during night [hours]")

    ax.set_xlabel("Right Ascension [deg]")
    ax.set_ylabel("Declination [deg]")

    plt.show()
