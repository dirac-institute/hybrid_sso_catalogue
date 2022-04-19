import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile

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


def plot_neo_scores(neo_scores, density=False, bins=np.linspace(0, 100, 30), night=0):
    fig, ax = plt.subplots(figsize=(12, 8))

    handles = []

    for score in neo_scores:
        ax.hist(score["scores"], bins=bins, facecolor=score["colour"][:-1] + [0.2],
                label=score["label"], lw=3, density=density)
        ax.hist(score["scores"], bins=bins, color=score["colour"], histtype="step",
                lw=3, density=density)

        handles.append(Patch(facecolor=score["colour"][:-1] + [0.2],
                             edgecolor=score["colour"],
                             label=score["label"], lw=3))

    ax.legend(handles=handles, loc="upper center", ncol=2, fontsize=0.8 * fs)

    ax.set_xlabel("NEO Score")

    if density:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Number of objects")

    ax.set_title(f"Night {night}", fontsize=fs)

    return fig, ax


def get_specific_neo_score(path, file_name):
    if file_name.endswith(".filtered.dat"):
        if isfile(file_name):
            with open(path + file_name, "r") as f:
                ignore_me = f.readline().rstrip() == ""
            if not ignore_me:
                df = pd.read_fwf(path + file_name)
                return df["NEO"].values, df["Desig."].values

    return None, None


def get_neo_scores(path, night=None):
    if night is None:
        neo_scores = np.array([])
        ids = np.array([])
        files = listdir(path)

        for file_name in files:
            neo, ID = get_specific_neo_score(path, file_name)
            if neo is not None:
                neo_scores = np.concatenate((neo_scores, neo))
                ids = np.concatenate((ids, ID))
    else:
        neo_scores, ids = get_specific_neo_score(path, f"night_{night:03d}.filtered.dat")
    return neo_scores, ids
