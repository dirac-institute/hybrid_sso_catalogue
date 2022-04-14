import pandas as pd
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
import astropy.units as u
import numpy as np
from os import listdir
import subprocess
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time


""" --- Analysis functions --- """

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
    fig, ax = plt.subplots(figsize=(12,8))

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
        neo_scores, ids = get_specific_neo_score(path, f"night_{night:02d}.filtered.dat")
    return neo_scores, ids


def print_time_delta(start, end, label):
    delta = end - start
    print(f"  {int(delta // 60):02d}m{int(delta % 60):02d}s - ({label})")


""" --- digest2 prep functions --- """

def filter_tracklets(df, min_obs=2, min_arc=1, max_time=90):
    
    init = SkyCoord(ra=df["AstRA(deg)"].iloc[0], dec=df["AstDec(deg)"].iloc[0], unit="deg")
    final = SkyCoord(ra=df["AstRA(deg)"].iloc[-1], dec=df["AstDec(deg)"].iloc[-1], unit="deg")

    return np.logical_and.reduce((len(df) >= min_obs,
                                  init.separation(final).to(u.arcsecond).value > min_arc,
                                  df["FieldMJD"].diff().min() * 1440 < max_time))


def create_digest2_input(in_path="/data/epyc/projects/jpl_survey_sim/10yrs/detections/march_start_v2.1/S0/",
                         out_path="neo/", night_zero=59638, start_night=0, final_night=31, timeit=False,
                         min_obs=2, min_arc=1, max_time=90):
    
    if timeit:
        start = time.time()
    
    # convert all S3M IDs to hex so they fit
    s3m = pd.read_hdf("../catalogues/s3m_initial.h5")
    hex_ids = np.array([f'{num:07X}' for num in np.arange(len(s3m.index.values))])
    s3m_to_hex7 = dict(zip(s3m.index.values, hex_ids))
    print("done creating S3M ID mapping")
    
    if timeit:
        print_time_delta(start, time.time(), label="S3M Mapping Creation")
        start = time.time()
    
    night, file = start_night, 0
    files = sorted(listdir(in_path[0])) if isinstance(in_path, list) else sorted(listdir(in_path))

    # flags for whether to move on to the next file and whether to append or not
    next_file = True
    append = False

    # loop until all of the nights have been read in
    while night < final_night:
        # if reading the next file
        if next_file:
            # convert hdf to pandas
            if isinstance(in_path, str):
                df = pd.read_hdf(in_path + files[file])
            else:
                dfs = [pd.read_hdf(in_path[i] + files[file]) for i in range(len(in_path))]
                df = pd.concat(dfs)

            # create night column relative to night_zero
            df["night"] = (df["FieldMJD"] - 0.5).astype(int)
            df["night"] -= night_zero
            next_file = False
            
            if timeit:
                print_time_delta(start, time.time(), label=f"Reading file {file}")
                start = time.time()

        # get only the rows on this night
        nightly_obs = df[df["night"] == night]

        # create a mask based on min # of obs, min arc length, max time between shortest pair
        mask = nightly_obs.groupby("ObjID").apply(filter_tracklets, min_obs, min_arc, max_time)
        nightly_obs[nightly_obs["ObjID"].isin(mask[mask].index)]

        # sort by the object and then the time
        nightly_obs = nightly_obs.sort_values(["ObjID", "FieldMJD"])

        # convert RA and Dec to hourangles and MJD to regular dates
        ra_degrees = Angle(nightly_obs["AstRA(deg)"], unit="deg").hms
        dec_degrees = Angle(nightly_obs["AstDec(deg)"], unit="deg").hms
        datetimes = Time(nightly_obs["FieldMJD"], format="mjd").datetime
        
        if timeit:
            print_time_delta(start, time.time(), label=f"Filter data for night {night}")
            start = time.time()

        # match to 80 column format: https://www.minorplanetcenter.net/iau/info/OpticalObs.html
        # each line stars with 5 spaces
        lines = [" " * 5 for i in range(len(nightly_obs))]
        for i in range(len(nightly_obs)):
            # squish the ID into the available space (TODO: this may be problematic)
            lines[i] += s3m_to_hex7[nightly_obs.iloc[i]["ObjID"]]
            
            # add two spaces and a C (the C is important for some reason)
            lines[i] += " " * 2 + "C"

            # convert time to HH MM DD.ddddd format
            t = datetimes[i]
            lines[i] += "{:4.0f} {:02.0f} {:08.5f} ".format(t.year, t.month, t.day + nightly_obs.iloc[i]["FieldMJD"] % 1.0)
            
            # convert RA to HH MM SS.ddd
            lines[i] += "{:02.0f} {:02.0f} {:06.3f}".format(ra_degrees.h[i], ra_degrees.m[i], ra_degrees.s[i])
            
            # convert Dec to sHH MM SS.dd
            lines[i] += "{:+03.0f} {:02.0f} {:05.2f}".format(dec_degrees.h[i], abs(dec_degrees.m[i]), abs(dec_degrees.s[i]))

            # leave some blank columns
            lines[i] += " " * 9

            # add the magntiude and filter (right aligned)
            lines[i] += "{:04.1f}  {}".format(nightly_obs.iloc[i]["MaginFilter"], nightly_obs.iloc[i]["filter"])

            # add some more spaces and an observatory code
            lines[i] += " " * 5 + "I11" + "\n"

        # write that to a file
        with open(out_path + "night_{:02d}.obs".format(night), "a" if append else "w") as obs_file:
            obs_file.writelines(lines)
        append = False
        
        if timeit:
            print_time_delta(start, time.time(), label=f"Writing observations for night {night}")
            start = time.time()

        # if we've exhausted the file then move on
        if night >= df["night"].max():
            file += 1
            next_file = True
            
            # append obs next time
            append = True
        # if the file is still going then move to the next night
        else:
            print(str(len(nightly_obs)) + " observations in night " + str(night))
            night += 1
            
            
def create_bash_script(out_path="neo/", start_night=0, final_night=31, digest2_folder="/data/epyc/projects/hybrid-sso-catalogs/digest2/", cpu_count=32):
    bash = "for NIGHT in " + " ".join(["{:02d}".format(i) for i in range(start_night, final_night)]) + "\n"
    bash += "do\n"

    bash += 'echo "Now running night $NIGHT through digest2..."\n'
    bash += "\t" + f"time {digest2_folder}digest2 -p {digest2_folder} -c {digest2_folder}MPC.config --cpu {cpu_count}"
    bash += " {}night_$NIGHT.obs > {}night_$NIGHT.dat".format(out_path, out_path) + "\n"
    bash += "grep -a -v tracklet {}night_$NIGHT.dat > {}night_$NIGHT.filtered.dat \n".format(out_path, out_path)

    bash += "done\n"
    
    return bash

        
def main():
    
    parser = argparse.ArgumentParser(description='Run digest2 on LSST mock observations')
    parser.add_argument('-i', '--in-path',
                        default="/data/epyc/projects/jpl_survey_sim/10yrs/detections/march_start_v2.1/S0/",
                        type=str, help='Path to the folder containing mock observations')
    parser.add_argument('-o', '--out-path', default="neo/", type=str,
                        help='Path to folder in which to place output')
    parser.add_argument('-d', '--digest2-path', default="/data/epyc/projects/hybrid-sso-catalogs/digest2/",
                        type=str, help='Path to digest2 folder')
    parser.add_argument('-z', '--night-zero', default=59638, type=int,
                        help='MJD value for the first night')
    parser.add_argument('-s', '--start-night', default=0, type=int,
                        help='First night to run through digest2')
    parser.add_argument('-f', '--final-night', default=31, type=int,
                        help='Last night to run through digest2')
    parser.add_argument('-mo', '--min-obs', default=2, type=int,
                        help='Minimum number of observations per night')
    parser.add_argument('-ma', '--min-arc', default=1, type=int,
                        help='Minimum arc length in arcseconds')
    parser.add_argument('-mt', '--max-time', default=90, type=int,
                        help='Maximum time between shortest pair of tracklets in night')
    parser.add_argument('-c', '--cpu-count', default=32, type=int,
                        help='How many CPUs to use for the digest2 calculations')
    parser.add_argument('-M', '--mba', action="store_true",
                        help="Replace in_path and out_path with defaults for MBAs")
    parser.add_argument('-t', '--timeit', action="store_true",
                        help="Whether to time the code and print it out")
    args = parser.parse_args()
    
    if args.mba:
        args.in_path = ["/data/epyc/projects/jpl_survey_sim/10yrs/detections/march_start_v2.1/S1_{:02d}/".format(i) for i in range(14)]
        args.out_path = "mba/"
        
    print("Starting digest2 run for", args.out_path)

    create_digest2_input(in_path=args.in_path, out_path=args.out_path, timeit=args.timeit,
                         night_zero=args.night_zero, start_night=args.start_night, final_night=args.final_night,
                         min_obs=args.min_obs, min_arc=args.min_arc, max_time=args.max_time)
    
    script = create_bash_script(out_path=args.out_path, start_night=args.start_night, final_night=args.final_night,
                                digest2_path=args.digest2_path, cpu_count=args.cpu_count)
    
    start = time.time()
    subprocess.call(script, shell=True)
    print_time_delta(start, time.time(), "total digest2 run")
    print("Hurrah!")


if __name__ == "__main__":
    main()