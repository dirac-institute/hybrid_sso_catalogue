import pandas as pd
from os.path import isfile


def create_mpcorb_from_json(in_path="../catalogues/mpcorb_extended.json",
                            out_path="../catalogues/mpcorb_initial.h5", force=False):
    # if the file doesn't exist yet or you forcibly want to recreate it
    if not isfile(out_path) or force:
        # convert json to pandas Dataframe
        mpcorb_df = pd.read_json(in_path)

        # only select pertinent columns
        mpcorb_df = mpcorb_df[["H", "G", "Principal_desig", "Epoch",
                            "M", "Peri", "Node", "i", "e", "a"]]

        # adjust to column names from S3m
        mpcorb_df.rename(columns={"Principal_desig": "des", "Epoch": "t_0",
                                "M": "mean_anom", "Peri": "argperi", "Node": "Omega"},
                        inplace=True)

        # adjust to modified JD and save file
        mpcorb_df.t_0 = mpcorb_df.t_0 - 2400000.5
        mpcorb_df.to_hdf(out_path, mode="w", key="df")
    else:
        # just read the file
        mpcorb_df = pd.read_hdf(out_path, key="df")
    return mpcorb_df


def create_s3m_from_files(in_path="../catalogues/s3m_files/", out_path="../catalogues/s3m_initial.h5"):
    # define the column and file names
    names_s3m = ['id', 'format', 'q', 'e', 'i', 'Omega', 'argperi', 't_p',
                'H', 't_0', 'INDEX', 'N_PAR', 'MOID', 'COMPCODE']

    files_s3m = ['S0', 'S1_00', 'S1_01', 'S1_02', 'S1_03', 'S1_04', 'S1_05',
                'S1_06', 'S1_07', 'S1_08', 'S1_09', 'S1_10', 'S1_11', 'S1_12',
                'S1_13', "SL", "St5", "ST"]

    # create a dataframe for each file
    dfs = [pd.read_csv(in_path + "{}.s3m".format(files_s3m[i]), comment="!", delim_whitespace=True,
                       header=None, names=names_s3m, skiprows=2)
           for i in range(len(files_s3m))]

    # stick them all together
    s3m_df = pd.concat(dfs)

    # make sure the indices are unique
    s3m_df.set_index("id", inplace=True)

    # drop anything with e = 1.0 because openorb can't do it
    s3m_df.drop(labels=s3m_df.index[s3m_df.e == 1.0], axis=0, inplace=True)

    # save to hdf5
    s3m_df.to_hdf(out_path, key="df", mode="w")
