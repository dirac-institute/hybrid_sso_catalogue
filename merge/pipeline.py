import transform_orbits
import pandas as pd


def main():
    print("Started")
    mpcorb = pd.read_hdf("../catalogues/mpcorb.h5", key="df")
    print("Catalogues read in")
    mpcorb_com = transform_orbits.transform_catalogue(mpcorb, coord_system="KEP", final_coord_system="COM")
    print(mpcorb_com.q)


if __name__ == "__main__":
    main()