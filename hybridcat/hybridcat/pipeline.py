import hybridcat.data as data
import hybridcat.transform_orbits as transform_orbits
# import hybridcat.merge_catalogues as merge
import pandas as pd
from astropy.time import Time
from time import time


def preprocessing():
    print("Started")
    start = time()

    # mpcorb = data.create_mpcorb_from_json()
    s3m = pd.read_hdf("../catalogues/initial/s3m.h5", key="df")
    print("Catalogue created/read: {:1.2f}s".format(time() - start))
    
    # mpcorb_com = transform_orbits.transform_catalogue(mpcorb, current_coords="KEP", transformed_coords="COM")
    # mpcorb_com.to_hdf("../catalogues/cometary/mpcorb.h5", key="df", mode="w")
    print("Coordinate conversion done: {:1.2f}s".format(time() - start))
    
    march_first = Time("2022-03-01").mjd
    # mpcorb_prop = transform_orbits.propagate_catalogues(mpcorb_com, until_when=march_first, initialise=False)
    # mpcorb_prop.to_hdf("../catalogues/propagated/mpcorb.h5", key="df", mode="w")
    s3m_prop = transform_orbits.propagate_catalogues(s3m, until_when=march_first, initialise=True)
    s3m_prop.to_hdf("../catalogues/propagated/s3m.h5", key="df", mode="w")

    print("Orbits propagated")
    print("Total Runtime: {:1.2f}s".format(time() - start))

def main():
    print("hey there")
    # preprocessing()


if __name__ == "__main__":
    main()