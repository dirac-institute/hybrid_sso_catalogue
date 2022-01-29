import hybridcat.data as data
import hybridcat.transform_orbits as transform_orbits
# import hybridcat.merge_catalogues as merge
import pandas as pd
from astropy.time import Time
from time import time


class Hybrid():
    def __init__(self, catalogue_folder="catalogues/", output_folder="output/",
                 d_max=0.1, n_neighbours=100, propagate_date="2022-03-01", recreate=False,
                 save_all=True, save_final=True):
        self.catalogue_folder = catalogue_folder
        self.output_folder = output_folder
        self.d_max = d_max
        self.n_neighbours = n_neighbours
        self.propagate_date = propagate_date
        self.recreate = recreate
        self.save_all = save_all
        self.save_final = save_final

    def create_initial_catalogues(self):
        self.mpcorb = data.create_mpcorb_from_json(in_path=self.catalogue_folder + "mpcorb_extended.json",
                                                   out_path=self.catalogue_folder + "mpcorb_initial.h5",
                                                   recreate=self.recreate, save=self.save_all)

        self.s3m = data.create_s3m_from_files(in_path=self.catalogue_folder + "/s3m_files/",
                                              out_path=self.catalogue_folder + "s3m_initial.h5",
                                              recreate=self.recreate, save=self.save_all)

    def transform_catalogues

    def preprocessing(self):
        print("Started")
        start = time()

        self.create_initial_catalogues()
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
    pass
    # preprocessing()


def test():
    print("hey there tom")


if __name__ == "__main__":
    main()