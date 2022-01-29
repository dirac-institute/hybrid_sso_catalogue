import hybridcat.data as data
import hybridcat.transform as transform
import hybridcat.merge as merge
from astropy.time import Time
import numpy as np
import pandas as pd


class HybridCreator():
    def __init__(self, catalogue_folder="catalogues/", output_folder="output/",
                 d_max=0.1, n_neighbours=100, propagate_date="2022-03-01", recreate=False,
                 dynmodel="2", H_bins=np.arange(-2, 28 + 1), n_workers=48, memory_limit="16GB", timeout=300,
                 save_all=True, save_final=True, verbose=False):
        self.catalogue_folder = catalogue_folder
        self.output_folder = output_folder
        self.d_max = d_max
        self.n_neighbours = n_neighbours
        self.propagate_date = propagate_date
        self.recreate = recreate
        self.dynmodel = dynmodel
        self.H_bins = H_bins
        self.n_workers = n_workers
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.save_all = save_all
        self.save_final = save_final
        self.verbose = verbose

    def create_initial_catalogues(self):
        self.mpcorb = data.create_mpcorb_from_json(in_path=self.catalogue_folder + "mpcorb_extended.json",
                                                   out_path=self.catalogue_folder + "mpcorb_initial.h5",
                                                   recreate=self.recreate, save=self.save_all)

        self.s3m = data.create_s3m_from_files(in_path=self.catalogue_folder + "/s3m_files/",
                                              out_path=self.catalogue_folder + "s3m_initial.h5",
                                              recreate=self.recreate, save=self.save_all)

    def transform_mpcorb(self):
        self.mpcorb = transform.transform_catalogue(self.mpcorb,
                                                    current_coords="KEP", transformed_coords="COM")
        # convert back to degress for the angles
        self.mpcorb.i = np.rad2deg(self.mpcorb.i)
        self.mpcorb.argperi = np.rad2deg(self.mpcorb.argperi)
        self.mpcorb.Omega = np.rad2deg(self.mpcorb.Omega)
        if self.save_all:
            self.mpcorb.to_hdf(self.catalogue_folder + "mpcorb_cometary.h5", key="df", mode="w")

    def propagate_catalogues(self):
        until_when = Time(self.propagate_date).mjd
        self.mpcorb = transform.propagate_catalogues(self.mpcorb, until_when=until_when,
                                                     dynmodel=self.dynmodel, initialise=True)
        self.s3m = transform.propagate_catalogues(self.s3m, until_when=until_when,
                                                  dynmodel=self.dynmodel, initialise=False)

        if self.save_final:
            self.mpcorb.to_hdf(self.catalogue_folder + "mpcorb_propagated.h5", key="df", mode="w")
            self.s3m.to_hdf(self.catalogue_folder + "s3m_propagated.h5", key="df", mode="w")

    def preprocessing(self):
        if self.verbose:
            print("Started preprocessing")

        self.create_initial_catalogues()
        if self.verbose:
            print("Catalogues created/read")
        
        self.transform_mpcorb()
        if self.verbose:
            print("mpcorb coordinate conversion done")
        
        self.propagate_catalogues()
        if self.verbose:
            print("Orbits propagated\nPreprocessing done")

    def merge_catalogues(self):
        if self.verbose:
            print("Starting the merge now")
        merge.merge_catalogues(self.mpcorb, self.s3m, output_folder=self.output_folder, H_bins=self.H_bins,
                               n_workers=self.n_workers, memory_limit=self.memory_limit, timeout=self.timeout,
                               k=self.n_neighbours, d_max=self.d_max)
        if self.verbose:
            print("Done merging!")

    def save_hybrid(self):
        # first work out which S3m objects we can delete
        delete_these = []
        for left, right in zip(self.H_bins[:-1], self.H_bins[1:]):
            matched = np.load(self.output_folder + "matched_{}_{}.npy".format(left, right))
            delete_these.extend(matched)

        # delete them
        remaining_s3m = self.s3m.drop(delete_these, axis=0)

        # add in the MPCORB objects (that have reasonable magnitudes)
        self.hybrid = pd.concat([remaining_s3m, self.mpcorb[self.mpcorb.H < 35]])

        # save it and celebrate
        self.hybrid.to_hdf(self.catalogue_folder + "hybrid.h5", key="df", mode="w")

def merge_it():
    the_creator = HybridCreator()
    the_creator.preprocessing()
    the_creator.merge_catalogues()
    the_creator.save_hybrid()
