import hybridcat.data as data
import hybridcat.transform_orbits as transform_orbits
# import hybridcat.merge_catalogues as merge
import pandas as pd
from astropy.time import Time
from time import time


class HybridCreator():
    def __init__(self, catalogue_folder="catalogues/", output_folder="output/",
                 d_max=0.1, n_neighbours=100, propagate_date="2022-03-01", recreate=False,
                 dynmodel="2", save_all=True, save_final=True, verbose=False):
        self.catalogue_folder = catalogue_folder
        self.output_folder = output_folder
        self.d_max = d_max
        self.n_neighbours = n_neighbours
        self.propagate_date = propagate_date
        self.recreate = recreate
        self.dynmodel = dynmodel
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

    def propagate_catalogues(self):
        until_when = Time(self.propagate_date).mjd
        self.mpcorb = transform_orbits.propagate_catalogues(self.mpcorb, until_when=until_when,
                                                            dynmodel=self.dynmodel, initialise=True)
        self.s3m = transform_orbits.propagate_catalogues(self.s3m, until_when=until_when,
                                                         dynmodel=self.dynmodel, initialise=False)

        if self.save_final:
            self.mpcorb.to_hdf(self.catalogue_folder + "mpcorb_propagated.h5", key="df", mode="w")
            self.s3m.to_hdf(self.catalogue_folder + "s3m_propagated.h5", key="df", mode="w")

    def preprocessing(self):
        if verbose:
            print("Started preprocessing")

        self.create_initial_catalogues()
        
        if verbose:
            print("Catalogues created/read")
        
        self.mpcorb = transform_orbits.transform_catalogue(self.mpcorb, current_coords="KEP",
                                                           transformed_coords="COM")
        if self.save_all:
            self.mpcorb.to_hdf(self.catalogue_folder + "mpcorb_cometary.h5", key="df", mode="w")

        if verbose:
            print("mpcorb coordinate conversion done")
        
        self.propagate_catalogues()
        if verbose:
            print("Orbits propagated\nPreprocessing done")

def merge_it():
    the_creator = HybridCreator()
    the_creator.preprocessing()
