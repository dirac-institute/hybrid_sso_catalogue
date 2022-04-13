import sys
from .data import *
from .transform import *
from .merge import *
from astropy.time import Time
import numpy as np
import pandas as pd
import argparse


class HybridCreator():
    def __init__(self, catalogue_folder="catalogues/", output_folder="output/",
                 d_max=0.1, n_neighbours=100, propagate_date="2022-03-01", recreate=False,
                 dynmodel="2", H_bins=np.arange(-2, 28 + 1), n_workers=48, memory_limit="16GB", timeout=300,
                 save_all=True, save_final=True, verbose=False):
        """HybridCreator masterClass used for creating a hybrid catalogue

        Parameters
        ----------
        catalogue_folder : `str`, optional
            Path to the folder containing catalogues, by default "catalogues/"
        output_folder : `str`, optional
            Path to folder in which to place temporary output during merging, by default "output/"
        d_max : `float`, optional
            Maximum distance of a neighbour in AU, by default 0.1
        n_neighbours : `int`, optional
            Number of neighbours to consider in the matching, by default 100
        propagate_date : `str`, optional
            Date to which you want to propagate the catalogues, by default "2022-03-01"
        recreate : `bool`, optional
            Whether to recreate files if they already exist, by default False
        dynmodel : `str`, optional
            Which dynmodel to use (2-body or N-body), by default "2"
        H_bins : `list`, optional
            Magnitude bins to split the catalogues into during merge, by default np.arange(-2, 28 + 1)
        n_workers : `int`, optional
            How many workers to use with dask whilst merging, by default 48
        memory_limit : `str`, optional
            How much memory to assign each worker, by default "16GB"
        timeout : `int`, optional
            Dask timeout, by default 300
        save_all : `bool`, optional
            Whether to save every dataframe at each preprocessing step, by default True
        save_final : `bool`, optional
            Whether to save the final dataframes after all preprocessing, by default True
        verbose : `bool`, optional
            Whether to print stuff about progress, by default False
        """
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
        self.init = True

    def create_initial_catalogues(self):
        """ Create the initial catalogues or simply read them if they exist """
        self.mpcorb = create_mpcorb_from_json(in_path=self.catalogue_folder + "mpcorb_extended.json",
                                              out_path=self.catalogue_folder + "mpcorb_initial.h5",
                                              recreate=self.recreate, save=self.save_all)
        if self.verbose:
            print("\tMPCORB df created from JSON file")

        self.s3m = create_s3m_from_files(in_path=self.catalogue_folder + "/s3m_files/",
                                         out_path=self.catalogue_folder + "s3m_initial.h5",
                                         recreate=self.recreate, save=self.save_all)
        
        if self.verbose:
            print("\tS3M df created from files")

    def transform_both_to_cart(self):
        """Transform both catalogues to cartesian coordinates to avoid singularities"""
        self.mpcorb = transform_catalogue(self.mpcorb, initialise=self.init,
                                          current_coords="KEP", transformed_coords="CART")
        self.init = False
        
        if self.verbose:
            print("\tMPCORB transformed from KEP to CART")
        
        self.s3m = transform_catalogue(self.s3m, initialise=self.init,
                                       current_coords="COM", transformed_coords="CART")
        
        if self.verbose:
            print("\tS3M transformed from COM to CART")
        
        if self.save_all or self.save_final:
            self.mpcorb.to_hdf(self.catalogue_folder + "mpcorb_cart.h5", key="df", mode="w")
            self.s3m.to_hdf(self.catalogue_folder + "s3m_cart.h5", key="df", mode="w")

    def propagate_catalogues(self):
        """ Propagate both catalogues to the same time """
        until_when = Time(self.propagate_date).mjd
        # initialise=False so that openorb doesn't get mad
        self.mpcorb = propagate_catalogues(self.mpcorb, until_when=until_when, coords="CART",
                                           dynmodel=self.dynmodel, initialise=self.init)
        self.init = False
        
        if self.verbose:
            print("\tMPCORB propagated until {}".format(self.propagate_date))
        
        self.s3m = propagate_catalogues(self.s3m, until_when=until_when, coords="CART",
                                        dynmodel=self.dynmodel, initialise=self.init)
        
        if self.verbose:
            print("\tS3M propagated until {}".format(self.propagate_date))

        if self.save_all:
            self.mpcorb.to_hdf(self.catalogue_folder + "mpcorb_propagated_cart.h5", key="df", mode="w")
            self.s3m.to_hdf(self.catalogue_folder + "s3m_propagated_cart.h5", key="df", mode="w")

    def preprocessing(self):
        """ Preprocess the mpcorb and s3m catalogues to prepare for merging """
        if self.verbose:
            print("Started preprocessing")

        self.create_initial_catalogues()
        if self.verbose:
            print("Catalogues created/read")

        self.transform_both_to_cart()
        if self.verbose:
            print("Both transformed to cartesian")
        
        self.propagate_catalogues()
        if self.verbose:
            print("Orbits propagated\nPreprocessing done")

    def merge_catalogues(self):
        """ Merge the two catalogues! (Output saved in self.output_folder) """
        if self.verbose:
            print("Starting the merge now")
        merge_catalogues(self.mpcorb, self.s3m, output_folder=self.output_folder, H_bins=self.H_bins,
                         n_workers=self.n_workers, memory_limit=self.memory_limit, timeout=self.timeout,
                         k=self.n_neighbours, d_max=self.d_max)
        if self.verbose:
            print("Done merging!")

    def transform_both_to_cometary(self):
        """ Convert both catalogues to cometary coordinates so hybrid matches S3m """
        self.mpcorb = transform_catalogue(self.mpcorb, current_coords="CART", transformed_coords="COM",
                                          initialise=self.init)
        self.init = False
        
        if self.verbose:
            print("\tMPCORB transformed from CART to COM")
        self.s3m = transform_catalogue(self.s3m, current_coords="CART", transformed_coords="COM",
                                       initialise=self.init)
        
        if self.verbose:
            print("\tS3m transformed from CART to COM")

        # convert back to degrees for the angles
        self.mpcorb["i"] = np.rad2deg(self.mpcorb["i"])
        self.mpcorb["argperi"] = np.rad2deg(self.mpcorb["argperi"])
        self.mpcorb["Omega"] = np.rad2deg(self.mpcorb["Omega"])
        self.s3m["i"] = np.rad2deg(self.s3m["i"])
        self.s3m["argperi"] = np.rad2deg(self.s3m["argperi"])
        self.s3m["Omega"] = np.rad2deg(self.s3m["Omega"])

        if self.save_all:
            self.mpcorb.to_hdf(self.catalogue_folder + "mpcorb_propagated_cometary.h5", key="df", mode="w")
            self.s3m.to_hdf(self.catalogue_folder + "s3m_propagated_cometary.h5", key="df", mode="w")

    def save_hybrid(self):
        """ Save the hybrid catalogue as a new h5 file """
        if self.verbose:
            print("Building hybrid catalogue")

        # convert both catalogues to cometary coordinates
        self.transform_both_to_cometary()

        if self.verbose:
            print("Both catalogues transformed back to cometary")

        # first work out which S3m objects we can delete
        delete_these = []
        for left, right in zip(self.H_bins[:-1], self.H_bins[1:]):
            matched = np.load(self.output_folder + "matched_{}_{}.npy".format(left, right))
            delete_these.extend(matched)
            
        if self.verbose:
            print("Replacement IDs collected")

        # delete them
        remaining_s3m = self.s3m.drop(self.s3m.iloc[delete_these].index.values, axis=0)

        if self.verbose:
            print("IDs dropped from S3M")
        
        # add in the MPCORB objects (that have reasonable magnitudes)
        self.hybrid = pd.concat([remaining_s3m, self.mpcorb[self.mpcorb.H < 35]])
        
        if self.verbose:
            print("MPCORB injected into modified S3M")

        # save it and celebrate
        self.hybrid.to_hdf(self.catalogue_folder + "hybrid.h5", key="df", mode="w")

def merge_it():
    """ Quick function that does the whole merging process (using this as entry point) """
    parser = argparse.ArgumentParser(description='Merge MPCORB and S3m into a hybrid catalogue')
    parser.add_argument('-c', '--catalogue_folder', default="catalogues/", type=str,
                        help='Path to the folder containing catalogues, by default "catalogues/"')
    parser.add_argument('-o', '--output_folder', default="output/", type=str,
                        help='Path to folder in which to place temporary output during merging,\
                              by default "output/"')
    parser.add_argument('-d', '--max-distance', default=0.1, type=float,
                        help='Maximum distance of a neighbour in AU, by default 0.1')
    parser.add_argument('-n', '--neighbours', default=100, type=int,
                        help='Number of neighbours to consider in the matching, by default 100')
    parser.add_argument('-p', '--propagate-date', default="2022-03-01", type=str,
                        help='Date to which you want to propagate the catalogues, by default "2022-03-01"')
    parser.add_argument('-r', '--recreate-files', action="store_true",
                        help='Whether to recreate files if they already exist, by default False')
    parser.add_argument('-m', '--dynmodel', default="2", type=str,
                        help='Which dynmodel to use (2-body or N-body), by default "2"')
    parser.add_argument('-b', '--H-bins', default=list(range(-2, 28 + 1)), type=list,
                        help='Magnitude bins to split the catalogues into during merge,\
                              by default range(-2, 28 + 1)')
    parser.add_argument('-w', '--workers', default=48, type=int,
                        help='How many workers to use with dask whilst merging, by default 48')
    parser.add_argument('-l', '--mem-limit', default="16GB", type=str,
                        help='How much memory to assign each worker, by default "16GB"')
    parser.add_argument('-t', '--timeout', default=300, type=int,
                        help='Dask timeout, by default 300')
    parser.add_argument('-s', '--save-all', action="store_true",
                        help='Whether to save every dataframe at each preprocessing step, by default True')
    parser.add_argument('-f', '--save-final', action="store_true",
                        help='Whether to save the final dataframes after all preprocessing, by default True')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Whether to print stuff about progress, by default False')
    parser.add_argument('-S', "--skip-prep", action="store_true",
                        help="Whether to skip the preprocessing, by default False")

    parser.set_defaults(save_all=True, save_final=True)

    args = parser.parse_args()

    the_creator = HybridCreator(catalogue_folder=args.catalogue_folder, output_folder=args.output_folder,
                                d_max=args.max_distance, n_neighbours=args.neighbours,
                                propagate_date=args.propagate_date, recreate=args.recreate_files,
                                dynmodel=args.dynmodel, H_bins=args.H_bins, n_workers=args.workers,
                                memory_limit=args.mem_limit, timeout=args.timeout, save_all=args.save_all,
                                save_final=args.save_final, verbose=args.verbose)
    if args.skip_prep:
        the_creator.mpcorb = pd.read_hdf(the_creator.catalogue_folder + "mpcorb_propagated_cart.h5", key="df")
        the_creator.s3m = pd.read_hdf(the_creator.catalogue_folder + "s3m_propagated_cart.h5", key="df")
    else:
        the_creator.preprocessing()
    the_creator.merge_catalogues()
    the_creator.save_hybrid()


if __name__ == "__main__":
    merge_it()
