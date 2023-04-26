import sys
import json
from tqdm import tqdm
import pickle
import os
import pandas as pd

sys.path.append('../')
from utils import *
from processing import *
import numpy as np
import random
from joblib import Parallel, delayed
from pyimzml.ImzMLWriter import ImzMLWriter
from pyImagingMSpec.inMemoryIMS import inMemoryIMS
from scipy.stats import median_abs_deviation as mad
import matplotlib.pyplot as plt


class scMSData():

    """
    """
    def __init__(self):

        self.parameters = {}    #load parameter file stored as .json
        self.spectra = {}       #raw spectra
        self.peak_list = {}
        self.coords = []        #cell x,y coordinates, e.g. tuple(71232, 10321)
        self.random_state = 19
        self.use_index = []
        self.file_paths = []
        self.names = []


    def getXMLPath(self, path):
        """
        """

        file_paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".xml"):
                     file_paths.append(os.path.join(root, file))

        self.file_paths += file_paths
        self.names += [file_path.split('/')[-1].split('.')[0] for file_path in file_paths]

        return file_paths


    def loadXMLData(self,detailed=False):
        """
        """

        for i in tqdm(range(len(self.file_paths))):
            peak_data = parseBrukerXML(self.file_paths[i], detailed)
            if len(peak_data['mzs']) >0:
                self.peak_list[self.names[i]] = peak_data

        self.names = list(self.peak_list.keys())


    def loadXMLData_tof(self,detailed=False):

        for i in tqdm(range(len(self.file_paths))):
            self.peak_list[self.names[i]] = parseBrukerXML_tof(self.file_paths[i], detailed)



    def getIntensMtxData(self, mz_range, feature_n, ppm=2, from_imzml=True, mz_features=[], intens_array=np.array([])):
        """
        """

        if from_imzml:

            if len(mz_features) == 0:
                intens_array, mz_bins_use, c = extractMZFeatures(self.imzML_dataset, ppm, mz_range, feature_n=feature_n)
            else:
                intens_array, mz_bins_use, c = extractMZFeatures(self.imzML_dataset, ppm, mz_range, feature_n=feature_n, mz_bins=mz_features)

            intens_mtx = pd.DataFrame(intens_array, columns=mz_bins_use, index=self.names)

        else:
            intens_mtx = pd.DataFrame(intens_array, columns=mz_features, index=self.names)

        self.intens_mtx = intens_mtx



    def loadPeakList(self, file_name):

        with open(file_name, "rb") as file:
            self.peak_list=pickle.load(file)
        
        self.names = list(self.peak_list.keys())


    
    def loadimzMLData(self, file_name):

        self.imzML_dataset = inMemoryIMS(file_name)


    def convertPeak2imzML(self, file_name):
        """
        """

        coords = [tuple([1,i]) for i in range(len(self.names))]
        pklist2imzML(self.peak_list, file_name, coords)

        self.loadimzMLData(file_name+'.imzML')


    def getParams(self, method_file_path):

        """
        """
        parameters = parseBrukerMethod(method_file_path)

        T = 1/(parameters['SW_h']*2)
        t = np.arange(0, parameters['TD'])*T
        f = parameters['SW_h']*2*np.arange(0, parameters['TD']/2+1)/parameters['TD']
        m = fticr_mass_axis(f, [parameters['ML1'],parameters['ML2'],parameters['ML3']])

        return {'parameters':parameters,'T':T,'t':t,'f':f,'m':m}


    def loadICRData(self, path, method_file_path=None):
        """
        """
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith("apexAcquisition.method"):
                    method_file_path = os.path.join(root, file)


        if method_file_path == None:
            params = None
            fid = np.array([])
        else: 
            params = self.getParams(method_file_path)
            fid = loadBrukerFIDs(path+'/ser', params['parameters']['TD'],'all', 1)

        return params, fid


    def getICRSpectra(self, path, mz_range):

        #params, fid = self.loadICRData(path)
        D = proc_solarix(path)

        if D.axis1.mz_axis().size >0:
            mz, sp = fid2spec_solarix(D, mz_range)
        else:
            mz = np.array([])
            sp = np.array([])

        return mz, sp


    def processICRData_solarix(self, paths, mz_range, thres, centroid):
        """
        """
        self.names = []

        for i in tqdm(range(len(paths))):
            path = paths[i]
            mz, sp = self.getICRSpectra(path, mz_range)

            if sp.size > 0:
                if return_peak:
                    MAD = mad(sp)
                    peak_list = peak_detection(mz, sp, prominence = MAD*prominence_multiplier, threshold = MAD*thres_multipier)

                    self.peak_list[path] = peak_list
                    self.names.append(path)


    def processICRData(self, paths, mz_range, return_peak = True, prominence_multiplier = 5, thres_multipier = 5):
        """
        """
        self.names = []

        for i in tqdm(range(len(paths))):
            path = paths[i]
            mz, sp = self.getICRSpectra(path, mz_range)

            if sp.size > 0:
                if return_peak:
                    MAD = mad(sp)
                    peak_list = peak_detection(mz, sp, prominence = MAD*prominence_multiplier, threshold = MAD*thres_multipier)

                    self.peak_list[path] = peak_list
                    self.names.append(path)


    def show_ICRSpectra(self, path, mz_low, mz_high, peak_centroid = True):

        plt.close()

        plt.figure(figsize=(8,4))
        mz, sp = self.getICRSpectra(path,mz_range=(mz_low,mz_high))
        plt.plot(mz, sp)

        if peak_centroid:
            mzs = self.peak_list[path]['mzs']
            intens = self.peak_list[path]['intensity']
            plt.stem(mzs[(mzs>mz_low)&(mzs<mz_high)], intens[(mzs>mz_low)&(mzs<mz_high)],
             markerfmt=' ', basefmt=' ', linefmt='r')
        plt.show()
