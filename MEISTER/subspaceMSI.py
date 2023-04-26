import sys
import json
from tqdm import tqdm
import pickle

sys.path.append('../')
from utils import *
from processing import *
import numpy as np
import random
from sklearn.utils.extmath import randomized_svd 
from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed

class Subspace():

    """
    Handles the data for the reconstruction pipelines based on the subspace approach. 
    """

    def __init__(self,out_dir):

        self.parameters = {}    #load parameter file stored as .json
        self.SpatialCoef = {}
        self.V_hat = {}     #basis transients
        self.S = {}     #singular values
        self.peak_list = {}     #list of centroided peak features for the reconstructed spectra
        self.peak_list_fid = {}     #list of centroided peak features for the original spectra
        self.random_state = 19
        self.out_dir = out_dir


    def experimentInfo(self, file_dict_path, if_simu = False, sampling_pattern = 'random'):

        """
        """
        with open(file_dict_path, 'r') as fp:
            path_dict = json.load(fp)

        self.loadParams(path_dict['parameter_file_path'])       #load data parameters
        self.imaginginfo_HR = parseImagingInfo(path_dict['imaging_info_path_HR'])     #load imaging experiment information
        self.imaginginfo_BASIS = parseImagingInfo(path_dict['imaging_info_path_BASIS'])
        self.ser_file_path_HR = path_dict['ser_file_path_HR']
        self.ser_file_path_BASIS = path_dict['ser_file_path_BASIS']
        self.simulation = if_simu
        self.sampling_pattern = sampling_pattern

        if self.simulation:

            self.ser_file_path_LR = self.ser_file_path_HR
            self.imaginginfo_LR = parseImagingInfo(path_dict['imaging_info_path_HR'])
            self.ser_file_path_LR = path_dict['ser_file_path_HR']

        else:

            if path_dict['ser_file_path_LR'] =='':
                raise Exception('no file path of the low-resolution data is provided for reconstruction.')

            if path_dict['imaging_info_path_LR'] == '':
                raise Exception('no file path of the low-resolution imaging file is provided for reconstruction.')
            else:
                self.imaginginfo_LR = parseImagingInfo(path_dict['imaging_info_path_LR'])
                self.ser_file_path_LR = path_dict['ser_file_path_LR']

        
    def loadParams(self, parameter_file_path):

        """
        """

        with open(parameter_file_path, 'r') as fp:
            self.parameters = json.load(fp)

        self.parameters['T'] = 1/self.parameters['sw_h']

        self.parameters['t_HR'] = np.arange(0, self.parameters['fid_length_HR'])*self.parameters['T']
        self.parameters['t_LR'] = np.arange(0, self.parameters['fid_length_LR'])*self.parameters['T']

        self.parameters['f_HR'] = self.parameters['sw_h']*np.arange(0, self.parameters['fid_length_HR']/2+1)/self.parameters['fid_length_HR']
        self.parameters['f_LR'] = self.parameters['sw_h']*np.arange(0, self.parameters['fid_length_LR']/2+1)/self.parameters['fid_length_LR']

        self.parameters['m_HR'] = fticr_mass_axis(self.parameters['f_HR'], self.parameters['CALIB'])
        self.parameters['m_LR'] = fticr_mass_axis(self.parameters['f_LR'], self.parameters['CALIB'])


        print('loaded parameters for the experiment...')
        print(self.parameters)


    

    def ProcSerFile(
        self, 
        ROI, 
        mz_range, 
        prominence = 10000, 
        threshold_multiplier = 5, 
        n_jobs = 8, 
        if_save = True,
        return_peaks = 'individual',
        zero_pad = False,
        mz_index = []):

        """
        Processes Bruker .ser files which contain the concatenated FIDs from the imaging runs. Due to the large size of the original
        spectra ,it only returns the centroided peak data based on the peak detection parameters. Multi-core processing was utilized.
        """
        peak_list_orig = []
        if self.simulation:
            fid_length = self.parameters['fid_length_HR']
            m = self.parameters['m_HR']
            imaginginfo = self.imaginginfo_HR[ROI]
            ser_file_path = self.ser_file_path_HR
        else:
            fid_length = self.parameters['fid_length_LR']
            m = self.parameters['m_LR']
            imaginginfo = self.imaginginfo_LR[ROI]
            ser_file_path = self.ser_file_path_LR

        fid_length_HR = self.parameters['fid_length_HR']
        m_HR = self.parameters['m_HR']
        scan_index = imaginginfo['scan_index']

        if len(mz_index) > 0 and zero_pad:
            intens_mtx = []
            average_sp = np.zeros(m_HR[(m_HR>mz_range[0])&(m_HR<mz_range[1])].size)
            for i in tqdm(range(0, scan_index.size, 100)):

                fid_idx = scan_index[i:i+100]
                fid_loaded = loadBrukerFIDs(ser_file_path, fid_length, 'all', fid_idx)
                if zero_pad:
                    pad_length = fid_length_HR - fid_loaded.shape[1]
                    fid_loaded = np.pad(fid_loaded,[(0,0),(0,pad_length)])
                    mz, sp = fid2spec(fid_loaded, m_HR, mz_range)
                    intens_mtx.append(sp[:, mz_index])
                    average_sp += np.sum(sp,0)
                else:
                    break
                
            average_sp = average_sp/fid_idx.size
            with open('{}/{}_{}_propagated_orig_zeropad.pkl'.format(self.out_dir, self.parameters['project_name'], ROI), 'wb') as fp:
                pickle.dump({'mz':m_HR[mz_index],'intens_mtx':np.concatenate(intens_mtx)}, fp, protocol=pickle.HIGHEST_PROTOCOL)
            with open('{}/{}_{}_avgspec_orig_zeropad.pkl'.format(self.out_dir, self.parameters['project_name'], ROI), 'wb') as fp:
                pickle.dump(average_sp, fp, protocol=pickle.HIGHEST_PROTOCOL)

            return

        if return_peaks == 'average':
            print('calculating average spectrum...')

            fid_loaded = loadBrukerFIDs(ser_file_path, fid_length, 'all', imaginginfo['scan_index'][0])
            mz, sp = fid2spec(fid_loaded, m, mz_range)
            average_sp = np.zeros(sp.shape)

        for i in tqdm(range(0, scan_index.size, 100)):

            fid_idx = scan_index[i:i+100]
            fid_loaded = loadBrukerFIDs(
                ser_file_path, 
                fid_length, 
                'all', 
                fid_idx)

            if return_peaks == 'individual':
                pack = Parallel(n_jobs=n_jobs,verbose=51)(delayed(self.fid2peaks)(
                    fid, 
                    m, 
                    mz_range, prominence, 
                    threshold_multiplier) for fid in fid_loaded)

                for peak_list in pack:
                    peak_list_orig.append(peak_list)
            
            if return_peaks == 'average':

                mz, sp = fid2spec(fid_loaded, m, mz_range)
                #for j in range(sp.shape[0]):
                average_sp += np.sum(sp,0)

        if return_peaks == 'average':

            average_sp = average_sp/scan_index.size
            self.average_spec_orig = average_sp.flatten()

            peak_list_orig = peak_detection(mz, self.average_spec_orig, prominence = mad(self.average_spec_orig), threshold = mad(self.average_spec_orig)*threshold_multiplier)
        
            print('propagating intensities from the average spectrum')
            mz_index = peak_list_orig['mz_index']

            print('now propagating {} number of detected peaks...'.format(len(mz_index)))

            intens_mtx = []
            TIC = []

            for i in tqdm(range(0, scan_index.size, 100)):

                fid_idx = scan_index[i:i+100]
                fid_loaded = loadBrukerFIDs(ser_file_path, fid_length, 'all', fid_idx)
                mz, sp = fid2spec(fid_loaded, m, mz_range)
                TIC.append(np.sum(sp, 1))
                intens_mtx.append(sp[:, mz_index])

            with open('{}/{}_{}_propagated_orig.pkl'.format(self.out_dir, self.parameters['project_name'], ROI), 'wb') as fp:
                pickle.dump({'mz':peak_list_orig['mz'],'intens_mtx':np.concatenate(intens_mtx),'tic':np.concatenate(TIC)}, fp, protocol=pickle.HIGHEST_PROTOCOL)

            self.peak_list_orig = peak_list_orig
            
        if if_save:
            with open('{}/{}_{}_peak_data_orig.pkl'.format(self.out_dir, self.parameters['project_name'], ROI), 'wb') as fp:
                pickle.dump(
                    peak_list_orig, 
                    fp, 
                    protocol=pickle.HIGHEST_PROTOCOL)

            del peak_list_orig



    def Basis(
        self, 
        ROI, 
        solver='randomized'):


        #sampling from the full high-resolution dataset for basis extraction.

        if self.sampling_pattern == 'random':

            random.seed(self.random_state)

            fid_idx_sampled = np.array(random.sample(list(self.imaginginfo_BASIS[ROI]['scan_index']), self.parameters['pixel_num_HR']))
            
            fid_idx_sampled = fid_idx_sampled[np.argsort(fid_idx_sampled)]

            self.fid_idx_basis = fid_idx_sampled

            print('loading {} FID from file...'.format(len(fid_idx_sampled)))

            S1 = loadBrukerFIDs(self.ser_file_path_BASIS, 
                                self.parameters['fid_length_HR'], 
                                'all', 
                                fid_idx_sampled)

        if self.sampling_pattern == 'grid':

            fid_idx = np.linspace(
                1, self.imaginginfo_BASIS[ROI]['scan_index'].shape[0], 
                self.imaginginfo_BASIS[ROI]['scan_index'].shape[0])

            print('loading {} FID from file...'.format(len(fid_idx)))

            S1 = loadBrukerFIDs(
                self.ser_file_path_BASIS, 
                self.parameters['fid_length_HR'], 
                'all', 
                fid_idx)

        S, Vt = self.extractBasis(S1, solver)

        self.V_hat[ROI] = Vt
        self.S[ROI] = S



    def FitBasis(
        self, 
        ROI, 
        basis_ROI, 
        mz_range, 
        return_peaks = 'individual', 
        save_peaks = True,
        prominence = 10000, 
        threshold_multiplier = 5, 
        n_jobs = 8):

        """
        Fits spatial coefficients.
        """
        if save_peaks:

            peak_list = []

        U = []        #initialize spatial coefficients

        #for i in tqdm(range(len(self.imaginginfo_LR[ROI]['scan_index']))):
        if self.simulation:
            scan_index = self.imaginginfo_HR[ROI]['scan_index']
        else:
            scan_index = self.imaginginfo_LR[ROI]['scan_index']
            
        print('fitting coefficients for {} FID for reconstruction...'.format(len(scan_index)))

        for i in tqdm(range(0, scan_index.size, n_jobs)):

            if self.simulation:

                fid_idx = scan_index[i:i+n_jobs]
                fid_loaded = loadBrukerFIDs(
                    self.ser_file_path_HR, 
                    self.parameters['fid_length_HR'], 
                    self.parameters['fid_length_LR'], 
                    fid_idx)

                U_ = self.fitSpatialCoef(
                    A = fid_loaded, 
                    V_hat = self.V_hat[basis_ROI], 
                    aa = 0)

                U.append(U_)

            else:

                #print('processing to reconstruct transients...{} out of {}'.format(i,len(self.imaginginfo_LR[ROI]['scan_index'])))

                fid_idx = scan_index[i:i+n_jobs]
                #fid = loadBrukerFIDs(self.ser_file_path_LR, self.parameters['fid_length_LR'], 'all', idx)
                fid_loaded = loadBrukerFIDs(
                    self.ser_file_path_LR, 
                    self.parameters['fid_length_LR'], 
                    'all', 
                    fid_idx)

                U_ = self.fitSpatialCoef(
                    A = fid_loaded, 
                    V_hat = self.V_hat[basis_ROI], 
                    aa = 0)

                U.append(U_)

            if return_peaks == 'individual':

                pack = Parallel(n_jobs=n_jobs,verbose=51)(delayed(self.coef2Peaks)(
                    U_[:,i], 
                    self.V_hat[basis_ROI], 
                    self.parameters['m_HR'], 
                    mz_range, 
                    prominence, 
                    threshold_multiplier) for i in range(U_.shape[1]))

                peak_list += pack

        self.SpatialCoef[ROI] = np.concatenate(U,axis=1)

        if return_peaks == 'average':

            print('calculating average spectrum...')
            
            _, mz, average_sp = self.getSpecFromCoef(self.SpatialCoef[ROI][:,0], self.V_hat[basis_ROI], self.parameters['m_HR'], mz_range)
            average_sp = average_sp.reshape(1,-1)
            for i in tqdm(range(1, self.SpatialCoef[ROI].shape[1], 100)):

                _, _, sp = self.getSpecFromCoef(self.SpatialCoef[ROI][:,i:i+100], self.V_hat[basis_ROI], self.parameters['m_HR'], mz_range)
                #for j in range(sp.shape[0]):
                average_sp += np.sum(sp,0)

            average_sp = average_sp/scan_index.size

            self.average_spec_recon = average_sp.flatten()

            peak_list = peak_detection(mz, self.average_spec_recon, prominence = mad(self.average_spec_recon)*threshold_multiplier, threshold = mad(self.average_spec_recon)*threshold_multiplier)

            self.peak_list_recon = peak_list

        if save_peaks:

            with open('{}/{}_{}_peak_data_recon.pkl'.format(self.out_dir, self.parameters['project_name'], ROI), 'wb') as fp:
                pickle.dump(peak_list, fp, protocol=pickle.HIGHEST_PROTOCOL)

            del peak_list

                
    def FindPeaks(
        self, 
        ROI, 
        basis_ROI, 
        mz_range, 
        prominence = 10000, 
        threshold_multiplier = 5, 
        n_jobs = 8, 
        if_save = True):

        """
        """
        
        print('performing peak detection for {} reconstructed spectra'.format(self.SpatialCoef[ROI].shape[1]))

        pack = Parallel(n_jobs=n_jobs,verbose=51)(delayed(self.coef2Peaks)(
            ROI, 
            basis_ROI, 
            i, 
            mz_range, 
            prominence, 
            threshold_multiplier) for i in tqdm(range(self.SpatialCoef[ROI].shape[1])))
        
        self.peak_list[ROI] = pack

        if if_save:
            with open('{}/{}_{}_peak_data_recon.pkl'.format(self.out_dir, self.parameters['project_name'], ROI), 'wb') as fp:
                pickle.dump(pack, fp, protocol=pickle.HIGHEST_PROTOCOL)



    def coef2Peaks(
        self, 
        U, 
        V_hat, 
        m, 
        mz_range, 
        prominence, 
        threshold_multiplier):

        """
        """
        _, mz, sp = self.getSpecFromCoef(U, V_hat, m, mz_range)

        peak_list = peak_detection(mz, sp, prominence = prominence, threshold = mad(sp)*threshold_multiplier)

        return peak_list


    def fid2peaks(
        self, 
        fid, 
        m, 
        mz_range, 
        prominence = 10000, 
        threshold_multiplier = 5):

        """
        """
        mz, sp = fid2spec(fid, m, mz_range)
        
        peak_list = peak_detection(
            mz, 
            sp, 
            prominence = prominence, 
            threshold = mad(sp)*threshold_multiplier)

        return peak_list



    def extractBasis(self, A, solver):

        """
        Performs singular value decomposition to obtain basis transients from the provided
        dataset A. The number of basis returned is defined by users through n_basis.
        """

        n_basis = self.parameters['n_basis']
        print('Extracting {} basis transients from {} number of high-resolution transients'.format(n_basis, A.shape[0]))

        if A.shape[0] > 1000:
            print('Decomposition on large number of transients...randomized SVD is recommended')

        if solver == 'randomized':
            print('Now performing randomized SVD...')

            U, S, Vt = randomized_svd(
                A, 
                n_components = n_basis, 
                random_state = 19, 
                n_iter = 10)

        else:
            print('Now performing exact SVD...it may take awhile')

            svd = TruncatedSVD(
                n_components = n_basis, 
                algorithm = 'arpack')

            svd.fit(A)

            Vt = svd.components_
            S = svd.singular_values_

        return S, Vt


    def fitSpatialCoef(self, A, V_hat, aa):

        """
        fits a set of coefficients through least-squares for each FID at a given pixel with 
        the basis transients V_hat. X can be a vector (a single FID) or a matrix (a set of FIDs)
        """

        L_lr = self.parameters['fid_length_LR']
        n_basis = self.parameters['n_basis']

        proj_op = np.dot(np.linalg.inv(np.dot(V_hat[:, :L_lr], V_hat[:, :L_lr].T) + aa*np.identity(n_basis)), V_hat[:, :L_lr])

        U = np.dot(proj_op, A.T)

        #self.U = np.concatenate((self.U, U), axis = 1)
        #self.U.append(U)

        return U


    def getSpecFromCoef(self, U, V_hat, m, mz_range):

        """
        linearly combine a set of basis transients using spatial coefficients to reconstruct 
        a mass spectrum A_hat with a defined mass range
        """

        A_hat = np.dot(U.T, V_hat)

        mz, sp = fid2spec(A_hat, m, mz_range)

        return A_hat, mz, sp


    