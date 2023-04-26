r"""Subspace reconstruction of the low-res MSI transients from the predefined
high resolution basis. Basis transients need to be computed from the given high
resolution experimental transients. 


"""
import os
import argparse
import logging
import pickle
from tqdm import tqdm
import subspaceMSI as subspace
from processing import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'A subspace approach to reconstruct FT-ICR mass spectrometry imaging data.'
        )
    parser.add_argument(
        '--path_file', required = True, type = str,
        help='Path to the .json file that specifies the raw .ser and imaginginfo files.'
        )
    parser.add_argument(
        '--out_dir', required = True, type = str,
        help='An output directory to store the processed results.'
        )
    parser.add_argument(
        '--basis_ROI', required = True, type = str,
        help='The ROI(region of interest) pointer to the imaginginfo file to extract basis transients from high resolution experimental long transients from the ser file.'
        )
    parser.add_argument(
        '--recon_ROI', required = True, type = str,
        help='The ROI(region of interest) pointer to the imaginginfo file to reconstruct long transients from the defined basis.'
        )
    parser.add_argument(
        '--basis_path', type = str,
        help='The path to the predefined basis file stored as a numpy array.'
        )
    parser.add_argument(
        '--mz_range', required = True, type = int, nargs = '+',
        help='Two values that defines the lower and upper bound of m/z range for analyzing spectra.'
        )
    parser.add_argument(
        '--return_peaks', nargs = '?', const = 'individual', default = 'individual', type = str, choices = ['none','individual','average'],
        help='Wether to return peaks of individual spectra. If set True, then the peak lists are saved as a .pkl file'
        )
    parser.add_argument(
        '--n_jobs', nargs = '?', const = 1, default = 1, type = int,
        help='Number of jobs to run in parallel.'
        )
    parser.add_argument(
        '--svd_type', nargs = '?', const = 'exact', default = 'exact', type = str, choices = ['exact','randomized'],
        help='Type of SVD for decomposing long transients (exact or randomized).'
        )
    parser.add_argument(
        '--if_simulate', nargs = '?', const = 'False', default = 'False', type = str, choices = ['False','True'],
        help='If evaluate on the original high resolution data, which involves truncating the measured long transients. If false, the long transients data are not given.'
        )
    parser.add_argument(
        '--bin_method', nargs = '?', const = 'none', default = 'none', type = str, choices = ['none','histogram','average'],
        help='The binning method for the generated peaks.'
        )
    parser.add_argument(
        '--sampling_pattern', nargs = '?', const = 'random', default = 'random', type = str, choices = ['random','grid'],
        help='Sampling method to compute the basis transients.'
        )
    parser.add_argument(
        '--if_process_raw', nargs = '?', const = 'False', default = 'False', type = str, choices = ['True','False'],
        help='If evaluate on the original high resolution data, which involves truncating the measured long transients. If false, the long transients data are not given.'
        )
    
    args = parser.parse_args()

    print('-'*40, 'script parameters','-'*40)
    print(args)


    recon = subspace.Subspace(out_dir = args.out_dir)
    print('-'*40, 'experimental parameters','-'*40)

    if args.if_simulate == 'True':
        if_simu = True
    else:
        if_simu = False

    if args.if_process_raw == 'True':
        process_raw = True
    else:
        process_raw = False

    recon.experimentInfo(
        args.path_file, 
        if_simu = if_simu, 
        sampling_pattern = args.sampling_pattern)

    print('-'*40, 'processing original raw','-'*40)
    if process_raw:
        recon.ProcSerFile(ROI = args.recon_ROI, mz_range = args.mz_range, return_peaks = args.return_peaks)


    print('-'*40, 'get the basis','-'*40)

    if not args.basis_path:
        print('-'*40, 'computing basis from long transients','-'*40)
    
        recon.Basis(
            ROI = args.basis_ROI, 
            solver = args.svd_type)
        with open('{}/{}_{}_basis.pkl'.format(args.out_dir, recon.parameters['project_name'], args.basis_ROI), 'wb') as f:
            pickle.dump(
                recon.V_hat[args.basis_ROI], 
                f, 
                protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('loading the pre-computed basis...')
        with open(args.basis_path, 'rb') as fp:
            recon.V_hat[args.basis_ROI] = pickle.load(fp)
    
    print('-'*40, 'fitting basis transients','-'*40)
    
    recon.FitBasis(
        ROI = args.recon_ROI, 
        basis_ROI = args.basis_ROI, 
        mz_range = args.mz_range, 
        return_peaks = args.return_peaks, 
        n_jobs = args.n_jobs)
    
    with open('{}/{}_{}_SpatialCoef.pkl'.format(args.out_dir, recon.parameters['project_name'], args.recon_ROI), 'wb') as f:
            pickle.dump(recon.SpatialCoef[args.recon_ROI], f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.return_peaks == 'average':
        if process_raw:
            with open('{}/{}_{}_avgsp_orig.pkl'.format(args.out_dir, recon.parameters['project_name'], args.recon_ROI), 'wb') as fp:
                pickle.dump(recon.average_spec_orig, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open('{}/{}_{}_avgsp_recon.pkl'.format(args.out_dir, recon.parameters['project_name'], args.recon_ROI), 'wb') as fp:
                    pickle.dump(recon.average_spec_recon, fp, protocol=pickle.HIGHEST_PROTOCOL)

    if args.bin_method == 'histogram':

        print('-'*40, 'histogram binning','-'*40)
        m_orig, intens_orig = alignMass(
            recon.parameters['m_HR'], 
            '{}/{}_{}_peak_data_recon'.format(args.out_dir, recon.parameters['project_name'], args.recon_ROI) , 
            0.05)

    if args.bin_method == 'average':

        print('-'*40, 'propagating intensities from the average spectrum','-'*40)
        mz_index = recon.peak_list_recon['mz_index']
        print('now propagating {} number of detected peaks...'.format(len(mz_index)))
        intens_mtx = []
        TIC = []

        for i in tqdm(range(0, recon.SpatialCoef[args.recon_ROI].shape[1], 100)):

            _, _, sp = recon.getSpecFromCoef(recon.SpatialCoef[args.recon_ROI][:,i:i+100], recon.V_hat[args.basis_ROI], recon.parameters['m_HR'], args.mz_range)
            TIC.append(np.sum(sp, 1))
            intens_mtx.append(sp[:, mz_index])

        with open('{}/{}_{}_propagated_recon.pkl'.format(args.out_dir, recon.parameters['project_name'], args.recon_ROI), 'wb') as fp:
            pickle.dump({'mz':recon.peak_list_recon['mz'],'intens_mtx':np.concatenate(intens_mtx),'tic':np.concatenate(TIC),'coordinates':recon.imaginginfo_LR[args.recon_ROI]['coordinates']}, fp, protocol=pickle.HIGHEST_PROTOCOL)



        



        