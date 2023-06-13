r"""DeepMSI reconstruction of the high-res MSI transients by predicting the encodings of
the high-res data. Autoencoder and the encoding regressor are trained on existing high-res
dataset from a few tissue sections of a 3D volume. Encodings can then be predicted from 
low-res data (short transients) of the remaining tissue sections to enable highly accelerated
data acquisition and improved SNRs and image quality. 
"""
import sys
sys.path.append('MEISTER')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from utils import *
from processing import *
import subspaceMSI
from signal_model import *

import argparse
import logging
import pickle
from tensorflow import keras 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'DeepMSI reconstruction of FT-ICR mass spectrometry imaging data.'
        )
    parser.add_argument(
        '--path_file', required = True, type = str,
        help='Path to the .json file that specifies the raw .ser and imaginginfo files.'
        )
    parser.add_argument(
        '--decoder_dir', type = str,
        help='A directory to the decoder.'
        )
    parser.add_argument(
        '--regressor_dir', type = str,
        help='A directory to the regressor.'
        )
    parser.add_argument(
        '--out_dir', default = './processed_data', type = str,
        help='An output directory to store the processed results.'
        )
    parser.add_argument(
        '--mz_index', default = 'none', type = str,
        help='mz index for intensity propogation of the peaks.'
        )
    parser.add_argument(
        '--embedding', default = 'False', type = str,
        help='mz index for intensity propogation of the peaks.'
        )
    parser.add_argument(
        '--recon_ROI', required = True, type = str,
        help='The ROI(region of interest) pointer to the imaginginfo file to reconstruct long transients from the defined basis.'
        )
    parser.add_argument(
        '--mz_range', required = True, type = int, nargs = '+',
        help='Two values that defines the lower and upper bound of m/z range for analyzing spectra.'
        )
    parser.add_argument(
        '--if_process_raw', nargs = '?', const = 'False', default = 'False', type = str, choices = ['True','False'],
        help='If evaluate on the original high resolution data, which involves truncating the measured long transients. If false, the long transients data are not given.'
        )
    parser.add_argument(
        '--if_simu', nargs = '?', const = 'False', default = 'False', type = str, choices = ['True','False'],
        help='If evaluate on the original high resolution data, which involves truncating the measured long transients. If false, the long transients data are not given.'
        )
    
    args = parser.parse_args()

    print('-'*40, 'script parameters','-'*40)
    print(args)

    if args.if_simu == 'True':
        if_simu = True
    else:
        if_simu = False

    if args.embedding == 'True':
        embedding = True
    else:
        embedding = False

    recon = subspaceMSI.Subspace(out_dir=args.out_dir)
    recon.experimentInfo(args.path_file, if_simu = if_simu, sampling_pattern = None)
    
    m = recon.parameters['m_HR']
    t = recon.parameters['t_HR']
    t_LR = recon.parameters['t_LR']
    ser_file_path = recon.ser_file_path_LR
    fid_length = t.size
    signal_size = t_LR.size
    coord = recon.imaginginfo_LR[args.recon_ROI]['coordinates']
    scan_index = recon.imaginginfo_LR[args.recon_ROI]['scan_index']

    if args.if_process_raw == 'True':
        process_raw = True
    else:
        process_raw = False

    if args.mz_index == 'none':
        mz_index = []

    decoder = keras.models.load_model(args.decoder_dir)
    model_lp = keras.models.load_model(args.regressor_dir)

    if not embedding:
        if process_raw:
            print('-'*40, 'processing original raw','-'*40)
            recon.ProcSerFile(ROI = args.recon_ROI, mz_range = args.mz_range, return_peaks = 'average')
            with open('{}/{}_{}_avgsp_orig.pkl'.format(args.out_dir, recon.parameters['project_name'], args.recon_ROI), 'wb') as fp:
                    pickle.dump(recon.average_spec_orig, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    

        print('-'*40, 'predicting latent space encodings','-'*40)

        if if_simu:
            encoded_pred = predictEncoded(ser_file_path,fid_length,signal_size,scan_index,model_lp)
        else:
            encoded_pred = predictEncoded(ser_file_path,signal_size,signal_size,scan_index,model_lp)

        with open('{}/{}_{}_encoded_pred.pkl'.format(args.out_dir, recon.parameters['project_name'], args.recon_ROI), 'wb') as f:
            pickle.dump(encoded_pred, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    else:
        with open('{}/{}_{}_encoded_pred.pkl'.format(args.out_dir, recon.parameters['project_name'], args.recon_ROI), 'rb') as f:
            encoded_pred = pickle.load(f)


    print('-'*40, 'decoding high resolution peak data','-'*40)
    average_sp, peak_list_avg, peak_data_pred_decoded = decodeAvgSpectraPeaks(encoded_pred, decoder, args.mz_range)
    #average_sp, peak_data_pred_decoded = decodeSpectraPeaks(encoded_pred, decoder, args.mz_range)

    with open('{}/{}_{}_avgsp_decoded.pkl'.format(args.out_dir, recon.parameters['project_name'], args.recon_ROI), 'wb') as fp:
            pickle.dump(average_sp, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('{}/{}_{}_peak_list_decoded.pkl'.format(args.out_dir, recon.parameters['project_name'], args.recon_ROI), 'wb') as fp:
            pickle.dump(peak_list_avg, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('{}/{}_{}_propagated_decoded.pkl'.format(args.out_dir, recon.parameters['project_name'], args.recon_ROI), 'wb') as fp:
        pickle.dump({'mz':peak_list_avg['mz'],'peak_data':peak_data_pred_decoded,'coordinates':recon.imaginginfo_LR[args.recon_ROI]['coordinates']}, fp, protocol=pickle.HIGHEST_PROTOCOL)