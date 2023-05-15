
import numpy as np
import os
from os import path
from glob import glob
import sys
import h5py

from tensorflow.python.ops.gen_batch_ops import batch
sys.path.append('MEISTER')
from utils import *
from processing import *
from signal_model import *
from train import *
import subspaceMSI as subspace

from tensorflow import keras 
import tensorflow as tf

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'A subspace approach to reconstruct FT-ICR mass spectrometry imaging data.'
        )
    parser.add_argument(
        '--train_ROI', required = True, type = str, nargs='+',
        help='ROI header for the training data.'
        )
    parser.add_argument(
        '--path_file', required = True, type = str, nargs='+',
        help='Path to the .json file that specifies the raw .ser and imaginginfo files.'
        )
    parser.add_argument(
        '--out_dir', default = './processed_data', type = str,
        help='An output directory to store the processed results.'
        )
    parser.add_argument(
        '--model_dir', default = './saved_model', type = str,
        help='An output directory to store the models.'
        )
    parser.add_argument(
        '--batch_size', nargs = '?', const = 128, default = 128, type = int,
        help='Batch size for training.'
        )
    parser.add_argument(
        '--epochs_encoder', nargs = '?', const = 10, default = 10, type = int,
        help='Epochs for training.'
        )
    parser.add_argument(
        '--epochs_regressor', nargs = '?', const = 30, default = 30, type = int,
        help='Epochs for training.'
        )
    parser.add_argument(
        '--latent_dim', nargs = '?', const = 15, default = 15, type = int,
        help='Latent dimensions.'
        )
    
    args = parser.parse_args()

    print('-'*40, 'script parameters','-'*40)
    print(args)
    
    recon = subspace.Subspace(out_dir = args.out_dir)
    ser_file_path_train = []
    scan_index_train = []

    if isinstance(args.path_file, list): 
        for i in range(len(args.path_file)):
            recon.experimentInfo(args.path_file[i], if_simu = True, sampling_pattern = None)
            scan_index = list(recon.imaginginfo_BASIS[args.train_ROI[i]]['scan_index'].copy())
            scan_index_train += scan_index
            ser_file_path_train += [recon.ser_file_path_BASIS]*len(scan_index)

    fid_length = recon.parameters['fid_length_HR']
    signal_size = recon.parameters['fid_length_LR']


    print('-'*40, 'Getting model architectures','-'*40)
    model, encoder,decoder = ae_architecture(fid_length, args.latent_dim)
    model_lp = lp_architecture(signal_size, args.latent_dim)

    print('-'*40, 'Autoencoder architecture','-'*40)
    model.summary()

    ##train the signal autoencoder
    print('training the signal autoencoder...')
    scan_index_train_ = scan_index_train.copy()
    ser_file_path_train_ = ser_file_path_train.copy()

    losses_ae = trainDecoder(args.batch_size,args.epochs_encoder,ser_file_path_train_,fid_length,fid_length,scan_index_train_,model)
    encoder.save('{}/{}_encoder'.format(args.model_dir,recon.parameters['project_name']))
    decoder.save('{}/{}_decoder'.format(args.model_dir,recon.parameters['project_name']))
    with open('{}/{}_ae_trainloss.pkl'.format(args.out_dir,recon.parameters['project_name']), 'wb') as f:
        pickle.dump(losses_ae, f, protocol=pickle.HIGHEST_PROTOCOL)

    encoder = keras.models.load_model('{}/{}_encoder'.format(args.model_dir,recon.parameters['project_name']))
    decoder = keras.models.load_model('{}/{}_decoder'.format(args.model_dir,recon.parameters['project_name']))

    print('encoding the signals...')
    #scan_index = recon.imaginginfo_BASIS[args.train_ROI]['scan_index'].copy()
    encoded_HR = encodeSignal(ser_file_path_train,fid_length,'all',scan_index_train,encoder)
    del model, encoder, decoder

    ##train the signal regressor
    print('-'*40, 'Regressor architecture','-'*40)
    model_lp.summary()

    print('training the signal regressor...')
    scan_index_train_ = scan_index_train.copy()
    ser_file_path_train_ = ser_file_path_train.copy()

    losses_lp = trainRegressor(args.batch_size,args.epochs_regressor,encoded_HR,ser_file_path_train_,fid_length,signal_size,scan_index_train_,model_lp)
    model_lp.save('{}/{}_regressor'.format(args.model_dir,recon.parameters['project_name']))
    with open('{}/{}_regressor_trainloss.pkl'.format(args.out_dir,recon.parameters['project_name']), 'wb') as f:
        pickle.dump(losses_lp, f, protocol=pickle.HIGHEST_PROTOCOL)