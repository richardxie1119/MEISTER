import numpy as np
import os
from os import path
from glob import glob
import sys
import h5py

from tensorflow.python.ops.gen_batch_ops import batch
sys.path.append('../')
from utils import *
from processing import *
from signal_model import *
import subspaceMSI as subspace

from tensorflow import keras 
import tensorflow as tf

import argparse


def trainDecoder(batch_size, epochs, ser_file_path, fid_length, signal_size, scan_index, model):
    
    optimizer = keras.optimizers.Adam()
    mse_loss = keras.losses.MeanSquaredError()
    
    losses = []
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        step = 0
        # Iterate over the batches of the dataset.
        data_idx = np.arange(0,len(scan_index))
        np.random.shuffle(data_idx)
        
        loss = []
        for i in tqdm(range(0, len(data_idx), batch_size)):
            step += 1
            # fid_idx = scan_index[i:i+batch_size]
            # fid_loaded = loadBrukerFIDs(ser_file_path, 
            #                             fid_length, 
            #                             signal_size,
            #                             fid_idx)

            fid_loaded = []
            for j in range(i,i+batch_size):
                if j < len(data_idx):
                    fid_loaded.append(loadBrukerFIDs(ser_file_path[data_idx[j]], fid_length, signal_size, scan_index[data_idx[j]]))
            fid_loaded = np.concatenate(fid_loaded)

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                logits = model(fid_loaded, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = mse_loss(fid_loaded, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 50 batches.
            if step % 50 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
            loss.append(loss_value)
        losses.append(np.mean(loss))
    return losses


def trainDecoder_singlecell(batch_size, epochs, names_train , data_file_path, fid_length, signal_size, model):
    
    optimizer = keras.optimizers.Adam()
    mse_loss = keras.losses.MeanSquaredError()
    
    losses = []
    h5 = h5py.File(data_file_path)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        step = 0
        # Iterate over the batches of the dataset.
        data_idx = np.arange(0,len(names_train))
        np.random.shuffle(data_idx)
        
        loss = []
        for i in tqdm(range(0, len(data_idx), batch_size)):
            step += 1

            fid_loaded = []
            for j in range(i,i+batch_size):
                if j < len(data_idx):
                    fid_loaded.append(np.array(h5[names_train[data_idx[j]]].get('transient'))[0,:fid_length])
            fid_loaded = np.stack(fid_loaded)

            with tf.GradientTape() as tape:

                logits = model(fid_loaded, training=True) 

                loss_value = mse_loss(fid_loaded, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 15 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
            loss.append(loss_value)
        losses.append(np.mean(loss))
    h5.close()
    return losses

def trainRegressor(batch_size, epochs, encoded_train, ser_file_path, fid_length, signal_size, scan_index, model):
    
    optimizer = keras.optimizers.Adam()
    mse_loss = keras.losses.MeanSquaredError()
    
    losses = []
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        step = 0
        # Iterate over the batches of the dataset.
        data_idx = np.arange(0,len(scan_index))
        np.random.shuffle(data_idx)
        
        loss = []
        for i in tqdm(range(0, len(data_idx), batch_size)):
            step += 1

            fid_loaded = []
            for j in range(i,i+batch_size):
                if j < len(data_idx):
                    fid_loaded.append(loadBrukerFIDs(ser_file_path[data_idx[j]], fid_length, signal_size, scan_index[data_idx[j]]))
            fid_loaded = np.concatenate(fid_loaded)

            with tf.GradientTape() as tape:

                logits = model(fid_loaded, training=True)

                loss_value = mse_loss(encoded_train[data_idx[i:i+batch_size]], logits)


            grads = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 50 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
            loss.append(loss_value)
        losses.append(np.mean(loss))
    return losses

def trainRegressor_singlecell(batch_size, epochs, encoded_train, names_train, data_file_path, fid_length, signal_size, model):
    
    optimizer = keras.optimizers.Adam()
    mse_loss = keras.losses.MeanSquaredError()
    
    losses = []
    fid_loaded_train = np.zeros((len(names_train), signal_size))

    print('loading training data for the regressor...')
    for i in tqdm(range(len(names_train))):
        with h5py.File(data_file_path,'r') as h5:
            fid_loaded_train[i] = np.array(h5[names_train[i]].get('transient'))[:,:signal_size]

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        step = 0
        # Iterate over the batches of the dataset.
        data_idx = np.arange(0,len(names_train))
        np.random.shuffle(data_idx)
        
        loss = []
        for i in tqdm(range(0, len(data_idx), batch_size)):
            step += 1

            fid_loaded_batch = []
            for j in range(i,i+batch_size):
                if j < len(data_idx):
                    fid_loaded_batch.append(fid_loaded_train[data_idx[j]])

            fid_loaded_batch = np.stack(fid_loaded_batch)

            with tf.GradientTape() as tape:

                logits = model(fid_loaded_batch, training=True) 

                loss_value = mse_loss(encoded_train[data_idx[i:i+batch_size]], logits)

            grads = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 50 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
            loss.append(loss_value)
        losses.append(np.mean(loss))
    h5.close()
    return losses


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



