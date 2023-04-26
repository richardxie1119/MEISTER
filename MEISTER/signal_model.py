from tensorflow import keras 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization,ReLU
#from tensorflow.keras.layers import Conv1D, Reshape, MaxPooling1D,Conv1DTranspose, UpSampling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model
import numpy as np
import tqdm
import sys
sys.path.append('../')
from utils import *
from processing import *
import random


def decodeSignal(encoded, decoder):

    decoded = decoder(encoded)

    return np.array(decoded)

def decodeSpectra(path,encoded, decoder, mz_range):
    
    decoded = decodeSignal(encoded, decoder)
    #mz, sp = fid2spec(decoded, m, mz_range)
    sp = []
    for i in range(len(decoded)):
        D = proc_solarix_imaging(path,decoded[i])
        mz, sp_ = fid2spec_solarix(D, mz_range)
        sp.append(sp_)

    return mz, np.stack(sp), D

def encodeSignal(ser_file_path,fid_length,signal_size,scan_index,encoder):
    
    ENCODED = []
    for i in tqdm(range(0, len(scan_index), 128)):
        fid_loaded = []
        for j in range(i,i+128):
            if j < len(scan_index):
                fid_loaded.append(loadBrukerFIDs(ser_file_path[j], fid_length, signal_size, scan_index[j]))
        fid_loaded = np.concatenate(fid_loaded)
        
        encoded = encoder(fid_loaded)
        ENCODED.append(np.array(encoded))

    return np.concatenate(ENCODED,axis=0)


def predictEncoded(ser_file_path,fid_length,signal_size,scan_index,model):
    
    ENCODED = []
    for i in tqdm(range(0, scan_index.size, 128)):
        fid_loaded = loadBrukerFIDs(ser_file_path, 
                                        fid_length, 
                                        signal_size, 
                                        scan_index[i:i+128])
        
        encoded = model.predict(fid_loaded)
        ENCODED.append(np.array(encoded))

    return np.concatenate(ENCODED,axis=0)


def decodeAvgSpectraPeaks(encoded, decoder, mz_range, sample_perc=0.2, mz_index=[]):

    random.seed(19)
    rows_id = random.sample(range(0, encoded.shape[0]), int(encoded.shape[0]*sample_perc))
    encoded_sampled = encoded[rows_id]

    peak_data = {}
    mz, sp, _ = decodeSpectra('./temp_fid', encoded_sampled[0].reshape(1,-1), decoder, mz_range)
    average_sp = np.zeros(sp.shape)

    
    for i in tqdm(range(0, encoded_sampled.shape[0], 128)):
        mz, sp_decoded, _ = decodeSpectra('./temp_fid', encoded_sampled[i:i+128], decoder, mz_range)
        if len(mz_index) != 0:
            peak_data[idx] = {'mz':mz[mz_index],'intensity':sp_decoded[:, mz_index]}
            average_sp += np.sum(sp_decoded,0)
        else:
            average_sp += np.sum(sp_decoded,0)

    average_sp = average_sp[0]/encoded_sampled.shape[0]
    print(average_sp.shape)
    

    if len(mz_index) == 0:
        
        peak_list_avg = peak_detection(mz, average_sp, prominence = mad(average_sp)*10, threshold = mad(average_sp)*10)

        print('propogating encoded intensities of {} peaks...'.format(peak_list_avg['mz_index'].size))
        idx = 1
        for i in tqdm(range(0, encoded.shape[0], 128)):
            mz, sp_decoded, _ = decodeSpectra('./temp_fid', encoded[i:i+128], decoder, mz_range)
            for j in range(len(sp_decoded)):
                peak_data[idx] = {'mz':mz[peak_list_avg['mz_index']],'intensity':sp_decoded[j, peak_list_avg['mz_index']]}
                idx+=1
    else:
        peak_list_avg = {'mz':mz[mz_index], 'intensity':average_sp[mz_index],'mz_index':mz_index}
        
    return average_sp, peak_list_avg, peak_data


def decodeSpectraPeaks(encoded, decoder, mz_range):
    #d_buffer = load_buffer('./temp_fid')
    mz, sp, _ = decodeSpectra('./temp_fid', encoded[0].reshape(1,-1), decoder, mz_range)
    average_sp = np.zeros(sp.shape)

    peak_data = {}
    idx = 1
    for i in tqdm(range(0, encoded.shape[0], 128)):
        #d_buffer = load_buffer('./temp_fid')
        mz,sp_decoded,d = decodeSpectra('./temp_fid', encoded[i:i+128], decoder, mz_range)
        average_sp += np.sum(sp_decoded,0)
        
        for j in range(len(sp_decoded)):
            peak_data[idx] = peak_detection(mz, sp_decoded[j], d.robust_stats()[1]*10,d.robust_stats()[1]*10)
            idx += 1
    average_sp = average_sp[0]/encoded.shape[0]

    return [average_sp,mz], peak_data


def ae_architecture(nSpecFeatures, latent_dim):

    input_layer = Input(shape=(nSpecFeatures,))

    ## encoding architecture
    encode_layer1 = Dense(512, activation='relu')(input_layer)
    #encode_layer1 = BatchNormalization()(encode_layer1)
    encode_layer2 = Dense(256, activation='relu')(encode_layer1)
    #encode_layer2 = BatchNormalization()(encode_layer2)
    encode_layer3 = Dense(128, activation='relu')(encode_layer2)
    #encode_layer3 = BatchNormalization()(encode_layer3)
    encode_layer4 = Dense(64, activation='relu')(encode_layer3)
    #encode_layer4 = BatchNormalization()(encode_layer4)
    latent_view   = Dense(latent_dim)(encode_layer4)
    #latent_view_bn = BatchNormalization()(latent_view)
    #latent_view_bn = ReLU()(latent_view)

    ## decoding architecture
    decode_layer1 = Dense(64, activation='relu')(latent_view)
    #decode_layer1 = BatchNormalization()(decode_layer1)
    decode_layer2 = Dense(128, activation='relu')(decode_layer1)
    #decode_layer2 = BatchNormalization()(decode_layer2)
    decode_layer3 = Dense(256, activation='relu')(decode_layer2)
    #decode_layer3 = BatchNormalization()(decode_layer3)
    decode_layer4 = Dense(512, activation='relu')(decode_layer3)
    #decode_layer4 = BatchNormalization()(decode_layer4)

    output_layer  = Dense(nSpecFeatures)(decode_layer4)

    model = Model(input_layer, output_layer)
    encoder = Model(input_layer, latent_view)


    model.compile(optimizer='adam', loss='mse',metrics=['mse'])

    encoded_input = Input(shape=(latent_dim,))
    decode = model.layers[-5](encoded_input)
    decode = model.layers[-4](decode)
    decode = model.layers[-3](decode)
    decode = model.layers[-2](decode)
    decode = model.layers[-1](decode)
    decoder = Model(encoded_input, decode)

    return model, encoder, decoder


def lp_architecture(nSpecFeatures, latent_dim):

    input_layer = Input(shape=(nSpecFeatures,))

    ## encoding architecture
    encode_layer1 = Dense(512, activation='relu')(input_layer)
    #encode_layer1 = BatchNormalization()(encode_layer1)
    
    encode_layer2 = Dense(256, activation='relu')(encode_layer1)
    #encode_layer2 = BatchNormalization()(encode_layer2)
    
    encode_layer3 = Dense(128, activation='relu')(encode_layer2)
    #encode_layer3 = BatchNormalization()(encode_layer3)
    
    encode_layer4 = Dense(64, activation='relu')(encode_layer3)
    #encode_layer4 = BatchNormalization()(encode_layer4)

    latent_view   = Dense(latent_dim)(encode_layer4)
    
    model = Model(input_layer, latent_view)
    
    model.compile(optimizer='adam', loss='mse',metrics=['mse'])
    
    return model