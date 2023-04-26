import numpy as np
import glob
import pickle

from umap.parametric_umap import ParametricUMAP
import tensorflow as tf

#tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Input, Dense, ReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import  categorical_crossentropy
from tensorflow.keras import Sequential
import tensorflow.keras.layers as L


tf.random.set_seed(1299)

from sklearn.metrics import mean_squared_error
from keras import regularizers
import os
import h5py

import random
random.seed(19)

class VAE_model():
    """
    A class to tain a autoencoder for analysis of the reconstructed 3D FTICR MSI data set. 
    """
    def __init__ (self, data_dir):
        """
        Initialize the data variables where the training and test metadata will be stored. 
        Each dictionary stores the pixel coordinates, total ion current (for normalization),
        and the m/z values. Intensity matrices will be loaded on the fly during training/testing.
        """
        self.train_data_info = {}
        self.data_info = {}
        self.data_dir = data_dir
        self.group_spec_index = []

        if os.path.isfile(data_dir):
            f = h5py.File(data_dir)
            self.group_names = list(f.keys())       # group names for the h5 data file
            self.n_features = len(np.array(f[self.group_names[0]].get('mz_use_idx')))
            self.mz_use = np.array(f[self.group_names[0]].get('mz_common'))
            for group_name in self.group_names:
                self.group_spec_index += [[group_name,i] for i in range(f[group_name].get('intensity matrix').shape[0])]
            f.close()
        else:
            raise Exception('the provided file path does not exist, please double check')

    def read_h5data(self, data_dir, group_name):

        """
        Loading postprocessed .h5 file by groups, which stores the picked peak intensity results in each individual group.
        """
        f = h5py.File(data_dir)
        mz = np.array(f[group_name].get('mz'))
        intens_mtx = np.array(f[group_name].get('intensity matrix'))
        coord = np.array(f[group_name].get('coordinates'))
        tic = np.array(f[group_name].get('tic'))
        mz_use_idx = np.array(f[group_name].get('mz_use_idx'))

        intens_mtx_use = intens_mtx[:,mz_use_idx]

        f.close()

        return mz, mz_use_idx, tic, intens_mtx_use, coord
    
    def get_training_data(self, ratio, group_to_select):

        """
        Prepare training data (sampling from all data or selecting by groups).
        """
        train_group_spec_index = random.sample(self.group_spec_index, int(ratio*len(self.group_spec_index)))
        self.train_data_info['train_group_spec_index'] = train_group_spec_index

        train_data = []
        train_group_spec_index = np.array(train_group_spec_index)

        f = h5py.File(self.data_dir)
        for group_name in group_to_select:
            mz, mz_use_idx, tic, intens_mtx_use, coord = self.read_h5data(self.data_dir, group_name)
            self.data_info[group_name] = {'mz':mz, 'mz_use_idx':mz_use_idx,'tic':tic,'coordinates':coord}
            train_data_group = intens_mtx_use[train_group_spec_index[train_group_spec_index[:,0]==group_name,1].astype(int)]
            train_data.append(train_data_group)
            print('sample {} of pixels for training data from {}'.format(train_data_group.shape[0], group_name))
        f.close()

        train_data = np.concatenate(train_data)

        print('total sampled training data with shape {}'.format(train_data.shape))
        self.train_data = train_data

        return train_data
    
    def paraUMAP_train(self, train_data, latent_dim, save_model=False):

        _,encoder,decoder = self.ae_architecture(nSpecFeatures=self.n_features, latent_dim=latent_dim)

        embedder = ParametricUMAP(encoder=encoder,decoder=decoder, verbose=True ,n_components=latent_dim, n_training_epochs=2)
        embedding = embedder.fit_transform(train_data/train_data.mean(1).reshape(-1,1))

        self.train_embedding_UMAP = embedding
        self.embedder_UMAP = embedder

        if save_model:
            embedder.save('./')
        
    
    def paraUMAP_predict(self):

        print('predicting slices...')

        f = h5py.File(self.data_dir)

        for group_name in self.group_names:
            #_, _, _, intens_mtx_use, _ = self.read_h5data(self.data_dir, group_name)
            mz, mz_use_idx, tic, intens_mtx_use, coord = self.read_h5data(self.data_dir, group_name)
            self.data_info[group_name] = {'mz':mz, 'mz_use_idx':mz_use_idx,'tic':tic,'coordinates':coord}

            print('embedding {}'.format(group_name))
            embedding = self.embedder_UMAP.transform(intens_mtx_use/intens_mtx_use.mean(1).reshape(-1,1))

            self.data_info[group_name]['embeddings'] = embedding

        f.close()

    
    def train_model(self, train_data, epochs, batch_size, intermediate_dim, latent_dim):
        """
        """

        #model, encoder = self.vae_architecture(original_dim = self.n_features, intermediate_dim = intermediate_dim, latent_dim = latent_dim)
        model, encoder, decoder = self.ae_architecture(nSpecFeatures = self.n_features, latent_dim = latent_dim)
        model.summary()

        history = model.fit(self.train_data,self.train_data, epochs=epochs, batch_size=batch_size, shuffle=True)

        self.model = model
        self.encoder = encoder 
        self.train_data_info['loss'] = history.history['loss']
        return model, encoder


    def predict(self):

        print('predicting the training data...')

        f = h5py.File(self.data_dir)

        for group_name in self.group_names:
            #_, _, _, intens_mtx_use, _ = self.read_h5data(self.data_dir, group_name)
            mz, mz_use_idx, tic, intens_mtx_use, coord = self.read_h5data(self.data_dir, group_name)
            self.data_info[group_name] = {'mz':mz, 'mz_use_idx':mz_use_idx,'tic':tic,'coordinates':coord}

            data_pred = self.model.predict(intens_mtx_use)
            embeddings = self.encoder(intens_mtx_use)
            tic_pred = np.sum(data_pred, axis=-1)
            mse = mean_squared_error(intens_mtx_use, data_pred)

            print('{}, MSE:'.format(group_name), mse)
            self.data_info[group_name]['mse'] = mse
            self.data_info[group_name]['embeddings'] = embeddings
            self.data_info[group_name]['tic_pred'] = tic_pred

        f.close()


    def ae_architecture(self, nSpecFeatures, latent_dim):

        input_layer = Input(shape=(nSpecFeatures,))

        ## encoding architecture
        encode_layer1 = Dense(256, activation='relu')(input_layer)
        encode_layer2 = Dense(128, activation='relu')(encode_layer1)
        encode_layer3 = Dense(64, activation='relu')(encode_layer2)
        latent_view   = Dense(latent_dim)(encode_layer3)
        decode_layer1 = Dense(64, activation='relu')(latent_view)
        decode_layer2 = Dense(128, activation='relu')(decode_layer1)
        decode_layer3 = Dense(256, activation='relu')(decode_layer2)

        output_layer  = Dense(nSpecFeatures)(decode_layer3)

        model = Model(input_layer, output_layer)
        encoder = Model(input_layer, latent_view)


        model.compile(optimizer='adam', loss='mse',metrics=['mse'])

        encoded_input = Input(shape=(latent_dim,))
        decode = model.layers[-4](encoded_input)
        decode = model.layers[-3](decode)
        decode = model.layers[-2](decode)
        decode = model.layers[-1](decode)

        decoder = Model(encoded_input, decode)

        return model, encoder, decoder


    def nll(self, y_true, y_pred):
        """ Negative log likelihood (Bernoulli). """

        # keras.losses.binary_crossentropy gives the mean
        # over the last axis. we require the sum
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    
    def vae_architecture(self, original_dim, intermediate_dim, latent_dim):

        x = Input(shape=(original_dim,))
        h = Dense(intermediate_dim, activation='relu')(x)

        z_mu = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

        eps = Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent_dim)))
        z_eps = L.Multiply()([z_sigma, eps])
        z = L.Add()([z_mu, z_eps])

        encoder = Model(x, z_mu)
        decoder = Sequential([
            Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
            Dense(original_dim, activation='sigmoid')
        ])

        x_pred = decoder(z)

        vae = Model(inputs=[x, eps], outputs=x_pred)
        vae.compile(optimizer='rmsprop', loss=self.nll)

        return vae, encoder
    
class KLDivergenceLayer( L.Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

