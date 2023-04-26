import numpy as np
import matplotlib.pyplot as plt
from utils import *
import pickle
from skimage import io
from skimage.transform import rescale,resize
from skimage import color
import cv2 as cv
from pystackreg import StackReg
from skimage import transform, io, exposure
import SimpleITK as sitk
import h5py
from processing import *
from tqdm import tqdm
from scipy.io import loadmat


class Image():

    def __init__ (self, data_dir, registration_dir, group_order):
        """
        """
        self.data = {}
        self.transform = {}
        self.data_dir = data_dir
        self.registration_dir = registration_dir
        self.group_order = group_order

        if os.path.isfile(data_dir):
            f = h5py.File(data_dir)
            self.mzs = np.array(f[self.group_order[0]].get('mz_common'))
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

    def load_data(self, normalize):

        for i in tqdm(range(len(self.group_order))):
            _,_,_,intens_mtx,coord = self.read_h5data(self.data_dir, self.group_order[i])
            if normalize:
                intens_mtx = intens_mtx/intens_mtx.sum(1).reshape(-1,1)
            self.data[self.group_order[i]] = {'intens_mtx':intens_mtx,'coordinates':coord}


    def load_transform(self,file_dir):

        idx = 0
        for group in self.group_order:
            spec_transform = loadmat(file_dir+'/spec_idx_imgs_transformed_'+group)['spec_idx_img_transformed_new']
            self.transform[group] = spec_transform
            idx+=1


    def get_3DImages(self, mz_index, background):

        images = []
        for group in self.group_order:
            img = self.data[group]['intens_mtx'][:,mz_index]
            coord = self.data[group]['coordinates']
            ion_image = IonImg(img, coord, background, False) 
            images.append(ion_image)

        return images
    


    def get_3DImages_transform(self, mz_index, background):
        
        images = []
        for group in self.group_order:
            img = self.data[group]['intens_mtx'][:,mz_index]
            coord = self.data[group]['coordinates']
            ion_image = self.IonImg_transform(img, self.transform[group],background)

            images.append(ion_image)

        return images



    def transform_msi_image(self, input_img, moving_shape, fixed_image, affine_matrix, rigid_transform, scaling_factor):

        #perform affine transformation
        tform = transform.AffineTransform(matrix=affine_matrix)
        input_img_resized = resize(input_img,moving_shape,order=1)
        input_img_affine =  transform.warp(input_img_resized, tform)

        #perform rigid body
        input_img_affine_rigid = sitk.Resample(sitk.GetImageFromArray(input_img_affine), fixed_image, rigid_transform)
        input_img_affine_rigid = sitk.GetArrayFromImage(input_img_affine_rigid)

        #rescaling
        input_img_affine_rigid_rescaled = rescale(input_img_affine_rigid, [scaling_factor,scaling_factor])

        return input_img_affine_rigid_rescaled


    def IonImg_transform(self, data, spec_idx_transform, background):

        if background:
            ion_img_transform = np.zeros(spec_idx_transform.size)
        else:
            ion_img_transform = np.empty(spec_idx_transform.size)
            ion_img_transform[:] = np.nan
        img_shape = spec_idx_transform.shape
        spec_idx_transform_flatten = spec_idx_transform.flatten()
        ion_img_transform[spec_idx_transform_flatten!=0] = data[spec_idx_transform_flatten[spec_idx_transform_flatten!=0]-1]
        ion_img_transform = ion_img_transform.reshape(img_shape)

        return ion_img_transform


def overlay_images(imgs, equalize=False, aggregator=np.mean):

    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = np.stack(imgs, axis=0)

    return aggregator(imgs, axis=0)

def composite_images(imgs, equalize=False, aggregator=np.mean):

    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]
    imgs = [img / img.max() for img in imgs]

    if len(imgs) < 3:
        imgs += [np.zeros(shape=imgs[0].shape)] * (3-len(imgs))
    imgs = np.dstack(imgs)

    return imgs