import numpy as np


class ion_datacube():
    # a class for holding a datacube from an MSI dataset and producing ion images
    def __init__(self, step_size=[]):  # define mandatory parameters (todo- validator to check all present)
        self.xic = []  # 'xtracted ion chromatogram (2d array of intensities(vector of time by mzs))
        self.bounding_box = []  # tuple (x,y,w,h) in co-ordinate space
        self.coords = []  # co-ordiantes of every pixel
        self.mzs = []  # centroids of each xic vector
        self.tol = []  # tolerance window around centroids in m/z
        if step_size == []:
            self.step_size = []
        else:
            self.step_size = step_size
        self.n_im = 0

    def add_xic(self, xic, mz, tol):
        # add an eXtracted Ion Chromatogram to the datacube 
        if len(self.coords) != len(xic):
            raise ValueError(
                'size of co-ordinates to not match xic (coords:{} xic:{})'.format(len(self.coords), len(xic)))
        self.xic.append(xic)  # = np.concatenate((self.xic,xic),axis=1)
        self.mzs = np.concatenate((self.mzs, mz))
        self.tol = np.concatenate((self.tol, tol))
        self.n_im += 1

    def remove_xic(self, mz):
        # remove an xic and related info
        raise NotImplementedError
        index_to_remove = self.mz.index(mz)
        self.n_im -= 1

    def add_coords(self, coords):
        # record spatial co-ordinates for each spectrum
        # coords is a ix3 array with x,y,z coords
        self.coords = coords
        # if len(self.xic)==0:
        #    self.xic = np.zeros((len(coords),0))
        self.calculate_bounding_box()
        self.coord_to_index()

    def calculate_bounding_box(self):
        self.bounding_box = []
        for ii in range(0, 3):
            self.bounding_box.append(np.amin(self.coords[:, ii]))
            self.bounding_box.append(np.amax(self.coords[:, ii]))

    def coord_to_index(self, transform_type='reg_grid', params=()):
        # this function maps coords onto pixel indicies (assuming a grid defined by bounding box and transform type)
        # -implement methods such as interp onto grid spaced over coords
        # -currently treats coords as grid positions, 
        if transform_type == 'reg_grid':
            # data was collected on a grid
            # - coordinates can transformed directly into pixel indiceis
            # - subtract smallest value and divide by x & y step size
            pixel_indices = np.zeros(len(self.coords))
            _coord = np.asarray(self.coords)
            _coord = np.around(_coord, 5)  # correct for numerical precision
            _coord = _coord - np.amin(_coord, axis=0)
            if self.step_size == []:  # no additional info, guess step size in xyz
                self.step_size = np.zeros((3, 1))
                for ii in range(0, 3):
                    self.step_size[ii] = np.mean(np.diff(np.unique(_coord[:, ii])))

            # coordinate to pixels
            #_coord /= np.reshape(self.step_size, (3,))
            _coord_max = np.amax(_coord, axis=0)
            self.nColumns = _coord_max[1] + 1
            self.nRows = _coord_max[0] + 1
            pixel_indices = _coord[:, 0] * self.nColumns + _coord[:, 1]
            pixel_indices = pixel_indices.astype(np.int32)
        else:
            print('transform type not recognised')
            raise ValueError
        self.pixel_indices = pixel_indices

    def xic_to_image(self, xic_index):
        xic = self.xic[xic_index].copy()
        # turn xic into an image
        img = -1 + np.zeros(self.nRows * self.nColumns)
        img[self.pixel_indices] = xic
        img = np.reshape(img, (self.nRows, self.nColumns))
        return img

    def xic_to_image_fixed_size(self, xic_index, out_im_size):
        """
        returns an xic formatted as an image. This function returns an image of a pre-specfied size, cropping or padding equally around the image as required.
        :param xic_index: index of the xic to turn into an image
        :param out_im_size: desired image size
        :return:
        """
        def calc_pad(px1, px2):
            delta = px1-px2
            if delta < 0:
                return 0, 0
            return delta/2, delta/2+delta%2

        def calc_crop(px1, px2):
            delta = px1 - px2
            if delta > 0:
                return 0, -1
            delta = abs(delta)
            return delta/2, -1*delta/2+delta%2

        pad_total_pixels = map(calc_pad, out_im_size, (self.nRows, self.nColumns))
        crop_total_pixels = map(calc_crop, out_im_size, (self.nRows, self.nColumns))

        img = self.xic_to_image(xic_index)
        img=np.asarray(img)
        img = img[crop_total_pixels[0][0]:crop_total_pixels[0][1], crop_total_pixels[1][0]:crop_total_pixels[1][1]]
        img = np.pad(img, ((pad_total_pixels[0][0], pad_total_pixels[0][1]), (pad_total_pixels[1][0], pad_total_pixels[1][1])), 'constant', constant_values=-1)
        return img


    def image_to_xic(self, im):
        """
        takes an image (presumably generated by xic to image) and returns a vector suitable for insertion into the datacube
        :param im:
            numpy 2d array
        :return:
            numpy vector
        """
        im = np.reshape(im, self.nRows * self.nColumns)
        xic = im[self.pixel_indices]
        return xic

    def apply_image_processing(self, smooth_method, **smooth_params):
        """
        Function to apply pre-defined image processing methods to ion_datacube
        #todo: expose parameters in config
        :param ion_datacube:
            object from pyImagingMSpec.ion_datacube already containing images
        :return:
            ion_datacube is updated in place.
            None returned
        """
        from pyImagingMSpec import smoothing
        for ii in range(self.n_im):
            im = self.xic_to_image(ii)
            # todo: for method in smoothing_methods:
            methodToCall = getattr(smoothing, smooth_method)
            im_s = methodToCall(im, **smooth_params)
            self.xic[ii] = self.image_to_xic(im_s)
