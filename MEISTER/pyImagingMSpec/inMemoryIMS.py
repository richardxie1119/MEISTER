import os
import numpy as np
import bisect
import sys
#import matplotlib.pyplot as plt

# import our MS libraries
from pyMSpec.mass_spectrum import mass_spectrum
from pyImagingMSpec.ion_datacube import ion_datacube

class inMemoryIMS():
    def __init__(self, filename, min_mz=0., max_mz=np.inf, min_int=0., index_range=[],cache_spectra=True,do_summary=True,norm='none', norm_args={}, spectrum_type='centroids'):
        file_size = os.path.getsize(filename)
        self.load_file(filename, min_mz, max_mz, min_int, index_range=index_range,cache_spectra=cache_spectra,do_summary=do_summary,norm=norm, norm_args=norm_args, spectrum_type=spectrum_type)

    def load_file(self, filename, min_mz=0, max_mz=np.inf, min_int=0, index_range=[],cache_spectra=True,do_summary=True,norm=[], norm_args={}, spectrum_type='centroids'):
        # parse file to get required parameters
        # can use thin hdf5 wrapper for getting data from file
        self.file_dir, self.filename = os.path.split(filename)
        self.filename, self.file_type = os.path.splitext(self.filename)
        self.file_type = self.file_type.lower()
        self.norm=norm.lower()
        self.norm_args = norm_args
        if self.file_type == '.hdf5':
            import h5py
            self.hdf = h5py.File(filename, 'r')  # Readonly, fie must exist
            if index_range == []:
                self.index_list = map(int, self.hdf['/spectral_data'].keys())
            else:
                self.index_list = index_range
        elif self.file_type == '.imzml':
            from pyimzml.ImzMLParser import ImzMLParser
            self.imzml = ImzMLParser(filename)
            self.index_list=range(0,len(self.imzml.coordinates))
        else:
            raise TypeError('File type not recogised: {}'.format(self.file_type))
        self.max_index = max(self.index_list)
        self.coords = self.get_coords()
        step_size = self.get_step_size()
        cube = ion_datacube(step_size=step_size)
        cube.add_coords(self.coords)
        self.cube_pixel_indices = cube.pixel_indices
        self.cube_n_row, self.cube_n_col = cube.nRows, cube.nColumns
        self.histogram_mz_axis = {}
        self.mz_min = 9999999999999.
        self.mz_max = 0.
        self.spectrum_type = spectrum_type #todo this should be read for imzml files, not coded as an input
        if any([cache_spectra,do_summary]) == True:
            # load data into memory
            self.mz_list = []
            self.count_list = []
            self.idx_list = []
            if do_summary:
                self.mic=np.zeros((len(self.index_list),1))
                self.tic=np.zeros((len(self.index_list),1))
            for ii in self.index_list:
                # load spectrum, keep values gt0 (shouldn't be here anyway)
                this_spectrum = self.get_spectrum(ii)
                mzs, counts = this_spectrum.get_spectrum(source=spectrum_type)
                if len(mzs) != len(counts):
                    raise TypeError('length of mzs ({}) not equal to counts ({})'.format(len(mzs), len(counts)))
                # Enforce data limits
                valid = np.where((mzs > min_mz) & (mzs < max_mz) & (counts > min_int))
                counts = counts[valid]
                mzs = mzs[valid]
                # record min/max

                if not len(mzs) == 0:
                    if mzs[0]<self.mz_min:
                        self.mz_min = mzs[0]
                    if mzs[-1]>self.mz_max:
                        self.mz_max = mzs[-1]
                     #record summary values
                    if do_summary:
                        self.tic[ii]=sum(counts)
                        self.mic[ii]=max(counts)
                # append ever-growing lists (should probably be preallocated or piped to disk and re-loaded)
                if cache_spectra:
                    self.mz_list.append(mzs)
                    self.count_list.append(counts)
                    self.idx_list.append(np.ones(len(mzs), dtype=int) * ii)

            print('loaded spectra')
            if cache_spectra:
                self.mz_list = np.concatenate(self.mz_list)
                self.count_list = np.concatenate(self.count_list)
                self.idx_list = np.concatenate(self.idx_list)
                # sort by mz for fast image formation
                mz_order = np.argsort(self.mz_list)
                self.mz_list = self.mz_list[mz_order]
                self.count_list = self.count_list[mz_order]
                self.idx_list = self.idx_list[mz_order]
                # split binary searches into two stages for better locality
                self.window_size = 1024
                self.mz_sublist = self.mz_list[::self.window_size].copy()
        print('file loaded')


    def get_step_size(self):
        if self.file_type == '.imzml':
            return [1,1,1]
        else:
            return []


    def get_coords(self):
        # wrapper for redirecting requests to correct parser
        if self.file_type == '.imzml':
            coords = self.get_coords_imzml()
            coords[:,[0, 1]] = coords[:,[1, 0]]
        elif self.file_type == '.hdf5':
            coords = self.get_coords_hdf5()
        return coords


    def get_coords_imzml(self):# get real world coordinates
        print('TODO: convert indices into real world coordinates')
        coords = np.asarray(self.imzml.coordinates)
        if len(self.imzml.coordinates[0]) == 2: #2D - append zero z-coord
            coords = np.concatenate((coords,np.zeros((len(coords),1))),axis=1)
        return coords


    def get_coords_hdf5(self):
        coords = np.zeros((len(self.index_list), 3))
        for k in self.index_list:
            coords[k, :] = self.hdf['/spectral_data/' + str(k) + '/coordinates/']
        return coords


    def get_spectrum(self,index):
        # wrapper for redirecting requests to correct parser
        if self.file_type == '.imzml':
            this_spectrum = self.get_spectrum_imzml(index)
        elif self.file_type == '.hdf5':
            this_spectrum = self.get_spectrum_hdf5(index)
        if self.norm != []:
            this_spectrum.normalise_spectrum(method=self.norm, method_args=self.norm_args)
            #mzs,counts = this_spectrum.get_spectrum(source="centroids")
            #if self.norm == 'TIC':
            #    counts = counts / np.sum(counts)
            #elif self.norm == 'RMS':
            #    counts = counts / np.sqrt(np.mean(np.square(counts)))
            #elif self.norm == 'MAD':
            #    counts = counts/np.median(np.absolute(counts - np.mean(counts)))
            #this_spectrum.add_centroids(mzs,counts)
        return this_spectrum


    def get_spectrum_imzml(self,index):
        mzs, intensities = self.imzml.getspectrum(index)
        ## temp hack -> assume centroided
        this_spectrum = mass_spectrum()
        if self.spectrum_type == 'centroids':
            this_spectrum.add_centroids(mzs,intensities)
        else:
            this_spectrum.add_spectrum(mzs,intensities)
        return this_spectrum

    def get_spectrum_hdf5(self, index):
        import h5py
        this_spectrum = mass_spectrum()
        tmp_str = '/spectral_data/%d' % (index)
        try:
            this_spectrum.add_spectrum(self.hdf[tmp_str + '/mzs/'], self.hdf[tmp_str + '/intensities/'])
            got_spectrum = True
        except KeyError:
            got_spectrum = False
        try:
            this_spectrum.add_centroids(self.hdf[tmp_str + '/centroid_mzs/'],
                                        self.hdf[tmp_str + '/centroid_intensities/'])
            got_centroids = True
        except KeyError:
            got_centroids = False
        if not any([got_spectrum, got_centroids]):
            raise ValueError('No spectral data found in index {}'.format(index))
        return this_spectrum

    def empty_datacube(self):
        data_out = ion_datacube()
        # add precomputed pixel indices
        data_out.coords = self.coords
        data_out.pixel_indices = self.cube_pixel_indices
        data_out.nRows = self.cube_n_row
        data_out.nColumns = self.cube_n_col
        return data_out

    def get_ion_image(self, mzs, tols, tol_type='ppm'):
        try:
            len(mzs)
        except TypeError as e:
            mzs = [mzs,]
        try:
            len(tols)
        except TypeError as e:
            tols = [tols, ]
        mzs = np.asarray(mzs)
        tols = np.asarray(tols)
        data_out = self.empty_datacube()
        def search_sort(mzs,tols):
            data_out = blank_dataout()
            idx_left = np.searchsorted(self.mz_list, mzs - tols, 'l')
            idx_right = np.searchsorted(self.mz_list, mzs + tols, 'r')
            for mz, tol, il, ir in zip(mzs, tols, idx_left, idx_right):
                if any((mz<self.mz_list[0],mz>self.mz_list[-1])):
                    data_out.add_xic(np.zeros(np.shape(self.cube_pixel_indices)), [mz], [tol])
                    continue
                # slice list for code clarity
                mz_vect=self.mz_list[il:ir]
                idx_vect = self.idx_list[il:ir]
                count_vect = self.count_list[il:ir]
                # bin vectors
                ion_vect = np.bincount(idx_vect, weights=count_vect, minlength=self.max_index + 1)
                data_out.add_xic(ion_vect, [mz], [tol])
            return data_out
        def search_bisect(mzs,tols):
            data_out = blank_dataout()
            for mz,tol in zip(mzs,tols):
                if any((mz<self.mz_list[0],mz>self.mz_list[-1])):
                    data_out.add_xic(np.zeros(np.shape(self.cube_pixel_indices)), [mz], [tol])
                    continue
                mz_upper = mz + tol
                mz_lower = mz - tol
                il = bisect.bisect_left(self.mz_list,mz_lower)
                ir = bisect.bisect_right(self.mz_list,mz_upper)
                # slice list for code clarity
                mz_vect=self.mz_list[il:ir]
                idx_vect = self.idx_list[il:ir]
                count_vect = self.count_list[il:ir]
                # bin vectors
                ion_vect = np.bincount(idx_vect, weights=count_vect, minlength=self.max_index + 1)
                data_out.add_xic(ion_vect, [mz], [tol])
            return data_out
        if len(tols) == 1:
            tols = tols*np.ones(np.shape(mzs))
        if type(mzs) not in (np.ndarray, list):
            mzs = np.asarray([mzs, ])
        if tol_type == 'ppm':
            tols = tols * mzs / 1e6  # to m/z
        # Fast search for insertion point of mz in self.mz_list
        # First stage is looking for windows using the sublist
        idx_left = np.searchsorted(self.mz_sublist, mzs - tols, 'l')
        idx_right = np.searchsorted(self.mz_sublist, mzs + tols, 'r')
        for mz, tol, il, ir in zip(mzs, tols, idx_left, idx_right):
            l = max(il - 1, 0) * self.window_size
            r = ir * self.window_size
            # Second stage is binary search within the windows
            il = l + np.searchsorted(self.mz_list[l:r], mz - tol, 'l')
            ir = l + np.searchsorted(self.mz_list[l:r], mz + tol, 'r')
            # slice list for code clarity
            mz_vect=self.mz_list[il:ir]
            idx_vect = self.idx_list[il:ir]
            count_vect = self.count_list[il:ir]
            # bin vectors
            ion_vect = np.bincount(idx_vect, weights=count_vect, minlength=self.max_index + 1)
            data_out.add_xic(ion_vect, [mz], [tol])
        return data_out
        # Form histogram axis

    def generate_histogram_axis(self, ppm=1.):
        ppm_mult = ppm * 1e-6
        mz_current = self.mz_min
        mz_list = [mz_current,]
        while mz_current <= self.mz_max:
            mz_current = mz_current + mz_current * ppm_mult
            mz_list.append(mz_current)
        self.histogram_mz_axis[ppm] = mz_list

    def get_histogram_axis(self, ppm=1.):
        try:
            mz_axis = self.histogram_mz_axis[ppm]
        except KeyError as e:
            print('generating histogram axis for ppm {}'.format(ppm))
            self.generate_histogram_axis(ppm=ppm)
        return self.histogram_mz_axis[ppm]

    def generate_summary_spectrum(self, summary_type='mean', ppm=1., hist_axis = []):
        if hist_axis == []:
            hist_axis = self.get_histogram_axis(ppm=ppm)
        # calcualte mean along some m/z axis
        mean_spec = np.zeros(np.shape(hist_axis))
        for ii in range(0, len(hist_axis) - 1):
            mz_upper = hist_axis[ii + 1]
            mz_lower = hist_axis[ii]
            idx_left = bisect.bisect_left(self.mz_list, mz_lower)
            idx_right = bisect.bisect_right(self.mz_list, mz_upper)
            # slice list for code clarity
            count_vect = self.count_list[idx_left:idx_right]
            if summary_type == 'mean':
                count_vect = self.count_list[idx_left:idx_right]
                mean_spec[ii] = np.sum(count_vect)
            elif summary_type == 'freq':
                idx_vect = self.idx_list[idx_left:idx_right]
                mean_spec[ii] = float(len(np.unique(idx_vect)))
            else:
                raise ValueError('Summary type not recognised; {}'.format(summary_type))
        if summary_type == 'mean':
            mean_spec = mean_spec / len(self.index_list)
        elif summary_type == 'freq':
            mean_spec = mean_spec / len(self.index_list)
        return hist_axis, mean_spec

    def get_summary_image(self,summary_func='tic'):
        if summary_func not in ['tic','mic']: raise KeyError("requested type not in 'tic' mic'")
        #data_out = ion_datacube()
        # add precomputed pixel indices
        #data_out.coords = self.coords
        #data_out.pixel_indices = self.cube_pixel_indices
        #data_out.nRows = self.cube_n_row
        #data_out.nColumns = self.cube_n_col
        data_out=self.empty_datacube()
        data_out.add_xic(np.asarray(getattr(self, summary_func)), [0], [0])
        return data_out
