import h5py
import sys
import numpy as np
"""
sl files use an lz4 compression filter for which a plugin needs to be installed
Mac:
sudo port install hdf5-lz4-plugin
Ubuntu:   (http://gisaxs.com/index.php/HDF5)
sudo add-apt-repository ppa:eugenwintersberger/pni
sudo apt-get update
sudo apt-get install hdf5-plugin-lz4


note that the filter isn't actually registered until the first dataset is access so won't be reported by h5py until after then.

Remember to restart terminal running python

Whilst h5py claims to look in the default directory for filters I had to add the environment variable:
os.environ['HDF5_PLUGIN_PATH'] = "/opt/local/lib/hdf5"
"""

class slFile():
    def __init__(self, input_filename, region_name=""):
        self.region_name = region_name
        self.load_file(input_filename)

    def _check_file_version(self):
        self.file_version = self.sl['Version'][0]
        print 'sl file version: {}'.format(self.file_version)
        if not self.file_version in range(16, 23):
            raise ValueError('File version {} out of range.'.format(self.file_version))

    def _get_spotlist(self):
        ### get root groups from input data
        if self.region_name == "":
            self.spotlist = range(self.sl['SpectraGroups']['InitialMeasurement']['images'].shape[1])
        else:
            print self.region_name
            region_name = self.sl['Regions'].visit(self.find_name)
            if region_name == None:
                raise ValueError("Requested region {} not found".format(self.region_name))
            self.spotlist = self.sl['Regions'][region_name]['SpotList']
        self.spotlist = np.asarray(self.spotlist)

    def _get_spectragroup(self):
        self.initialMeasurement = self.sl['SpectraGroups']['InitialMeasurement']
        self.Mzs = np.asarray(self.initialMeasurement['SamplePositions'][
                              'SamplePositions'])  # we don't write this but will use it for peak detection
        self.spectra = self.initialMeasurement['spectra']

    def _get_coordinates(self):
        ### Get Coordinates for spotlist
        self.coords = np.asarray(self.sl['Registrations']['0']['Coordinates'])
        if np.shape(self.coords)[0] != 3:
            self.coords = self.coords.T
        if np.shape(self.coords)[0] != 3:
            raise ValueError('coords second dimension should be 3 {}'.format(np.shape(self.coords)))

    def load_file(self, input_filename):
        # get a handle on the file
        self.sl = h5py.File(input_filename, 'r')  # Readonly, file must exist
        self._check_file_version()
        self._get_spotlist()
        self._get_spectragroup()
        self._get_coordinates()

    def get_spectrum(self, index):
        intensities = np.asarray(self.spectra[index, :])
        return self.Mzs, intensities

    def find_name(self, name):
        if 'name' in self.sl['Regions'][name].attrs.keys():
            if self.sl['Regions'][name].attrs['name'] == self.region_name:
                assert isinstance(name, object)
                return name


def centroid_imzml(input_filename, output_filename, step=[], apodization=False, w_size=10, min_intensity=1e-5,
                   region_name="", prevent_duplicate_pixels=False):
    # write a file to imzml format (centroided)
    """
    :type min_intensity: float
    """
    from pyimzml.ImzMLWriter import ImzMLWriter
    from pyMSpec.centroid_detection import gradient
    sl = slFile(input_filename, region_name=region_name)
    mz_dtype = sl.Mzs.dtype
    int_dtype = sl.get_spectrum(0)[1].dtype
    # Convert coords to index -> kinda hacky
    coords = np.asarray(sl.coords.copy()).T.round(5)
    coords -= np.amin(coords, axis=0)
    if step == []:  # have a guesss
        step = np.array([np.median(np.diff(np.unique(coords[sl.spotlist, i]))) for i in range(3)])
        step[np.isnan(step)] = 1
    print 'estimated pixel size: {} x {}'.format(step[0], step[1])
    coords = coords / np.reshape(step, (3,)).T
    coords = coords.round().astype(int)
    ncol, nrow, _ = np.amax(coords, axis=0) + 1
    print 'new image size: {} x {}'.format(nrow, ncol)
    if prevent_duplicate_pixels:
        b = np.ascontiguousarray(coords).view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1])))
        _, coord_idx = np.unique(b, return_index=True)
        print np.shape(sl.spotlist), np.shape(coord_idx)

        print "original number of spectra: {}".format(len(coords))
    else:
        coord_idx = range(len(coords))
    n_total = len(coord_idx)
    print 'spectra to write: {}'.format(n_total)
    with ImzMLWriter(output_filename, mz_dtype=mz_dtype, intensity_dtype=int_dtype) as imzml:
        done = 0
        for key in sl.spotlist:
            if all((prevent_duplicate_pixels, key not in coord_idx)):# skip duplicate pixels
                #print 'skip {}'.format(key)
                continue
            mzs, intensities = sl.get_spectrum(key)
            if apodization:
                from pyMSpec import smoothing
                # todo - add to processing list in imzml
                mzs, intensities = smoothing.apodization(mzs, intensities)
            mzs_c, intensities_c, _ = gradient(mzs, intensities, weighted_bins=5, min_intensity=min_intensity)
            pos = coords[key]
            pos = (pos[0], nrow - 1 - pos[1], pos[2])
            imzml.addSpectrum(mzs_c, intensities_c, pos)
            done += 1
            if done % 1000 == 0:
                print "[%s] progress: %.1f%%" % (input_filename, float(done) * 100.0 / n_total)
        print "finished!"


def centroid_IMS(input_filename, output_filename, instrumentInfo={}, sharedDataInfo={}):
    from pyMS.centroid_detection import gradient
    # write out a IMS_centroid.hdf5 file
    sl = slFile(input_filename)
    n_total = np.shape(sl.spectra)[0]
    with h5py.File(output_filename, 'w') as f_out:
        ### make root groups for output data
        spectral_data = f_out.create_group('spectral_data')
        spatial_data = f_out.create_group('spatial_data')
        shared_data = f_out.create_group('shared_data')

        ### populate common variables - can hardcode as I know what these are for h5 data
        # parameters
        instrument_parameters_1 = shared_data.create_group('instrument_parameters/001')
        if instrumentInfo != {}:
            for tag in instrumentInfo:
                instrument_parameters_1.attrs[tag] = instrumentInfo[tag]
                # ROIs
                # todo - determine and propagate all ROIs
        roi_1 = shared_data.create_group('regions_of_interest/001')
        roi_1.attrs['name'] = 'root region'
        roi_1.attrs['parent'] = ''
        # Sample
        sample_1 = shared_data.create_group('samples/001')
        if sharedDataInfo != {}:
            for tag in sharedDataInfo:
                sample_1.attrs[tag] = sharedDataInfo[tag]

        done = 0
        for key in range(0, n_total):
            mzs, intensities = sl.get_spectrum(key)
            mzs_c, intensities_c, _ = gradient(mzs, intensities)
            this_spectrum = spectral_data.create_group(str(key))
            _ = this_spectrum.create_dataset('centroid_mzs', data=np.float32(mzs_c), compression="gzip",
                                             compression_opts=9)
            # intensities
            _ = this_spectrum.create_dataset('centroid_intensities', data=np.float32(intensities_c), compression="gzip",
                                             compression_opts=9)
            # coordinates
            _ = this_spectrum.create_dataset('coordinates',
                                             data=(sl.coords[0, key], sl.coords[1, key], sl.coords[2, key]))
            ## link to shared parameters
            # ROI
            this_spectrum['ROIs/001'] = h5py.SoftLink('/shared_data/regions_of_interest/001')
            # Sample
            this_spectrum['samples/001'] = h5py.SoftLink('/shared_data/samples/001')
            # Instrument config
            this_spectrum['instrument_parameters'] = h5py.SoftLink('/shared_data/instrument_parameters/001')
            done += 1
            if done % 1000 == 0:
                print "[%s] progress: %.1f%%" % (input_filename, float(done) * 100.0 / n_total)
        print "finished!"


if __name__ == '__main__':
    centroid_imzml(sys.argv[1], sys.argv[1][:-3] + ".imzML")
