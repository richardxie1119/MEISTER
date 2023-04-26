import h5py
import sys
import numpy as np

from pyimzml.ImzMLWriter import ImzMLWriter

def imzml(input_filename, output_filename,smoothMethod="nosmooth",centroid=False):
    import h5py
    import numpy as np
    ### Open files
    h5 = h5py.File(input_filename, 'r')  # Readonly, file must exist
    ### get root groups from input data
    root_group_names = h5.keys()
    spots = h5['Spots']
    spectraGroup = 'InitialMeasurement'
    mzs = np.asarray(h5['/SamplePositions/GlobalMassAxis/']['SamplePositions']) # we don't write this but will use it for peak detection
    file_version = h5['Version'][0]    # some hard-coding to deal with different file versions
    if file_version > 5:
        coords = h5['Registrations']['0']['Coordinates']
    else:
        coords = h5['Coordinates']

    coords = np.asarray(coords).T.round(5)
    coords -= np.amin(coords, axis=0)
    step = np.array([np.mean(np.diff(np.unique(coords[:, i]))) for i in range(3)])
    step[np.isnan(step)] = 1
    coords /= np.reshape(step, (3,))
    coords = coords.round().astype(int)
    ncol, nrow, _ = np.amax(coords, axis=0) + 1
    g = h5['Spots/0/'+spectraGroup+'/']
    mz_dtype = g['SamplePositions/SamplePositions'][:].dtype
    int_dtype = g['Intensities'][:].dtype
    print 'dim: {} x {}'.format(nrow,ncol)
    n_total = len(spots.keys())
    done = 0
    keys = map(str, sorted(map(int, h5['Spots'].keys())))
    ### write spectra
    with ImzMLWriter(output_filename, mz_dtype=mz_dtype, intensity_dtype=int_dtype) as imzml:
        n = 0
        for key, pos in zip(keys, coords):
            spot = spots[key]
            ## make new spectrum
            intensities = np.asarray(spot[spectraGroup]['Intensities'])
            if smoothMethod != []:
                    intensities = smooth_spectrum(mzs,intensities,smoothMethod)
            if centroid:
                from pyMS import centroid_detection
                mzs, intensities, _ = centroid_detection.gradient(mzs,intensities, max_output=-1, weighted_bins=3)
            # write to file
            pos = (nrow - 1 - pos[1], pos[0], pos[2])
            imzml.addSpectrum(mzs, intensities, pos)
            done += 1
            if done % 1000 == 0:
                print "[%s] progress: %.1f%%" % (input_filename, float(done) * 100.0 / n_total)
    print "finished!"


def centroid_imzml(input_filename, output_filename,smoothMethod="nosmooth"):
    raise NotImplementedError('Function removed: use h5.centroids(...centroid=True)')

def smooth_spectrum(mzs,intensities,smoothMethod):
    import pyMS.smoothing as smoothing
    if smoothMethod == 'sg_smooth':
        intensities =  smoothing.sg_smooth(mzs,intensities,n_smooth=1)
    elif smoothMethod == 'apodization':
        intensities = smoothing.apodization(mzs,intensities,w_size=10)
    elif smoothMethod == "rebin":
        intensities = smoothing.rebin(mzs,intensities,delta_mz = 0.1)
    else:
        raise ValueError("method {} not known")
    return intensities


def hdf5(filename_in, filename_out,info,smoothMethod="nosmooth"):
    import h5py
    import numpy as np
    import datetime
    import scipy.signal as signal
    from pyMS import centroid_detection
    import sys
    #from IPython.display import display, clear_output

    ### Open files
    f_in = h5py.File(filename_in, 'r')  # Readonly, file must exist
    f_out = h5py.File(filename_out, 'w')  # create file, truncate if exists
    print filename_in
    print filename_out
    ### get root groups from input data
    root_group_names = f_in.keys()
    spots = f_in['Spots']
    file_version = f_in['Version'][0]
    # some hard-coding to deal with different file versions
    if file_version > 5:
        coords = f_in['Registrations']['0']['Coordinates']
    else:
        coords = f_in['Coordinates']
    spectraGroup = 'InitialMeasurement'
    Mzs = np.asarray(f_in['/SamplePositions/GlobalMassAxis/']['SamplePositions']) # we don't write this but will use it for peak detection

    ### make root groups for output data
    spectral_data = f_out.create_group('spectral_data')
    spatial_data = f_out.create_group('spatial_data')
    shared_data = f_out.create_group('shared_data')

    ### populate common variables - can hardcode as I know what these are for h5 data
    # parameters
    instrument_parameters_1 = shared_data.create_group('instrument_parameters/001')
    instrument_parameters_1.attrs['instrument name'] = 'Bruker Solarix 7T'
    instrument_parameters_1.attrs['mass range'] = [Mzs[0],Mzs[-1]]
    instrument_parameters_1.attrs['analyser type'] = 'FTICR'
    instrument_parameters_1.attrs['smothing during convertion'] = smoothMethod
    instrument_parameters_1.attrs['data conversion'] = 'h5->hdf5:'+str(datetime.datetime.now())
    # ROIs
        #todo - determine and propagate all ROIs
    sample_1 = shared_data.create_group('samples/001')
    sample_1.attrs['name'] = info["sample_name"]
    sample_1.attrs['source'] = info["sample_source"]
    sample_1.attrs['preparation'] = info["sample_preparation"]
    sample_1.attrs['MALDI matrix'] = info["maldi_matrix"]
    sample_1.attrs['MALDI matrix application'] = info["matrix_application"]
    ### write spectra
    n = 0
    for key in spots.keys():
        spot = spots[key]
        ## make new spectrum
        #mzs,intensities = nosmooth(Mzs,np.asarray(spot[spectraGroup]['Intensities']))
        if smoothMethod == 'nosmooth':
            mzs,intensities = mzs,intensities = nosmooth(Mzs,np.asarray(spot[spectraGroup]['Intensities']))
        elif smoothMethod == 'nosmooth':
            mzs,intensities = sg_smooth(Mzs,np.asarray(spot[spectraGroup]['Intensities']))
        elif smoothMethod == 'apodization':
            mzs,intensities = apodization(Mzs,np.asarray(spot[spectraGroup]['Intensities']))
        else:
            raise ValueError('smooth method not one of: [nosmooth,nosmooth,apodization]')
        mzs_list, intensity_list, indices_list = centroid_detection.gradient(mzs,intensities, max_output=-1, weighted_bins=3)

        # add intensities
        this_spectrum = spectral_data.create_group(key)
        this_intensities = this_spectrum.create_dataset('centroid_intensities', data=np.float32(intensity_list),
                                                    compression="gzip", compression_opts=9)
        # add coordinates
        key_dbl = float(key)
        this_coordiantes = this_spectrum.create_dataset('coordinates',
                                                    data=(coords[0, key_dbl], coords[1, key_dbl], coords[2, key_dbl]))
        ## link to shared parameters
        # mzs
        this_mzs = this_spectrum.create_dataset('centroid_mzs', data=np.float32(mzs_list), compression="gzip",
                                            compression_opts=9)
        # ROI
        this_spectrum['ROIs/001'] = h5py.SoftLink('/shared_data/regions_of_interest/001')
        # Sample
        this_spectrum['samples/001'] = h5py.SoftLink('/shared_data/samples/001')
        # Instrument config
        this_spectrum['instrument_parameters'] = h5py.SoftLink('/shared_data/instrument_parameters/001')
        n += 1
        if np.mod(n, 10) == 0:
            #clear_output(wait=True)
            print('{:3.2f}\% complete\r'.format(100.*n/np.shape(spots.keys())[0], end="\r")),
            sys.stdout.flush()

    f_in.close()
    f_out.close()
    print 'fin'


if __name__ == '__main__':
    centroidh5(sys.argv[1], sys.argv[1][:-3] + ".imzML")
