__author__ = 'palmer'


def hdf5_centroids_IMS(input_filename, output_filename):
    # Convert hdf5 to imzML
    imsDataset = inMemoryIMS(input_filename,cache_spectra=False,do_summary=False)
    coords = imsDataset.coords
    coords -= np.amin(coords, axis=0)
    step = []
    for i in range(3):
        u = list(np.diff(np.unique(coords[:, i])))
        if u == []:
            step.append(1)
        else:
            step.append(np.median(u))
    step = np.asarray(step)
    coords /= np.reshape(step, (3,))
    coords = coords.round().astype(int)
    ncol, nrow, _ = np.amax(coords, axis=0) + 1
    print 'dim: {} x {}'.format(nrow,ncol)
    mz_dtype = imsDataset.get_spectrum(0).get_spectrum(source="centroids")[0].dtype
    int_dtype = imsDataset.get_spectrum(0).get_spectrum(source="centroids")[1].dtype
    n_total = len(imsDataset.index_list)
    with ImzMLWriter(output_filename, mz_dtype=mz_dtype, intensity_dtype=int_dtype) as imzml:
        done = 0
        for index in imsDataset.index_list:
            this_spectrum = imsDataset.get_spectrum(index)
            mzs,intensities = this_spectrum.get_spectrum(source='centroids')
            pos = coords[index]
            pos = (nrow - 1 - pos[0], pos[1], pos[2])
            imzml.addSpectrum(mzs, intensities, pos)
            done += 1
            if done % 1000 == 0:
                print "[%s] progress: %.1f%%" % (input_filename, float(done) * 100.0 / n_total)
        print "finished!"

def smooth_spectrum(mzs,intensities,smoothMethod):
    import pyMS
    if smoothMethod=='sg_smooth':
        intensities =  pyMS.sg_smooth(mzs,intensities,n_smooth=1)
    elif smoothMethod=='apodization':
        intensities = pyMS.apodization(mzs,intensities,w_size=10)
    else:
        raise ValueError("method {} not known".format(smoothMethod))
    return intensities