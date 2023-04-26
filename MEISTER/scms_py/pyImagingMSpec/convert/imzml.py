__author__ = 'palmer'
import argparse
import numpy as np

def centroid_imzml(input_filename, output_filename, step=[], apodization=False, w_size=10, min_intensity=1e-5, prevent_duplicate_pixels=False):

    # write a file to imzml format (centroided)
    """
    :type input_filename string - source file path (must be .imzml)
    :type output_filename string - output file path (must be .imzml)
    :type step tuple grid spacing of pixels (if [] the script will try and guess it)
    :type apodization boolean whether to try and remove FT wiglet artefacts
    :type w_size window side (m/z bins) for apodization
    :type min_intensity: float minimum intensity peaks to return during centroiding
    :type prevent_duplicate_pixels bool if True will only return the first spectrum for pixels with the same coodinates
    """
    from pyimzml.ImzMLParser import ImzMLParser
    from pyimzml.ImzMLWriter import ImzMLWriter
    from pyMSpec.centroid_detection import gradient

    imzml_in = ImzMLParser(input_filename)
    precisionDict = {'f':("32-bit float", np.float32), 'd': ("64-bit float", np.float64), 'i': ("32-bit integer", np.int32), 'l': ("64-bit integer", np.int64)}
    mz_dtype = precisionDict[imzml_in.mzPrecision][1]
    int_dtype = precisionDict[imzml_in.intensityPrecision][1]
    # Convert coords to index -> kinda hacky
    coords = np.asarray(imzml_in.coordinates).round(5)
    coords -= np.amin(coords, axis=0)
    if step == []:  # have a guesss
        step = np.array([np.median(np.diff(np.unique(coords[:, i]))) for i in range(coords.shape[1])])
        step[np.isnan(step)] = 1
    print 'estimated pixel size: {} x {}'.format(step[0], step[1])
    coords = coords / np.reshape(step, (3,)).T
    coords = coords.round().astype(int)
    ncol, nrow, _ = np.amax(coords, axis=0) + 1
    print 'new image size: {} x {}'.format(nrow, ncol)
    if prevent_duplicate_pixels:
        b = np.ascontiguousarray(coords).view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1])))
        _, coord_idx = np.unique(b, return_index=True)
        print np.shape(imzml_in.coordinates), np.shape(coord_idx)

        print "original number of spectra: {}".format(len(coords))
    else:
        coord_idx = range(len(coords))
    n_total = len(coord_idx)
    print 'spectra to write: {}'.format(n_total)
    with ImzMLWriter(output_filename, mz_dtype=mz_dtype, intensity_dtype=int_dtype) as imzml_out:
        done = 0
        for key in range(np.shape(imzml_in.coordinates)[0]):
            print key
            if all((prevent_duplicate_pixels, key not in coord_idx)):  # skip duplicate pixels
                continue
            mzs, intensities = imzml_in.getspectrum(key)
            if apodization:
                from pyMSpec import smoothing
                # todo - add to processing list in imzml
                mzs, intensities = smoothing.apodization(mzs, intensities, {'w_size':w_size})
            mzs_c, intensities_c, _ = gradient(mzs, intensities, min_intensity=min_intensity)
            pos = coords[key]
            if len(pos)==2:
                pos.append(0)
            pos = (pos[0], nrow - 1 - pos[1], pos[2])
            imzml_out.addSpectrum(mzs_c, intensities_c, pos)
            done += 1
            if done % 1000 == 0:
                print "[%s] progress: %.1f%%" % (input_filename, float(done) * 100.0 / n_total)
        print "finished!"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert centroids from profile imzml to centroided imzML")
    parser.add_argument('input', type=str, help="input imzml file")
    parser.add_argument('output', type=str, help="output filename (centroided imzml)")
    args = parser.parse_args()
    centroid_imzml(args.input, args.output)
