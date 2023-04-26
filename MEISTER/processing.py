import numpy as np
import scipy.signal as sig
import scipy.stats as stats
from scipy.stats import median_abs_deviation as mad
import pickle
from pyimzml.ImzMLWriter import ImzMLWriter
from scipy.io import loadmat
from brainpy import isotopic_variants, parse_formula
import random
from spike.File import Solarix 
import spike
from tqdm import tqdm
import pandas as pd

def load_buffer(path):

    D1 = Solarix.Import_1D(path)

    return D1

def proc_solarix_imaging(path, fid, apodize = True, flatten = False, zerofill=False):

    D1 = Solarix.Import_1D(path)     # Import_1D creates a SPIKE FTICRData object, from which everything is available
    #D1 = d1.copy() 
    D1.buffer = fid
    D1.center()
    if apodize:
        #D1.kaizer(4)
        D1.hamming()   # chaining  centering - apodisation - zerofill
    if zerofill:
        D1.zf(4)
                                  # kaiser(4) is an apodisation well adapted to FTICR, slightly more resolution than hamming - try varying the argument !
    if D1.axis1.itype == 0:       # means data is real (common case)
        D1.rfft().modulus()           # chaining  real FT - modulus
    else:                         #       data is complex, in Narrow-band acquisition
        D1.fft().modulus()            # chaining  complex FT - modulus
    if flatten:
        D1.bcorr(xpoints=50)          # flatten the baseline

    D1.set_unit('m/z')

    return D1


def fid2spec_solarix(D, mz_range):

    """
    converts a transient FID to a mass spectrum with a defined mass range
    """

    m = D.axis1.mz_axis()
    sp = D.buffer
    
    mz_filter = (m < mz_range[1]) & (m > mz_range[0])

    if len(sp.shape) == 1:
        sp_return = sp[mz_filter]
    if len(sp.shape) == 2:
        sp_return = sp[:,mz_filter]

    return m[mz_filter], sp_return

# def write_fid(path):

def get_peaks_solarix(D, thres=5, centroid=False):

    D.peakpick(autothresh=thres)
    if centroid:
        D.centroid()

    peak_list = D.pk2pandas()

    return {'mzs':peak_list['m/z'].values, 'intensity':peak_list['Intensity'].values}

def proc(fid):

    """
    the processing method for FT-ICR fid
    """
    #d = fid.copy()
    # hamming apodisation
    if len(fid.shape) == 1: 
        fid_apod = fid*np.hamming(fid.size)
    if len(fid.shape) == 2:
        fid_apod = fid*np.hamming(fid.shape[1])

    #dzf = zeros( (4*d.size), dtype=complex)  # zero-fill 4 times to improve quality
    #dzf[:d.size] = d[:]

    sp = np.fft.rfft(fid_apod)       # fourier transform
    spmod = np.real(np.sqrt(sp*sp.conj()))  # and take modulus

    return spmod


def fticr_mass_axis(h, calib):
    """
    returns an array which will calibrate a FT-ICR experiment
    h_values : the frequency axis after fft
    calib : mass calibration parameters
    """

    h[h<0.1] = 0.1
    if calib[2] == 0:
        m = calib[0]/(calib[1] + h)
    else:
        delta = calib[0]**2 + 4*calib[2]*(calib[1] + h)
        m = 2*calib[2] / (np.sqrt(delta) - calib[0])

    return m

def mass2freq(m, calib):

    m = np.maximum(m, 1.0)

    if calib[2] == 0:
        h = calib[0] / m - calib[1]
    else:
        h = calib[0] / m + calib[2] / (m**2)  - calib[1]
        
    return h


def fid2spec(fid, m, mz_range):

    """
    converts a transient FID to a mass spectrum with a defined mass range
    """
    sp = proc(fid)
    
    mz_filter = (m < mz_range[1]) & (m > mz_range[0])

    if len(sp.shape) == 1:
        sp_return = sp[mz_filter]
    if len(sp.shape) == 2:
        sp_return = sp[:,mz_filter]

    return m[mz_filter], sp_return


def peak_detection(mz, spec, prominence, threshold):

    foundpeaks = sig.find_peaks(spec, prominence = prominence, height=threshold)
    tic = np.sum(spec)
    # mzs = mz[foundpeaks[0]]
    # mzs_sorted = mzs[np.argsort(mzs)]
    # intensity_sorted = spec[foundpeaks[0]][np.argsort(mzs)]

    return {'mz':mz[foundpeaks[0]], 'intensity':spec[foundpeaks[0]],'tic':tic,'mz_index':foundpeaks[0]}


def alignMass(mz, peak_list_dir, drop_perc, save=True):

    with open(peak_list_dir+'.pkl', 'rb') as fp:
        peak_list = pickle.load(fp)

    hist = np.zeros((mz.size,))  
    print('binning peak locations of {} peak lists to the original mass axis'.format(len(peak_list)))

    TIC = []
    for peaks in peak_list:
        TIC.append(peaks['tic'])
        bin_idx = np.where(np.in1d(mz, peaks['mz']))[0]
        hist[bin_idx] +=1
    
    print('propogating peak intensities to the intensity matrix...')

    intens_mtx = []
    m_retained = mz[hist>drop_perc*len(peak_list)]

    for m in m_retained:
        print(m)
        column = []
        for peaks in peak_list:
            if m in peaks['mz']:
                column.append(peaks['intensity'][np.where(peaks['mz']==m)[0]][0])
            else:
                column.append(0)
        intens_mtx.append(column)
    if save:
        with open(peak_list_dir+'aligned.pkl', 'wb') as fp:
            pickle.dump({'mz':m_retained,'intens_mtx':np.array(intens_mtx),'tic':TIC}, fp, protocol=pickle.HIGHEST_PROTOCOL)


    return m_retained, np.array(intens_mtx)



def pklist2imzML(peak_list_name,coords):

    """
    """
    with open(peak_list_name+'.pkl', 'rb') as fp:
        peak_list = pickle.load(fp)

    with ImzMLWriter(peak_list_name+'.imzML') as w:
	        
	    for i in range(len(peak_list)):
	        
	        # writes data to the .ibd file
	        #print(i)
	        if peak_list[i]['mz'].size >0:
	            
	            w.addSpectrum(mzs = peak_list[i]['mz'],intensities = peak_list[i]['intensity'],
	                                    coords = tuple(coords[i]))
    
    print('succefully parsed the peak list to imzml!')


def extractMZFeatures(imzml_dataset, ppm, mz_range, feature_n = 0.05, mz_bins = []):
        
    if len(mz_bins) == 0:
        mz_bins = [mz_range[0]]
        while mz_bins[-1] < mz_range[1]:
            mz_bins.append(mz_bins[-1]+mz_bins[-1]*2*ppm*10**-6)
        
    print('number of mass bins {}'.format(len(mz_bins)))

    count = []
    for i in tqdm(range(len(mz_bins))):
        count.append(imzml_dataset.get_ion_image(mz_bins[i],ppm).xic_to_image(0).astype(bool).sum())
    
    mz_bins_filter = (np.array(count)>=int(imzml_dataset.coords.shape[0]*feature_n))
    mz_bins_use = np.array(mz_bins)[mz_bins_filter]
                
    datacube = imzml_dataset.get_ion_image(mz_bins_use, ppm)
    
    datacube_array = [datacube.xic_to_image(i) for i in range(len(mz_bins_use))]
    datacube_array = np.concatenate(datacube_array,axis=1)
        
    return datacube_array, mz_bins, mz_bins_use, count


def getIntensMtxData(imzML_dataset, mz_range, feature_n, ppm=2, from_imzml=True, mz_features=[], intens_array=np.array([])):
    """
    """

    if from_imzml:

        if len(mz_features) == 0:
            intens_array, mz_bins_use, c = extractMZFeatures(imzML_dataset, ppm, mz_range, feature_n=feature_n)
        else:
            intens_array, mz_bins_use, c = extractMZFeatures(imzML_dataset, ppm, mz_range, feature_n=feature_n, mz_bins=mz_features)

        intens_mtx = pd.DataFrame(intens_array, columns=mz_bins_use)

    else:
        intens_mtx = pd.DataFrame(intens_array, columns=mz_features)

    return intens_mtx


def IonImg_show(data, coord):

    img = np.empty((max(coord[:,1])+1,max(coord[:,0])+1,))
    img[:] = np.nan

    for i in range(len(coord)):
        img[coord[i,1],coord[i,0]] = data[i]
        
    return img



def IonImg(data, coord, background, return_mask):
    
    nx = max(coord[:,1])+1
    ny = max(coord[:,0])+1
    if (nx % 2) == 0:
        nx = nx
    else:
        nx += 1
    if (ny % 2) == 0:
        ny = ny
    else:
        ny += 1
    if background:
        img = np.zeros((nx,ny))
    else:
        img = np.empty((nx,ny,))
        img[:] = np.nan
    for i in range(len(coord)):
        img[coord[i,1],coord[i,0]] = data[i]
    if return_mask:
        img_mask = np.zeros((img.shape))
        for i in range(len(coord)):
            img_mask[coord[i,1],coord[i,0]] = 1

        return img, img_mask
    return img


def hyperspectral_vis(results,coord,background):
    
    results[:,0] = (results[:,0]-min(results[:,0]))/(max(results[:,0])-min(results[:,0]))
    results[:,1] = (results[:,1]-min(results[:,1]))/(max(results[:,1])-min(results[:,1]))
    results[:,2] = (results[:,2]-min(results[:,2]))/(max(results[:,2])-min(results[:,2]))
    
    results[:,0] *= np.uint8(255/results[:,0].max())
    results[:,1] *= np.uint8(255/results[:,1].max())
    results[:,2] *= np.uint8(255/results[:,2].max())
    
    if background:
        img = np.full((max(coord[:,1]),max(coord[:,0]),3),0,dtype='int')
    else:
        img = np.full((max(coord[:,1]),max(coord[:,0]),3),255,dtype='int')

    for i in range(len(coord)):
        img[coord[i,1]-1,coord[i,0]-1] = results[i]

    return img




def simulate_transient(t, compounds, adducts, abundances, calib, random_compounds=[], add_random = False):

    data0 = np.zeros(t.size,dtype=complex)
    noise = 10
    LB = 1.1
    npeaks = 3
    charge = 1

    idx = 0
    for compound in compounds:
        compound = compound.copy()
        idx_ = 0
        for adduct in adducts:
            if adduct in compound.keys():
                compound[adduct] = compound[adduct]+1
            else:
                compound[adduct] = 1
            theoretical_isotopic_cluster = isotopic_variants(compound, npeaks, charge)
            for peak in theoretical_isotopic_cluster:
                data0 +=  abundances[idx_][idx]* peak.intensity* np.sin(2*np.pi*mass2freq(peak.mz, calib)*t)* np.exp(-LB*t)
            idx_ = idx_ +1
        idx = idx +1

    if add_random:
        rand_comp = random.sample(random_compounds, 3)

        for compound in rand_comp:
            compound = compound.copy()
            idx_ = 0
            for adduct in adducts:
                if adduct in compound.keys():
                    compound[adduct] = compound[adduct]+1
                else:
                    compound[adduct] = 1
                theoretical_isotopic_cluster = isotopic_variants(compound, npeaks, charge)
                for peak in theoretical_isotopic_cluster:
                    #print(peak.mz, peak.intensity, mass2freq(peak.mz, calib))
                    data0 +=  random.uniform(0.3,0.6)* peak.intensity* np.sin(2*np.pi*mass2freq(peak.mz, calib)*t)* np.exp(-LB*t)
                idx_ = idx_ +1
            idx = idx +1


    data = data0 + noise*(np.random.randn(t.size))

    return data0, data