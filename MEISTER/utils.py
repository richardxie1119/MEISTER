import os
from os import path
import numpy as np
import xml.etree.ElementTree as ET
import re
from tqdm import tqdm
import pickle
import requests
import io
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, './')
from pyImagingMSpec.inMemoryIMS import inMemoryIMS


def loadimzMLData(file_name):

    imzML_dataset = inMemoryIMS(file_name)

    return imzML_dataset

def loadBrukerFIDs(file_path, fid_length, read_length, fid_idx, verbose = False):
    """
    Loads in binary FIDs from Bruker ser files. 
    file_path : path to Bruker ser file. The file is named "ser" and is contained in the Bruker .d folder.
    fid_length : the length (in data points) of each FID.
    read_length : specifies how much of each transient to read. Specify length or set read_length="all" to read the entire FID
    fid_idx : this is the index of the FID within the ser file. Can be one or multiple FIDs. By loading one or a small number of FIDs computer memory is saved.
    """
    fids = [] # initialize fids 
    if path.exists(file_path):

        f = open(file_path,'r') #open specified FID(s) from ser file

        if type(fid_idx) == list or type(fid_idx) == np.ndarray:

            #print('loading {} FID from file...'.format(len(fid_idx)))

            for i in range(len(fid_idx)):

                if verbose:
                    print('loading {} FID from file...'.format(i))

                f.seek(4*(fid_idx[i]-1)*fid_length) #seek FID locations within the .ser file

                if read_length == 'all':
                    fid = np.fromfile(f, count = fid_length, dtype = 'int32')
                    fids.append(fid)
                else:
                    fid = np.fromfile(f, count = read_length, dtype = 'int32')
                    fids.append(fid)
            
        else:

            f.seek(4*(fid_idx-1)*fid_length) #seek FID locations within the .ser file

            if read_length == 'all':
                fid = np.fromfile(f, count = fid_length, dtype = 'int32') #read entire transient
                fids.append(fid)
            else:
                fid = np.fromfile(f, count = read_length, dtype = 'int32') #read transient to specified read_length
                fids.append(fid)

        f.close()

    else:
        raise Exception('ser file does not exist in the provided file path. please double check.')
        #error if file path is not valid
    return np.array(fids,dtype='float64')


def loadBrukerMethod(file_path):

    """TODO"""

    return 'A'


def parseImagingInfo(file_path):

    """parses the ImagingInfo.xml file in the imaging .d folder, and returns the relative
    coordinates for each imaged regions, starting with RXX. The parsed dictionary contains
    the arrays of relative coordinates Xs and Ys under keys named as the regions (RXX).

    """

    tree = ET.parse(file_path)
    root = tree.getroot()

    parsed_spots = {}
    spotNames = []
    scan = []
    TIC = []
    for child in root:
        spotNames.append(child.find('spotName').text)
        scan.append(child.find('count').text)
        TIC.append(child.find('tic').text)

    ROI = set([spot[:3] for spot in spotNames])

    for roi in ROI:

        coord = []
        scan_idx = []
        tic = []

        for i in range(len(spotNames)):
            spot = spotNames[i]

            if roi in spot:
                coord.append([int(re.search('X(.*)Y' ,spot).group(1)),
                int(re.search('Y(.*)' ,spot).group(1))])
                scan_idx.append(scan[i])
                tic.append(TIC[i])
                
        scan_idx = np.array(scan_idx, dtype='int64')
        tic = np.array(tic, dtype='float')

        coord = np.array(coord)
        coord[:,0] -= coord[:,0].min()
        coord[:,1] -= coord[:,1].min()

        parsed_spots[roi] = {'coordinates':coord, 'scan_index':scan_idx, 'tic':tic}

    return parsed_spots



def LipidMaps_annotate(mass_list,adducts,ppm,site_url):
    
    Data = []
    matched = []
    unmatched = []
    
    for i in tqdm(range(len(mass_list))):
        mass = mass_list[i]
        tolerance = ppm*1e-6*mass
        Data_ = []
        for adduct in adducts:
            url = site_url+'/{}/{}/{}'.format(mass,adduct,tolerance)
            
            urlData = requests.get(url).content.decode('utf-8')[7:-9]            
            rawData = pd.read_csv(io.StringIO(urlData),sep='\t',error_bad_lines=False,index_col=False)
            
            Data_.append(rawData)
            #Data.append(rawData)
        df = pd.concat(Data_, ignore_index=True)
        df['Input m/z'] = [mass]*df.shape[0]
        
        if df.empty:
            unmatched.append(mass)
        else:
            matched.append(mass) 
            Data.append(df)
            
    annot_df = pd.concat(Data, ignore_index=True)
    return annot_df, matched, unmatched


from sklearn.metrics import classification_report, confusion_matrix,multilabel_confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure(figsize=(7,7))
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()