"""
Soufiane CHAMI
Created : 15/11/2018
"""
import os
import numpy as np
import pandas as pd
from pylab import find
import scipy.io
from sklearn.externals import joblib
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
# returns a list of the training and testing file locations for easier import
# -----------------------------------------------------------------------------


def get_files():
    header_loc, arousal_loc, signal_loc, is_training = [], [], [], []
    rootDir = 'path/to/PhysioNet Training Database/'
    for dirName, subdirList, fileList in os.walk(rootDir, followlinks=True):
        if dirName.startswith(rootDir):
                is_training.append(True)
                dirName=dirName.replace('\\', '/')

                for fname in fileList:
                    if '.hea' in fname:
                        header_loc.append(dirName + '/' + fname)
                    if '-arousal.mat' in fname:
                        arousal_loc.append(dirName + '/' + fname)
                    if 'mat' in fname and 'arousal' not in fname:
                        signal_loc.append(dirName + '/' + fname)

    # combine into a data frame
    data_locations = {'header':      header_loc,
                      'arousal':     arousal_loc,
                      'signal':      signal_loc,
                      }

    # Convert to a data-frame
    df = pd.DataFrame(data=data_locations)

    # Split the data frame into training and testing sets.
    
    return df

# -----------------------------------------------------------------------------
# import the output vector, given the file name.
# e.g. /training/tr04-0808/tr04-0808-arousal.mat
# -----------------------------------------------------------------------------

def import_arousals(file_name):
    import h5py
    import numpy
    f = h5py.File(file_name, 'r')
    arousals = numpy.array(f['data']['arousals'])
    return arousals

def import_signals(file_name):
    
    return np.transpose(scipy.io.loadmat(file_name)['val'])


# -----------------------------------------------------------------------------
# Take a header file as input, and returns the names of the signals
# For the corresponding .mat file containing the signals.
# -----------------------------------------------------------------------------

def import_signal_names(file_name):
    with open(file_name, 'r') as myfile:
        s = myfile.read()
        s = s.split('\n')
        s = [x.split() for x in s]

        n_signals = int(s[0][1])
        n_samples = int(s[0][3])
        Fs        = int(s[0][2])

        s = s[1:-1]
        s = [s[i][8] for i in range(0, n_signals)]
    return s, Fs, n_samples



# -----------------------------------------------------------------------------
# Get a given subject's train data based on index
# -----------------------------------------------------------------------------
def get_subject_data(index):
    df = get_files()
    file_name = list(df.iloc[index])
    signal_names = import_signal_names(file_name[0]) # tr0x-0xxx.hea
    this_arousal= import_arousals(file_name[1]) # tr0x-0xxx-arousal.mat
    this_signal = import_signals(file_name[2]) # tr0x-0xxx.mat
    signal_names[0].append("Target")
    this_data      = np.append(this_signal, this_arousal, axis=1)
    this_data      = pd.DataFrame(this_data, index=None, columns=signal_names[0])
    print("Hooray")
# return he target array 
    y_train= this_data["Target"]
    y_train.loc[y_train!=0]= 1
# return x_train 
    x_train= this_data.drop(columns=["Target"])
    return x_train, y_train


# -----------------------------------------------------------------------------
# Get a given subject's test data based on index
# -----------------------------------------------------------------------------


def get_subject_data_test(index):
    df = get_files()
    file_name      = list(df.iloc[index])
    signal_file    = import_signals(file_name)
    signal_names   = import_signal_names(file_name[2])
    this_signal    = import_signals(file_name[2])
    this_data      = this_signal
    this_data      = pd.DataFrame(this_data, index=None, columns=signal_names)

    return this_data
