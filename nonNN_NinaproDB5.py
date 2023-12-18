import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import multiprocessing
from scipy.signal import butter,filtfilt,iirnotch
#import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import random

leaveOut = 0
# SVM or RF (random forest)
classifier = "RF"

def getData (n):
    return  window(torch.tensor(((pd.read_csv('./NinaproDB5/s'+str(n)+'/emgS'+str(n)+'_E2.csv')).to_numpy()), dtype=torch.int))

def getRestim (n):
    return  window(torch.tensor((pd.read_csv('./NinaproDB5/s'+str(n)+'/restimulusS'+str(n)+'_E2.csv')).to_numpy(), dtype=torch.int))

# windowing
wLen = 250 #ms
def window (e):
    return e.unfold(dimension=0, size=int(wLen / 5), step=10)

#delete windows with both 0's and other number (for restimulus labels) and excess rest windows
def filterIndices (labels):
    indices = []
    numLabel = {}
    for x in range(len(labels)):
        L = torch.chunk(labels[x], 2, dim=1)
        if torch.equal(L[0], L[1]):
            if L[0][0][0] == 0:
                r = random.random()
                if r > 0.96:
                    indices += [x]
                    if 0 in numLabel:
                        numLabel[0] += 1
                    else:
                        numLabel[0] = 1
            else:
                indices += [x]
                temp = int(L[0][0][0])
                if temp in numLabel:
                    numLabel[temp] += 1
                else:
                    numLabel[temp] = 1
    #print(numLabel)
    return indices

def contract(R):
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(len(R), 18))
    for x in range(len(R)):
        labels[x][int(R[x][0][0])] = 1.0
    #print(labels.size())
    return labels

def filters(emg):
    b, a = butter(N=3, Wn=5, btype='highpass', analog=False, fs=200.0)
    emgB = torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())

    b, a = iirnotch(w0=50.0, Q=0.0001, fs=200.0)
    return torch.from_numpy(np.flip(filtfilt(b, a, emgB),axis=0).copy())

numFeatures = 3
def extractFeatures (emg):
    # feature extraction: sum of absolute values
    SAV = torch.sum(torch.abs(emg), dim=2)

    # feature extraction: standard deviation of fourier transform
    # based on formula in section 3.2 of https://www.sciencedirect.com/science/article/pii/S0925231220303283?via%3Dihub#sec0006
    emgFFT = torch.abs(torch.fft.fft(emg, dim=2))
    STD = torch.std(emgFFT, dim=2)

    #slope sign change
    threshold = 0.01  # volts (should range between 0.00005 (50 microV) and 0.1 (100 mV))
    differences = emg[:, :, 1:] - emg[:, :, :-1]
    sign_changes = torch.logical_and(differences[:, :, :-1] * differences[:, :, 1:] < 0, torch.abs(differences[:, :, 1:]) > threshold)
    SSC = torch.sum(sign_changes, dim=2)

    '''
    #mean absolute value
    MAV = torch.mean(torch.abs(emg), dim=2)

    # root mean squared
    RMS = torch.sqrt(torch.mean(emg ** 2, dim=2))

    # waveform length
    wl = torch.sum(torch.abs(emg[:, :, 1:] - emg[:, :, :-1]), dim=2)

    # mean frequency
    power_spectrum = torch.abs(torch.fft.fft(emg, dim=2)) ** 2
    sampling_frequency = 200  # sampling frequency in Hz
    frequency = torch.fft.fftfreq(emg.shape[2], d=1/sampling_frequency)
    frequency = frequency.unsqueeze(0).unsqueeze(1)
    sum_product = torch.sum(power_spectrum * frequency, dim=2)
    total_sum = torch.sum(power_spectrum, dim=2)
    MNF = sum_product / total_sum

    # power spectrum ratio
    max_index = torch.argmax(power_spectrum, dim=2)
    window_size = 10  # Adjustable range around maximum value to consider for P0
    lower_index = torch.clamp(max_index.unsqueeze(2) - window_size, min=0)
    upper_index = torch.clamp(max_index.unsqueeze(2) + window_size + 1, max=power_spectrum.shape[2])
    P = torch.sum(power_spectrum, dim=2)
    P0 = torch.zeros_like(P)
    for i in range(lower_index.shape[0]):
        for j in range(lower_index.shape[1]):
            P0[i, j] = torch.sum(power_spectrum[i, j, lower_index[i, j, 0]:upper_index[i, j, 0]], dim=0)    
    PSR = P0 / P
    #print(PSR)
    '''

    # combine active features
    features = torch.cat((STD, SAV, SSC), dim=1)

    # z transform
    s = preprocessing.StandardScaler().fit(features)
    return torch.from_numpy(s.transform(features))

def getLabels (combined):
    restim = combined[0][combined[1]]
    return contract(restim)

def getEMG (combined):
    n = combined[0]
    indices = combined[1]
    emg = getData(n)
    return extractFeatures(filters(emg[indices]))

participants = [(i+1) for i in range(10)]
with multiprocessing.Pool(processes=10) as pool:
    restim_async = pool.map_async(getRestim, participants)
    restim = restim_async.get()
    #print("EMG data extracted")

    indices_async = pool.map_async(filterIndices, restim)
    indices = indices_async.get()

    labels_async = pool.map_async(getLabels, [(restim[i], indices[i]) for i in range(10)])
    labels = labels_async.get()

    emg_async = pool.map_async(getEMG, [(participants[i], indices[i]) for i in range(10)])
    emg = emg_async.get()

if leaveOut != 0:
    X_test = emg.pop(leaveOut-1)
    Y_test = labels.pop(leaveOut-1)
    X_train = np.concatenate(emg, axis=0)
    Y_train = np.concatenate(labels, axis=0)
else:
    X_train, X_test, Y_train, Y_test =  model_selection.train_test_split(np.concatenate([np.array(x) for x in emg]), np.concatenate([np.array(x) for x in labels]), test_size=0.2)
    X_test = torch.from_numpy(X_test).to(torch.float32)
    Y_test = torch.from_numpy(Y_test).to(torch.float32)

X_train = torch.from_numpy(X_train).to(torch.float32)
Y_train = torch.from_numpy(Y_train).to(torch.float32)
print(X_train.size())
print(Y_train.size())
print(X_test.size())
print(Y_test.size())

if (classifier.upper() == "RF"):
    # 95.02% baseline accuracy
    #model = RandomForestClassifier(n_estimators=25)
    model = RandomForestClassifier(n_estimators=25)
elif (classifier.upper() == "SVM"):
    # C=100
    model = SVC(C=100)
    temp_train = [torch.argmax(pos).item() for pos in Y_train]
    temp_test = [torch.argmax(pos).item() for pos in Y_test]
    Y_train = torch.from_numpy(np.array(temp_train))
    Y_test = torch.from_numpy(np.array(temp_test))
else:
    print("classifier not found")

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")