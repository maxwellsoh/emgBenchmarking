import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.signal import butter,filtfilt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import multiprocessing

leaveOut = 0
# SVM or RF (random forest)
classifier = "SVM"

wLen = 250 #ms
stepLen = 50 #ms
freq = 4000 #Hz

def filter (emg):
    b, a = butter(N=1, Wn=120.0, btype='highpass', analog=False, fs=freq)
    return torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=-1).copy())

numFeatures = 2
def extractFeatures (emg):
    #slope sign change
    threshold = 0.01  # volts (should range between 0.00005 (50 microV) and 0.1 (100 mV))
    differences = emg[:, :, 1:] - emg[:, :, :-1]
    sign_changes = torch.logical_and(differences[:, :, :-1] * differences[:, :, 1:] < 0, torch.abs(differences[:, :, 1:]) > threshold)
    SSC = torch.sum(sign_changes, dim=2)
    
    # sum of absolute values
    SAV = torch.sum(torch.abs(emg), dim=2)

    # standard deviation of fourier transform
    emgFFT = torch.abs(torch.fft.fft(emg, dim=2))
    STD = torch.std(emgFFT, dim=2)

    # combine active features
    features = torch.cat((STD, SAV), dim=1)
    #features = torch.cat((STD, SAV, SSC), dim=1)

    # z transform
    s = preprocessing.StandardScaler().fit(features)
    return s.transform(features)

def getData(file, gesture):
    data = filter(torch.from_numpy(np.array(file[gesture])).unfold(dimension=-1, size=int(wLen/1000*freq), step=int(stepLen/1000*freq)))
    return torch.cat([x for x in data], axis=1).permute([1, 0, 2])

gestures = ['abduct_p1', 'adduct_p1', 'extend_p1', 'grip_p1', 'pronate_p1', 'rest_p1', 'supinate_p1', 'tripod_p1', 'wextend_p1', 'wflex_p1']

def getEMG(n):
    if (n<10):
        f = h5py.File('./Jehan_Dataset/p00' + str(n) +'/data_allchannels_initial.h5', 'r')
    else:
        f = h5py.File('./Jehan_Dataset/p0' + str(n) +'/data_allchannels_initial.h5', 'r')
    #print(f.keys())
    return extractFeatures(torch.cat([getData(f, x) for x in gestures], axis=0))


def getGestures(n):
    if (n<10):
        file = h5py.File('./Jehan_Dataset/p00' + str(n) +'/data_allchannels_initial.h5', 'r')
    else:
        file = h5py.File('./Jehan_Dataset/p0' + str(n) +'/data_allchannels_initial.h5', 'r')
    
    numGestures = []
    for gesture in gestures:
        data = filter(torch.from_numpy(np.array(file[gesture])).unfold(dimension=-1, size=int(wLen/1000*freq), step=int(stepLen/1000*freq)))
        numGestures += [len(data)]
    return numGestures


participants = [8,9,11,12,13,15,16,17,18,19,20,21,22]
with multiprocessing.Pool(processes=13) as pool:
    emg_async = pool.map_async(getEMG, participants)
    emg = emg_async.get()
    print("EMG data extracted")

    numGestures_async = pool.map_async(getGestures, participants)
    numGestures = numGestures_async.get()



# generating labels

labels = []
windowsPerSample = 36 # change this if wLen or stepLen is changed

for nums in numGestures:
    sub_labels = torch.tensor(()).new_zeros(size=(sum(nums)*windowsPerSample, 10))
    subGestures = [(i * windowsPerSample) for i in nums]
    index = 0
    count = 0

    for i in range(len(sub_labels)):
        sub_labels[i][index] = 1.0
        count += 1
        if (count >= subGestures[index]):
            index += 1
            count = 0
    
    labels += [sub_labels]
labels = list(labels)
print("labels generated")


if leaveOut != 0:
    X_test = emg.pop(leaveOut-1)
    Y_test = labels.pop(leaveOut-1)
    X_train = np.concatenate(emg, axis=0)
    Y_train = np.concatenate(labels, axis=0)
else:
    X_train, X_test, Y_train, Y_test =  model_selection.train_test_split(np.concatenate([np.array(x) for x in emg]), np.concatenate([np.array(x) for x in labels]), test_size=0.2)
    Y_test = torch.from_numpy(Y_test).to(torch.float32)

X_train = torch.from_numpy(X_train).to(torch.float32)
Y_train = torch.from_numpy(Y_train).to(torch.float32)
X_test = torch.from_numpy(X_test).to(torch.float32)
print(X_train.size())
print(Y_train.size())
print(X_test.size())
print(Y_test.size())

if (classifier.upper() == "RF"):
    # 95.02% baseline accuracy
    #model = RandomForestClassifier(n_estimators=25)
    model = RandomForestClassifier(n_estimators=25)
elif (classifier.upper() == "SVM"):
    # fix this; has low baseline accuracy right now (47.59%)
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