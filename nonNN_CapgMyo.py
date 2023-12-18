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
#import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import io

leaveOut = 0
# SVM or RF (random forest)
classifier = "SVM"

# windowing
wLen = 250 #ms
def window (e):
    return e.unfold(dimension=0, size=wLen, step=50)

#mat to tensor
def getData (subject, gesture, trial):
    sub = str(subject)
    if subject < 10:
        sub = '0' + str(subject)
    name = '0' + sub + '-00' + str(gesture) + '-00' +str(trial)
    if trial == 10:
        name = '0' + sub + '-00' + str(gesture) + '-010'
    mat_data = io.loadmat('./CapgMyo_B/dbb-preprocessed-0' + sub + '/' + name + '.mat')
    mat_array = mat_data['data']
    tensor_data = torch.from_numpy(mat_array)
    if trial < 10:
        return torch.cat((window(tensor_data), getData(subject, gesture, trial + 1)), dim=0)
    elif gesture < 8:
        return torch.cat((window(tensor_data), getData(subject, gesture + 1, 1)), dim=0)
    else:
        return window(tensor_data)

def getEMG (n):
    #return extractFeatures(torch.cat((getData(n*2 - 1, 1, 1), getData(n*2, 1, 1)), dim=0))
    return torch.cat((getData(n*2 - 1, 1, 1), getData(n*2, 1, 1)), dim=0)

def makeLabels (emg):
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(len(emg), 8))
    for x in range (8):
        for y in range (int(len(emg) / 8)):
            labels[int(x * (len(emg) / 8) + y)][x] = 1.0
    return labels

numFeatures = 2
def extractFeatures (emg):
    '''
    #slope sign change
    threshold = 0.1  # volts (should range between 0.00005 (50 microV) and 0.1 (100 mV))
    differences = emg[:, :, 1:] - emg[:, :, :-1]
    sign_changes = torch.logical_and(differences[:, :, :-1] * differences[:, :, 1:] < 0, 
                                     torch.abs(differences[:, :, 1:]) > threshold)
    SSC = torch.sum(sign_changes, dim=2) 

    #mean absolute value
    MAV = torch.mean(torch.abs(emg), dim=2)
    '''

    # sum of absolute values
    SAV = torch.sum(torch.abs(emg), dim=2)

    # standard deviation of fourier transform
    # from section 3.2 of https://www.sciencedirect.com/science/article/pii/S0925231220303283?via%3Dihub#sec0006
    emgFFT = torch.abs(torch.fft.fft(emg, dim=2))
    STD = torch.std(emgFFT, dim=2)

    '''
    # root mean squared
    RMS = torch.sqrt(torch.mean(emg ** 2, dim=2))

    # waveform length
    wl = torch.sum(torch.abs(emg[:, :, 1:] - emg[:, :, :-1]), dim=2)

    # mean frequency
    power_spectrum = torch.abs(torch.fft.fft(emg, dim=2)) ** 2
    sampling_frequency = 1000  # sampling frequency in Hz
    frequency = torch.fft.fftfreq(emg.shape[2], d=1/sampling_frequency)
    frequency = frequency.unsqueeze(0).unsqueeze(1)
    sum_product = torch.sum(power_spectrum * frequency, dim=2)
    total_sum = torch.sum(power_spectrum, dim=2)
    mean_freq = sum_product / total_sum
    print(mean_freq.size())
    '''

    # combine active features
    features = torch.cat((STD, SAV), dim=1)

    # z transform
    s = preprocessing.StandardScaler().fit(features)
    return torch.from_numpy(s.transform(features))

participants = [(i+1) for i in range(10)]
with multiprocessing.Pool(processes=10) as pool:
    emg_async = pool.map_async(getEMG, participants)
    emg = emg_async.get()
    print("EMG data extracted")

    labels_async = pool.map_async(makeLabels, emg)
    labels = labels_async.get()

    emg_async = pool.map_async(extractFeatures, emg)
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
'''
S1_emg = torch.cat((getData(1, 1, 1), getData(2, 1, 1)), dim=0)
S1_labels = makeLabels(S1_emg)
S2_emg = torch.cat((getData(3, 1, 1), getData(4, 1, 1)), dim=0)
S2_labels = makeLabels(S2_emg)
S3_emg = torch.cat((getData(5, 1, 1), getData(6, 1, 1)), dim=0)
S3_labels = makeLabels(S3_emg)
S4_emg = torch.cat((getData(7, 1, 1), getData(8, 1, 1)), dim=0)
S4_labels = makeLabels(S4_emg)
S5_emg = torch.cat((getData(9, 1, 1), getData(10, 1, 1)), dim=0)
S5_labels = makeLabels(S5_emg)
S6_emg = torch.cat((getData(11, 1, 1), getData(12, 1, 1)), dim=0)
S6_labels = makeLabels(S6_emg)
S7_emg = torch.cat((getData(13, 1, 1), getData(14, 1, 1)), dim=0)
S7_labels = makeLabels(S7_emg)
S8_emg = torch.cat((getData(15, 1, 1), getData(16, 1, 1)), dim=0)
S8_labels = makeLabels(S8_emg)
S9_emg = torch.cat((getData(17, 1, 1), getData(18, 1, 1)), dim=0)
S9_labels = makeLabels(S9_emg)
S10_emg = torch.cat((getData(19, 1, 1), getData(20, 1, 1)), dim=0)
S10_labels = makeLabels(S10_emg)

leaveOut = True
outEMG = torch.tensor(())
outLabels = torch.tensor(())
if leaveOut:
    outEMG = S8_emg
    outLabels = S8_labels
#print(outEMG.size())

emg = torch.cat((S1_emg, S2_emg, S3_emg, S4_emg, S5_emg, S6_emg, 
                 S7_emg, S9_emg, S10_emg), dim=0)
labels = torch.cat((S1_labels, S2_labels, S3_labels, S4_labels, 
                    S5_labels, S6_labels, S7_labels, S9_labels, S10_labels), dim=0)
#print(emg.size())

numFeatures = 3
def extractFeatures (emg):
    #slope sign change
    threshold = 0.1  # volts (should range between 0.00005 (50 microV) and 0.1 (100 mV))
    differences = emg[:, :, 1:] - emg[:, :, :-1]
    sign_changes = torch.logical_and(differences[:, :, :-1] * differences[:, :, 1:] < 0, 
                                     torch.abs(differences[:, :, 1:]) > threshold)
    SSC = torch.sum(sign_changes, dim=2)

    # sum of absolute values
    SAV = torch.sum(torch.abs(emg), dim=2)

    #mean absolute value
    MAV = torch.mean(torch.abs(emg), dim=2)

    # standard deviation of fourier transform
    # from section 3.2 of https://www.sciencedirect.com/science/article/pii/S0925231220303283?via%3Dihub#sec0006
    emgFFT = torch.abs(torch.fft.fft(emg, dim=2))
    STD = torch.std(emgFFT, dim=2)

    # root mean squared
    RMS = torch.sqrt(torch.mean(emg ** 2, dim=2))

    # waveform length
    wl = torch.sum(torch.abs(emg[:, :, 1:] - emg[:, :, :-1]), dim=2)

    # mean frequency
    power_spectrum = torch.abs(torch.fft.fft(emg, dim=2)) ** 2
    sampling_frequency = 1000  # sampling frequency in Hz
    frequency = torch.fft.fftfreq(emg.shape[2], d=1/sampling_frequency)
    frequency = frequency.unsqueeze(0).unsqueeze(1)
    sum_product = torch.sum(power_spectrum * frequency, dim=2)
    total_sum = torch.sum(power_spectrum, dim=2)
    mean_freq = sum_product / total_sum
    print(mean_freq.size())

    # combine active features
    features = torch.cat((STD, SAV, SSC), dim=1)

    # z transform
    s = preprocessing.StandardScaler().fit(features)
    return torch.from_numpy(s.transform(features))

# combine features and labels
combined = torch.cat((extractFeatures(emg), labels), dim=1)
#leaveOut
combinedOut = torch.tensor(())
if leaveOut:
    combinedOut = torch.cat((extractFeatures(outEMG), outLabels), dim=1)

# split into training, validation, and test set
train, validation = model_selection.train_test_split(combined.numpy(), test_size=0.2)
test = torch.tensor(())
if leaveOut:
    test = combinedOut.to(torch.float32)
else:
    train, test = model_selection.train_test_split(train, test_size=0.2)
    test = torch.from_numpy(test).to(torch.float32)

train = torch.from_numpy(train).to(torch.float32)
validation = torch.from_numpy(validation).to(torch.float32)
'''