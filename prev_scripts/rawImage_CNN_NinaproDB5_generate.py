import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pandas as pd
from sklearn import model_selection
from scipy.signal import butter,filtfilt,iirnotch
from PyEMD import EMD
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

#get emg and restimulus as tensors (E2 files only)
emg_S1 = torch.tensor(((pd.read_csv('./NinaproDB5/s1/emgS1_E2.csv')).to_numpy()), dtype=torch.float32)
restimulus_S1 = torch.tensor((pd.read_csv('./NinaproDB5/s1/restimulusS1_E2.csv')).to_numpy(), dtype=torch.float32)
emg_S2 = torch.tensor(((pd.read_csv('./NinaproDB5/s2/emgS2_E2.csv')).to_numpy()), dtype=torch.float32)
restimulus_S2 = torch.tensor((pd.read_csv('./NinaproDB5/s2/restimulusS2_E2.csv')).to_numpy(), dtype=torch.float32)
emg_S3 = torch.tensor(((pd.read_csv('./NinaproDB5/s3/emgS3_E2.csv')).to_numpy()), dtype=torch.float32)
restimulus_S3 = torch.tensor((pd.read_csv('./NinaproDB5/s3/restimulusS3_E2.csv')).to_numpy(), dtype=torch.float32)
emg_S4 = torch.tensor(((pd.read_csv('./NinaproDB5/s4/emgS4_E2.csv')).to_numpy()), dtype=torch.float32)
restimulus_S4 = torch.tensor((pd.read_csv('./NinaproDB5/s4/restimulusS4_E2.csv')).to_numpy(), dtype=torch.float32)
emg_S5 = torch.tensor(((pd.read_csv('./NinaproDB5/s5/emgS5_E2.csv')).to_numpy()), dtype=torch.float32)
restimulus_S5 = torch.tensor((pd.read_csv('./NinaproDB5/s5/restimulusS5_E2.csv')).to_numpy(), dtype=torch.float32)
emg_S6 = torch.tensor(((pd.read_csv('./NinaproDB5/s6/emgS6_E2.csv')).to_numpy()), dtype=torch.float32)
restimulus_S6 = torch.tensor((pd.read_csv('./NinaproDB5/s6/restimulusS6_E2.csv')).to_numpy(), dtype=torch.float32)
emg_S7 = torch.tensor(((pd.read_csv('./NinaproDB5/s7/emgS7_E2.csv')).to_numpy()), dtype=torch.float32)
restimulus_S7 = torch.tensor((pd.read_csv('./NinaproDB5/s7/restimulusS7_E2.csv')).to_numpy(), dtype=torch.float32)
emg_S8 = torch.tensor(((pd.read_csv('./NinaproDB5/s8/emgS8_E2.csv')).to_numpy()), dtype=torch.float32)
restimulus_S8 = torch.tensor((pd.read_csv('./NinaproDB5/s8/restimulusS8_E2.csv')).to_numpy(), dtype=torch.float32)
emg_S9 = torch.tensor(((pd.read_csv('./NinaproDB5/s9/emgS9_E2.csv')).to_numpy()), dtype=torch.float32)
restimulus_S9 = torch.tensor((pd.read_csv('./NinaproDB5/s9/restimulusS9_E2.csv')).to_numpy(), dtype=torch.float32)
emg_S10 = torch.tensor(((pd.read_csv('./NinaproDB5/s10/emgS10_E2.csv')).to_numpy()), dtype=torch.float32)
restimulus_S10 = torch.tensor((pd.read_csv('./NinaproDB5/s10/restimulusS10_E2.csv')).to_numpy(), dtype=torch.float32)

# windowing
#wLen = 250 #ms
wLen = 500 #ms
#stepLen = int(wLen/25) 
stepLen = 10 #50 ms
def balance (restimulus):
    numZero = 0
    indices = []
    #print(restimulus.shape)
    for x in range (len(restimulus)):
        L = torch.chunk(restimulus[x], 2, dim=1)
        if torch.equal(L[0], L[1]):
            if L[0][0][0] == 0:
                if (numZero < 380):
                    #print("working")
                    indices += [x]
                numZero += 1
            else:
                indices += [x]
    return indices

def contract(R):
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(len(R), 18))
    for x in range(len(R)):
        labels[x][int(R[x][0][0])] = 1.0
    return labels

def filter(emg):
    # sixth-order Butterworth highpass filter
    b, a = butter(N=3, Wn=5, btype='highpass', analog=False, fs=200.0)
    emgButter = torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())

    #second-order notch filter at 50 Hz
    b, a = iirnotch(w0=50.0, Q=0.0001, fs=200.0)
    return torch.from_numpy(np.flip(filtfilt(b, a, emgButter),axis=0).copy())

restimulus_S1 = restimulus_S1.unfold(dimension=0, size=int(wLen / 5), step=stepLen)
restimulus_S2 = restimulus_S2.unfold(dimension=0, size=int(wLen / 5), step=stepLen)
restimulus_S3 = restimulus_S3.unfold(dimension=0, size=int(wLen / 5), step=stepLen)
restimulus_S4 = restimulus_S4.unfold(dimension=0, size=int(wLen / 5), step=stepLen)
restimulus_S5 = restimulus_S5.unfold(dimension=0, size=int(wLen / 5), step=stepLen)
restimulus_S6 = restimulus_S6.unfold(dimension=0, size=int(wLen / 5), step=stepLen)
restimulus_S7 = restimulus_S7.unfold(dimension=0, size=int(wLen / 5), step=stepLen)
restimulus_S8 = restimulus_S8.unfold(dimension=0, size=int(wLen / 5), step=stepLen)
restimulus_S9 = restimulus_S9.unfold(dimension=0, size=int(wLen / 5), step=stepLen)
restimulus_S10 = restimulus_S10.unfold(dimension=0, size=int(wLen / 5), step=stepLen)

emg_S1 = filter(emg_S1.unfold(dimension=0, size=int(wLen / 5), step=10)[balance(restimulus_S1)])
emg_S2 = filter(emg_S2.unfold(dimension=0, size=int(wLen / 5), step=10)[balance(restimulus_S2)])
emg_S3 = filter(emg_S3.unfold(dimension=0, size=int(wLen / 5), step=10)[balance(restimulus_S3)])
emg_S4 = filter(emg_S4.unfold(dimension=0, size=int(wLen / 5), step=10)[balance(restimulus_S4)])
emg_S5 = filter(emg_S5.unfold(dimension=0, size=int(wLen / 5), step=10)[balance(restimulus_S5)])
emg_S6 = filter(emg_S6.unfold(dimension=0, size=int(wLen / 5), step=10)[balance(restimulus_S6)])
emg_S7 = filter(emg_S7.unfold(dimension=0, size=int(wLen / 5), step=10)[balance(restimulus_S7)])
emg_S8 = filter(emg_S8.unfold(dimension=0, size=int(wLen / 5), step=10)[balance(restimulus_S8)])
emg_S9 = filter(emg_S9.unfold(dimension=0, size=int(wLen / 5), step=10)[balance(restimulus_S9)])
emg_S10 = filter(emg_S10.unfold(dimension=0, size=int(wLen / 5), step=10)[balance(restimulus_S10)])
labels_S1 = contract(restimulus_S1[balance(restimulus_S1)])
labels_S2 = contract(restimulus_S2[balance(restimulus_S2)])
labels_S3 = contract(restimulus_S3[balance(restimulus_S3)])
labels_S4 = contract(restimulus_S4[balance(restimulus_S4)])
labels_S5 = contract(restimulus_S5[balance(restimulus_S5)])
labels_S6 = contract(restimulus_S6[balance(restimulus_S6)])
labels_S7 = contract(restimulus_S7[balance(restimulus_S7)])
labels_S8 = contract(restimulus_S8[balance(restimulus_S8)])
labels_S9 = contract(restimulus_S9[balance(restimulus_S9)])
labels_S10 = contract(restimulus_S10[balance(restimulus_S10)])
emg = [emg_S1, emg_S2, emg_S3, emg_S4, emg_S5, emg_S6, emg_S7, emg_S8, emg_S9, emg_S10]
labels = [labels_S1, labels_S2, labels_S3, labels_S4, labels_S5, labels_S6, labels_S7, labels_S8, labels_S9, labels_S10]

cmap = mpl.colormaps['jet']

def getImages (emg):
    allImages = []
    #emg = np.log(np.abs(emg.numpy()) + 1)
    emg = np.log(np.log(np.abs(emg.numpy())+1))
    for i in range (len(emg)):
    #for i in range (5):
        data = (emg[i])/1.5
        image = np.zeros((3, len(data), len(data[0])))
        for p in range (len(data)):
            for q in range (len(data[p])):
                image[:, p, q] = cmap(data[p][q])[:3]

        image = transforms.Resize([224, 224], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(torch.from_numpy(image))
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        allImages.append(image.numpy().astype(np.float32))

        if i % 1000 == 0:
            print("progress: " + str(i) + "/" + str(len(emg)))
            #print(labels[0][i])
        #if (i % 5000 == 4999):
            #plt.imshow(allImages[i].T, origin='lower')
            #plt.axis('off')
            #plt.show()
    return allImages

for i in range (10):
#i = 1
    file_path = "./NinaproDB5/newImages_" + str(i+1) + ".npy"
    data = getImages(emg[i])
    np.save(file_path, data)
    file_path = "./NinaproDB5/newLabel_" + str(i+1) + ".npy"
    np.save(file_path, labels[i])

'''
loc = 1
image = data[loc].T
plt.imshow(image, origin='lower')
plt.axis('off')
plt.show()
'''