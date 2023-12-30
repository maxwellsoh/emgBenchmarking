import torch
import numpy as np
import pandas as pd
import random
from scipy.signal import butter, filtfilt, iirnotch
import torchvision.transforms as transforms
import multiprocessing
from torch.utils.data import DataLoader, Dataset
import matplotlib as mpl
from math import ceil
import argparse

numGestures = 18
wLen = 250 #ms
stepLen = 10 #50 ms
cmap = mpl.colormaps['viridis']

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
    labels = labels.new_zeros(size=(len(R), numGestures))
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

def getRestim (n):
    restim = torch.tensor((pd.read_csv('./NinaproDB5/s' + str(n) + '/restimulusS' + str(n) + '_E2.csv')).to_numpy(), dtype=torch.float32)
    return restim.unfold(dimension=0, size=int(wLen / 5), step=stepLen)

def getEMG (n):
    restim = getRestim(n)
    emg = torch.tensor(((pd.read_csv('./NinaproDB5/s' + str(n) + '/emgS' + str(n) + '_E2.csv')).to_numpy()), dtype=torch.float32)
    return filter(emg.unfold(dimension=0, size=int(wLen / 5), step=10)[balance(restim)])

def getLabels (n):
    restim = getRestim(n)
    return contract(restim[balance(restim)])
def makeOneImage(args):
    data, cmap, length, width = args
    data = data - min(data)
    data = data / max(data)
    data = torch.from_numpy(data).view(length, width).to(torch.float32)
    
    imageL = np.zeros((3, length, width//2))
    imageR = np.zeros((3, length, width//2))
    for p in range (length):
        for q in range (width//2):
            imageL[:, p, q] = (cmap(float(data[p][q])))[:3]
            imageR[:, p, q] = (cmap(float(data[p][q+width//2])))[:3]
    
    imageL = transforms.Resize([96, 112], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(torch.from_numpy(imageL))
    imageR = transforms.Resize([96, 112], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(torch.from_numpy(imageR))
    imageL = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(imageL)
    imageR = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(imageR)
    
    return torch.cat([imageL, imageR], dim=2).numpy().astype(np.float32)

def getImages(emg, standardScaler, length, width):
    emg = standardScaler.transform(np.array(emg.view(len(emg), length*width)))

    with multiprocessing.Pool() as pool:
        args = [(emg[i], cmap, length, width) for i in range(len(emg))]
        images_async = pool.map_async(makeOneImage, args)
        images = images_async.get()
    
    return images

def periodLengthForAnnealing(num_epochs, annealing_multiplier, cycles):
    periodLength = 0
    for i in range(cycles):
        periodLength += annealing_multiplier ** i
    periodLength = num_epochs / periodLength
    
    return ceil(periodLength)

class Data(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)