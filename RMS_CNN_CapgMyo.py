import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from scipy.signal import butter,filtfilt
#import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import h5py
from random import gauss
import math
import multiprocessing
import time
import gc
from itertools import chain

# 0 for no LOSO; participants here are 1-13
leaveOut = 1

# root mean square instances per channel per image
#numRMS = 500 # must be a factor of 250
numRMS = 10

# image width - must be multiple of 128
width = 128
#width = 256

# gaussian Noise signal-to-noise ratio
SNR = 15

# magnitude warping std
std = 0.05

wLen = 250 #ms
stepLen = 50 #ms
freq = 2000 #Hz

