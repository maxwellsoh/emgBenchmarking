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
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map  # Use process_map from tqdm.contrib
import os
from poly5_reader import Poly5Reader
import mne

numGestures = 12
fs = 2000.0 # Hz 
wLen = 250.0 # ms
wLenTimesteps = int(wLen / 1000 * fs)
stepLen = int(50.0 / 1000 * fs) # 50 ms
numElectrodes = 256
num_subjects = 1
cmap = mpl.colormaps['viridis']
# labels for feb data
gesture_labels_02 = ["Pinky Extension", "Wrist Extension", "Wrist Flexion", "Middle Extension", 
                    "Pinky Flexion", "Middle Flexion", "Index Extension", "Index Flexion",
                    "Ring Flexion", "Thumb Extension", "Ring Extension", "Thumb Flexion"]
# all gestures here (for april data)
gesture_labels = ["Wrist Flexion", "Ring Extension", "Wrist Extension", "Pinky Extension", 
                    "Thumb Flexion", "Middle Flexion", "Middle Extension", #"Ulnar Deviation", 
                    "Index Extension", #"Radial Deviation", 
                    "Index Flexion", "Thumb Extension",
                    "Ring Flexion", "Pinky Flexion"]

gestures_in_common = ["Wrist Flexion", "Ring Extension", "Wrist Extension", "Pinky Extension",
                    "Thumb Flexion", "Middle Flexion", "Middle Extension",
                    "Index Extension", "Index Flexion", "Thumb Extension",
                    "Ring Flexion", "Pinky Flexion"]

# 4 kHz sampling rate for april 1 and 2; 2 kHz sampling rate for all other groups
april_1 = ["A_FLX/MCP01_2024_04_12_A_FLX_2.poly5",
"A_FLX/MCP01_2024_04_12_A_FLX_3.poly5",
"A_FLX/MCP01_2024_04_12_A_FLX_4.poly5",
"A_FLX/MCP01_2024_04_12_A_FLX_5.poly5",
"A_FLX/MCP01_2024_04_12_A_FLX_6.poly5",
"A_FLX/MCP01_2024_04_12_A_FLX_7.poly5",
"A_FLX/MCP01_2024_04_12_A_FLX_8.poly5",
# "A_FLX/MCP01_2024_04_12_A_FLX_9.poly5",
"A_FLX/MCP01_2024_04_12_A_FLX_10.poly5",
# "A_FLX/MCP01_2024_04_12_A_FLX_11.poly5",
"A_FLX/MCP01_2024_04_12_A_FLX_12.poly5",
"A_FLX/MCP01_2024_04_12_A_FLX_13.poly5",
"A_FLX/MCP01_2024_04_12_A_FLX_14.poly5",
"A_FLX/MCP01_2024_04_12_A_FLX_15.poly5"]

april_2 = ["B_EXT/MCP01_2024_04_12_B_EXT_2.poly5",
"B_EXT/MCP01_2024_04_12_B_EXT_3.poly5",
"B_EXT/MCP01_2024_04_12_B_EXT_4.poly5",
"B_EXT/MCP01_2024_04_12_B_EXT_5.poly5",
"B_EXT/MCP01_2024_04_12_B_EXT_6.poly5",
"B_EXT/MCP01_2024_04_12_B_EXT_7.poly5",
"B_EXT/MCP01_2024_04_12_B_EXT_8.poly5",
# "B_EXT/MCP01_2024_04_12_B_EXT_9.poly5",
"B_EXT/MCP01_2024_04_12_B_EXT_10.poly5",
# "B_EXT/MCP01_2024_04_12_B_EXT_11.poly5",
"B_EXT/MCP01_2024_04_12_B_EXT_12.poly5",
"B_EXT/MCP01_2024_04_12_B_EXT_13.poly5",
"B_EXT/MCP01_2024_04_12_B_EXT_14.poly5",
"B_EXT/MCP01_2024_04_12_B_EXT_15.poly5"]

april_3 = ["Gestures GUI/Wrist Flexion/1712945596.6765876_dev1_-20240412_141316.poly5",
"Gestures GUI/Ring Extension/1712945730.9449246_dev1_-20240412_141530.poly5",
"Gestures GUI/Wrist Extension/1712945861.4286742_dev1_-20240412_141741.poly5",
"Gestures GUI/Pinky Extension/1712945980.8491514_dev1_-20240412_141940.poly5",
"Gestures GUI/Thumb Flexion/1712946162.160045_dev1_-20240412_142242.poly5",
"Gestures GUI/Middle Flexion/1712946292.6193693_dev1_-20240412_142452.poly5",
"Gestures GUI/Middle Extension/1712946480.0932148_dev1_-20240412_142800.poly5",
# "Gestures GUI/Ulnar Deviation/1712946600.397335_dev1_-20240412_143000.poly5",
"Gestures GUI/Index Extension/1712946730.8848639_dev1_-20240412_143210.poly5",
# "Gestures GUI/Radial Deviation/1712946856.289373_dev1_-20240412_143416.poly5",
"Gestures GUI/Index Flexion/1712946974.4568_dev1_-20240412_143614.poly5",
"Gestures GUI/Thumb Extension/1712947125.3284779_dev1_-20240412_143845.poly5",
"Gestures GUI/Ring Flexion/1712947246.0819142_dev1_-20240412_144046.poly5",
"Gestures GUI/Pinky Flexion/1712947367.0070963_dev1_-20240412_144247.poly5"]

april_4 = ["Gestures GUI/Wrist Flexion/1712945596.6765876_dev2_-20240412_141316.poly5",
"Gestures GUI/Ring Extension/1712945730.9449246_dev2_-20240412_141531.poly5",
"Gestures GUI/Wrist Extension/1712945861.4286742_dev2_-20240412_141741.poly5",
"Gestures GUI/Pinky Extension/1712945980.8491514_dev2_-20240412_141940.poly5",
"Gestures GUI/Thumb Flexion/1712946162.160045_dev2_-20240412_142242.poly5",
"Gestures GUI/Middle Flexion/1712946292.6193693_dev2_-20240412_142452.poly5",
"Gestures GUI/Middle Extension/1712946480.0932148_dev2_-20240412_142800.poly5",
# "Gestures GUI/Ulnar Deviation/1712946600.397335_dev2_-20240412_143000.poly5",
"Gestures GUI/Index Extension/1712946730.8848639_dev2_-20240412_143210.poly5",
# "Gestures GUI/Radial Deviation/1712946856.289373_dev2_-20240412_143416.poly5",
"Gestures GUI/Index Flexion/1712946974.4568_dev2_-20240412_143614.poly5",
"Gestures GUI/Thumb Extension/1712947125.3284779_dev2_-20240412_143845.poly5",
"Gestures GUI/Ring Flexion/1712947246.0819142_dev2_-20240412_144046.poly5",
"Gestures GUI/Pinky Flexion/1712947367.0070963_dev2_-20240412_144247.poly5"]


feb_1 = ["TMSi Saga 1 Flx Proximal/Pinky Extension/1708456044.613826_dev1_-20240220_140724.poly5",
"TMSi Saga 1 Flx Proximal/Wrist Extension/1708456142.5931377_dev1_-20240220_140902.poly5",
"TMSi Saga 1 Flx Proximal/Wrist Flexion/1708456230.4484155_dev1_-20240220_141030.poly5",
"TMSi Saga 1 Flx Proximal/Middle Extension/1708456320.5650508_dev1_-20240220_141200.poly5",
"TMSi Saga 1 Flx Proximal/Pinky Flexion/1708456398.2925904_dev1_-20240220_141318.poly5",
"TMSi Saga 1 Flx Proximal/Middle Flexion/1708456480.2945535_dev1_-20240220_141440.poly5",
"TMSi Saga 1 Flx Proximal/Index Extension/1708456570.5487607_dev1_-20240220_141610.poly5",
"TMSi Saga 1 Flx Proximal/Index Flexion/1708456661.0879076_dev1_-20240220_141741.poly5",
"TMSi Saga 1 Flx Proximal/Ring Flexion/1708456744.4323034_dev1_-20240220_141904.poly5",
"TMSi Saga 1 Flx Proximal/Thumb Extension/1708456837.9119933_dev1_-20240220_142037.poly5",
"TMSi Saga 1 Flx Proximal/Ring Extension/1708456927.8494475_dev1_-20240220_142207.poly5",
"TMSi Saga 1 Flx Proximal/Thumb Flexion/1708457024.8391547_dev1_-20240220_142344.poly5"]

feb_2 = ["TMSi Saga 5 Ext Proximal/Pinky Extension/1708456044.613826_dev2_-20240220_140724.poly5",
"TMSi Saga 5 Ext Proximal/Wrist Extension/1708456142.5931377_dev2_-20240220_140902.poly5",
"TMSi Saga 5 Ext Proximal/Wrist Flexion/1708456230.4484155_dev2_-20240220_141030.poly5",
"TMSi Saga 5 Ext Proximal/Middle Extension/1708456320.5650508_dev2_-20240220_141200.poly5",
"TMSi Saga 5 Ext Proximal/Pinky Flexion/1708456398.2925904_dev2_-20240220_141318.poly5",
"TMSi Saga 5 Ext Proximal/Middle Flexion/1708456480.2945535_dev2_-20240220_141440.poly5",
"TMSi Saga 5 Ext Proximal/Index Extension/1708456570.5487607_dev2_-20240220_141610.poly5",
"TMSi Saga 5 Ext Proximal/Index Flexion/1708456661.0879076_dev2_-20240220_141741.poly5",
"TMSi Saga 5 Ext Proximal/Ring Flexion/1708456744.4323034_dev2_-20240220_141904.poly5",
"TMSi Saga 5 Ext Proximal/Thumb Extension/1708456837.9119933_dev2_-20240220_142038.poly5",
"TMSi Saga 5 Ext Proximal/Ring Extension/1708456927.8494475_dev2_-20240220_142207.poly5",
"TMSi Saga 5 Ext Proximal/Thumb Flexion/1708457024.8391547_dev2_-20240220_142345.poly5"]

feb_3 = ["TMSi Saga 3 Flex Distal/20240220_FlxDist_Trl_001-20240220T140714.DATA.Poly5",
"TMSi Saga 3 Flex Distal/20240220_FlxDist_Trl_002-20240220T140849.DATA.Poly5",
"TMSi Saga 3 Flex Distal/20240220_FlxDist_Trl_003-20240220T141021.DATA.Poly5",
"TMSi Saga 3 Flex Distal/20240220_FlxDist_Trl_004-20240220T141152.DATA.Poly5",
"TMSi Saga 3 Flex Distal/20240220_FlxDist_Trl_005-20240220T141310.DATA.Poly5",
"TMSi Saga 3 Flex Distal/20240220_FlxDist_Trl_006-20240220T141430.DATA.Poly5",
"TMSi Saga 3 Flex Distal/20240220_FlxDist_Trl_007-20240220T141601.DATA.Poly5",
"TMSi Saga 3 Flex Distal/20240220_FlxDist_Trl_008-20240220T141730.DATA.Poly5",
"TMSi Saga 3 Flex Distal/20240220_FlxDist_Trl_009-20240220T141856.DATA.Poly5",
"TMSi Saga 3 Flex Distal/20240220_FlxDist_Trl_010-20240220T142029.DATA.Poly5",
"TMSi Saga 3 Flex Distal/20240220_FlxDist_Trl_011-20240220T142158.DATA.Poly5",
"TMSi Saga 3 Flex Distal/20240220_FlxDist_Trl_012-20240220T142333.DATA.Poly5"]

feb_4 = ["TMSi Saga 4 Ext Distal/20240220_ExtDist_Trl_001-20240220T140803.DATA.Poly5",
"TMSi Saga 4 Ext Distal/20240220_ExtDist_Trl_002-20240220T140940.DATA.Poly5",
"TMSi Saga 4 Ext Distal/20240220_ExtDist_Trl_003-20240220T141109.DATA.Poly5",
"TMSi Saga 4 Ext Distal/20240220_ExtDist_Trl_004-20240220T141241.DATA.Poly5",
"TMSi Saga 4 Ext Distal/20240220_ExtDist_Trl_005-20240220T141359.DATA.Poly5",
"TMSi Saga 4 Ext Distal/20240220_ExtDist_Trl_006-20240220T141521.DATA.Poly5",
"TMSi Saga 4 Ext Distal/20240220_ExtDist_Trl_007-20240220T141651.DATA.Poly5",
"TMSi Saga 4 Ext Distal/20240220_ExtDist_Trl_008-20240220T141821.DATA.Poly5",
"TMSi Saga 4 Ext Distal/20240220_ExtDist_Trl_009-20240220T141945.DATA.Poly5",
"TMSi Saga 4 Ext Distal/20240220_ExtDist_Trl_010-20240220T142118.DATA.Poly5",
"TMSi Saga 4 Ext Distal/20240220_ExtDist_Trl_011-20240220T142248.DATA.Poly5",
"TMSi Saga 4 Ext Distal/20240220_ExtDist_Trl_012-20240220T142425.DATA.Poly5"]

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

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

def filter(emg):
    # sixth-order Butterworth highpass filter
    b, a = butter(N=3, Wn=[5.0, 500.0], btype='bandpass', analog=False, fs=fs)
    emgButter = torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())

    #second-order notch filter at 50 Hz
    b, a = iirnotch(w0=50.0, Q=0.0001, fs=fs)
    return torch.from_numpy(np.flip(filtfilt(b, a, emgButter),axis=0).copy())

def getEMG (n):
    return torch.cat((getEMG_separateSessions((1, 1)), getEMG_separateSessions((1, 2))), dim=0)

def getEMG_separateSessions(args):
    subject_number, session = args
    if (session == 1):
        numFiles = 12
        path_start = "SCI/02-20/"
        fileGroups = [feb_1, feb_2, feb_3, feb_4]
    else:
        numFiles = 12
        path_start = "SCI/04-12/"
        fileGroups = [april_1, april_2, april_3, april_4]

    cummulative_emg = []
    for i in range(numFiles):
        data = [Poly5Reader(path_start + file[i]) for file in fileGroups]
        emg = []
        # print("Subject number, session number:", args)
        for d in data:
            info = mne.create_info(d.num_channels, sfreq=d.sample_rate)
            mne_raw = mne.io.RawArray(d.samples, info)
            emg.append(np.array(mne_raw.get_data()))
        
        # subsample APRIL 1 and 2
        if (session == 2):
            emg[0] = emg[0][:, ::2]
            emg[1] = emg[1][:, ::2]

        rep_inits = [[], [], [], []]
        rep_durations = []

        for group in range(4):
            subseq = emg[group][len(emg[group]) - 3]
            total_elapsed = 0
            skipped = False

            while(len(rep_inits[group]) < 10):
                start = np.argmax(subseq)
                subseq = subseq[start:]
                total_elapsed += start

                end = np.argmin(subseq)
                subseq = subseq[end:]

                if (skipped):
                    rep_inits[group].append(total_elapsed)
                    if (group == 0):
                        rep_durations.append(end)
                elif (end < 1000):
                    skipped = True
                
                total_elapsed += end
        
        for j in range(len(rep_durations)):
            # feb 1 and 2 have 1-63 as electrodes (missing last electrode)
            # feb 3 and 4 have 0-63 as electrodes
            if (session == 1):
                missing = np.zeros((1, rep_durations[j]))
                combined = np.concatenate((emg[0][1:64, rep_inits[0][j]:rep_inits[0][j]+rep_durations[j]], missing,
                                        emg[1][1:64, rep_inits[1][j]:rep_inits[1][j]+rep_durations[j]], missing,
                                        emg[2][:64, rep_inits[2][j]:rep_inits[2][j]+rep_durations[j]],
                                        emg[3][:64, rep_inits[3][j]:rep_inits[3][j]+rep_durations[j]]), axis=0)
            # april has 1-64 as electrodes
            else:
                combined = np.concatenate([emg[n][1:65, rep_inits[n][j]:rep_inits[n][j]+rep_durations[j]] for n in range(len(emg))], axis=0)

            combined = filter(torch.from_numpy(combined)).unfold(dimension=-1, size=wLenTimesteps, step=stepLen)
            print("Combined shape:", combined.shape)

            cummulative_emg.append(combined.permute((1, 0, 2)))

    print("Cummulative EMG shape:", torch.cat(cummulative_emg, dim=0).shape)
    return torch.cat(cummulative_emg, dim=0)

def getLabels (n):
    return torch.cat((getLabels_separateSessions((1, 1)), getLabels_separateSessions((1, 2))), dim=0)

def getLabels_separateSessions(args):
    subject_number, session = args
    if (session == 1):
        numFiles = 12
        path_start = "SCI/02-20/"
        fileGroups = [feb_1, feb_2, feb_3, feb_4]
    else:
        numFiles = 12
        path_start = "SCI/04-12/"
        fileGroups = [april_1, april_2, april_3, april_4]

    gesture_reps = [0 for i in range(numFiles)]
    for i in range(numFiles):
        data = [Poly5Reader(path_start + file[i]) for file in fileGroups]
        emg = []
        for d in data:
            info = mne.create_info(d.num_channels, sfreq=d.sample_rate)
            mne_raw = mne.io.RawArray(d.samples, info)
            emg.append(np.array(mne_raw.get_data()))
        
        # subsample APRIL 1 and APRIL 2
        if (session == 2):
            emg[0] = emg[0][:, ::2]
            emg[1] = emg[1][:, ::2]

        rep_durations = []
        subseq = emg[0][len(emg[0]) - 3]
        global_min = np.min(subseq)
        total_elapsed = 0
        skipped = False

        while (len(rep_durations) < 10):
            start = np.argmax(subseq)
            subseq = subseq[start:]

            end = np.argmin(subseq)
            subseq = subseq[end:]

            if (skipped):
                rep_durations.append(end)
            elif (end < 1000):
                skipped = True
        
        for j in range(len(rep_durations)):
            gesture_reps[i] += (rep_durations[j] - wLenTimesteps) // stepLen + 1

        print("Gesture reps:", gesture_reps)

    curr = 0
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(sum(gesture_reps), len(gestures_in_common)))

    if (session == 1):
        for i, ges in enumerate(gesture_labels_02):
            pos = gestures_in_common.index(ges)
            labels[curr:curr+gesture_reps[i], pos] = 1
            curr += gesture_reps[i]
    else:
        for i, ges in enumerate(gesture_labels):
            pos = gestures_in_common.index(ges)
            labels[curr:curr+gesture_reps[i], pos] = 1
            curr += gesture_reps[i]
    
    return labels

def optimized_makeOneCWTImage(data, length, width, resize_length_factor, native_resnet_size):
    emg_sample = data
    # Convert EMG sample to numpy array for CWT computation
    emg_sample_np = emg_sample.astype(np.float16).flatten()
    highest_cwt_scale = 31
    downsample_factor_for_cwt_preprocessing = 1 # used to make image processing tractable
    scales = np.arange(1, highest_cwt_scale)  
    wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet; adjust as needed
    # Perform Continuous Wavelet Transform (CWT)
    # Note: PyWavelets returns scales and coeffs (coefficients)
    coefficients, frequencies = pywt.cwt(emg_sample_np[::downsample_factor_for_cwt_preprocessing], scales, wavelet, sampling_period=1/fs*downsample_factor_for_cwt_preprocessing)
    coefficients_dB = 10 * np.log10(np.abs(coefficients) + 1e-6)  # Adding a small constant to avoid log(0)
    # Convert back to PyTorch tensor and reshape
    emg_sample = torch.tensor(coefficients_dB).float().reshape(-1, coefficients_dB.shape[-1])
    # Normalization
    emg_sample -= torch.min(emg_sample)
    emg_sample /= torch.max(emg_sample) - torch.min(emg_sample)  # Adjusted normalization to avoid divide-by-zero
    blocks = emg_sample.reshape(highest_cwt_scale-1, numElectrodes, -1)
    emg_sample = blocks.transpose(1,0).reshape(numElectrodes*(highest_cwt_scale-1), -1)
        
    # Update 'window_size' if necessary
    window_size = emg_sample.shape[1]

    emg_sample -= torch.min(emg_sample)
    emg_sample /= torch.max(emg_sample)
    data = emg_sample

    data_converted = cmap(data)
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))
    
    resize = transforms.Resize([length * resize_length_factor, native_resnet_size],
                           interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp between 0 and 1 using torch.clamp
    image_clamped = torch.clamp(image_resized, 0, 1)

    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    # Since no split occurs, we don't need to concatenate halves back together
    final_image = image_normalized.numpy().astype(np.float32)

    return final_image

def optimized_makeOneSpectrogramImage(data, length, width, resize_length_factor, native_resnet_size):
    spectrogram_window_size = 64
    emg_sample_unflattened = data.reshape(numElectrodes, -1)
    frequencies, times, Sxx = stft(emg_sample_unflattened, fs=fs, nperseg=spectrogram_window_size, noverlap=spectrogram_window_size-1) # defaults to hann window
    Sxx_dB = 10 * np.log10(np.abs(Sxx) + 1e-6) # small constant added to avoid log(0)
    emg_sample = torch.from_numpy(Sxx_dB)
    emg_sample -= torch.min(emg_sample)
    emg_sample /= torch.max(emg_sample)
    emg_sample = emg_sample.reshape(emg_sample.shape[0]*emg_sample.shape[1], emg_sample.shape[2])
    data = emg_sample

    data_converted = cmap(data)
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))
    
    resize = transforms.Resize([length * resize_length_factor, native_resnet_size],
                           interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp between 0 and 1 using torch.clamp
    image_clamped = torch.clamp(image_resized, 0, 1)

    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    final_image = image_normalized.numpy().astype(np.float32)

    return final_image


def optimized_makeOneMagnitudeImage(data, length, width, resize_length_factor, native_resnet_size, global_min, global_max):
    # Normalize with global min and max
    data = (data - global_min) / (global_max - global_min)
    data_converted = cmap(data)
    rgb_data = data_converted[:, :3]
    image_data = np.reshape(rgb_data, (numElectrodes, width, 3))
    image = np.transpose(image_data, (2, 0, 1))
    
    # Split image and resize
    imageL, imageR = np.split(image, 2, axis=2)
    resize = transforms.Resize([length * resize_length_factor, native_resnet_size // 2],
                               interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    imageL, imageR = map(lambda img: resize(torch.from_numpy(img)), (imageL, imageR))
    
    # Clamp between 0 and 1 using torch.clamp
    imageL, imageR = map(lambda img: torch.clamp(img, 0, 1), (imageL, imageR))
    
    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imageL, imageR = map(normalize, (imageL, imageR))
    
    return torch.cat([imageL, imageR], dim=2).numpy().astype(np.float32)

def optimized_makeOneImage(data, cmap, length, width, resize_length_factor, native_resnet_size):
    # Contrast normalize and convert data
    data = (data - data.min()) / (data.max() - data.min())
    data_converted = cmap(data)
    rgb_data = data_converted[:, :3]
    image_data = np.reshape(rgb_data, (numElectrodes, width, 3))
    image = np.transpose(image_data, (2, 0, 1))
    
    # Resize the whole image instead of splitting it
    resize = transforms.Resize([length * resize_length_factor, native_resnet_size],
                               interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image = resize(torch.from_numpy(image))
    
    # Get max and min values after interpolation
    max_val = image.max()
    min_val = image.min()
    
    # Contrast normalize again after interpolation
    image = (image - min_val) / (max_val - min_val)
    
    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image)
    
    return image.numpy().astype(np.float32)

def calculate_rms(array_2d):
    # Calculate RMS for 2D array where each row is a window
    return np.sqrt(np.mean(array_2d**2))

def getImages(emg, standardScaler, length, width, turn_on_rms=False, rms_windows=10, turn_on_magnitude=False, 
              global_min=None, global_max=None, turn_on_spectrogram=False, turn_on_cwt=False, turn_on_hht=False):

    if standardScaler is not None:
        emg = standardScaler.transform(np.array(emg.view(len(emg), length*width)))
    else:
        emg = np.array(emg.view(len(emg), length*width))
        
    # Use RMS preprocessing
    if turn_on_rms:
        emg = emg.reshape(len(emg), length, width)
        # Reshape data for RMS calculation: (SAMPLES, 16, 5, 10)
        emg = emg.reshape(len(emg), length, rms_windows, width // rms_windows)
        
        # Apply RMS calculation along the last axis (axis=-1)
        emg_rms = np.apply_along_axis(calculate_rms, -1, emg)
        emg = emg_rms  # Resulting shape will be (SAMPLES, 16, 5)
        width = rms_windows
        emg = emg.reshape(len(emg), length*width)

    # Parameters that don't change can be set once
    resize_length_factor = 1
    native_resnet_size = 224

    images = []
    for i in range(len(emg)):
        images.append(optimized_makeOneImage(emg[i], cmap, length, width, resize_length_factor, native_resnet_size))

    if turn_on_magnitude:
        images_magnitude = []
        for i in range(len(emg)):
            images_magnitude.append(optimized_makeOneMagnitudeImage(emg[i], length, width, resize_length_factor, native_resnet_size, global_min, global_max))
        images = np.concatenate((images, images_magnitude), axis=2)

    elif turn_on_spectrogram:
        images_spectrogram = []
        for i in range(len(emg)):
            images_spectrogram.append(optimized_makeOneSpectrogramImage(emg[i], length, width, resize_length_factor, native_resnet_size))
        images = images_spectrogram
    
    elif turn_on_cwt:
        images_cwt = []
        for i in range(len(emg)):
            images_cwt.append(optimized_makeOneCWTImage(emg[i], length, width, resize_length_factor, native_resnet_size))
        images = images_cwt
        
    elif turn_on_hht:
        NotImplementedError("HHT is not implemented yet")
    
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

def plot_confusion_matrix(true, pred, gesture_labels, testrun_foldername, args, formatted_datetime, partition_name):
    # Calculate confusion matrix
    cf_matrix = confusion_matrix(true, pred)
    df_cm_unnormalized = pd.DataFrame(cf_matrix, index=gesture_labels, columns=gesture_labels)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=gesture_labels,
                        columns=gesture_labels)
    plt.figure(figsize=(12, 7))
    
    # Plot confusion matrix square
    sn.set(font_scale=0.8)
    sn.heatmap(df_cm, annot=True, fmt=".0%", square=True)
    confusionMatrix_filename = f'{testrun_foldername}confusionMatrix_{partition_name}_seed{args.seed}_{formatted_datetime}.png'
    plt.savefig(confusionMatrix_filename)
    df_cm_unnormalized.to_pickle(f'{testrun_foldername}confusionMatrix_{partition_name}_seed{args.seed}_{formatted_datetime}.pkl')
    wandb.log({f"{partition_name} Confusion Matrix": wandb.Image(confusionMatrix_filename),
                f"Raw {partition_name.capitalize()} Confusion Matrix": wandb.Table(dataframe=df_cm_unnormalized)})
    
def denormalize(images):
    # Define mean and std from imageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    # Denormalize
    images = images * std + mean
    
    # Clip the values to ensure they are within [0,1] as expected for image data
    images = torch.clamp(images, 0, 1)
    
    return images


def plot_average_images(image_data, true, gesture_labels, testrun_foldername, args, formatted_datetime, partition_name):
    # Convert true to numpy for quick indexing
    true_np = np.array(true)        

    # Calculate average image of each gesture
    average_images = []
    print(f"Plotting average {partition_name} images...")
    for i in range(numGestures):
        # Find indices
        gesture_indices = np.where(true_np == i)[0]

        # Select and denormalize only the required images
        gesture_images = denormalize(transforms.Resize((224,224))(image_data[gesture_indices])).cpu().detach().numpy()
        average_images.append(np.mean(gesture_images, axis=0))

    average_images = np.array(average_images)

    # Plot average image of each gesture
    fig, axs = plt.subplots(2, 9, figsize=(10, 10))
    for i in range(numGestures):
        axs[i//9, i%9].imshow(average_images[i].transpose(1,2,0))
        axs[i//9, i%9].set_title(gesture_labels[i])
        axs[i//9, i%9].axis('off')
    fig.suptitle('Average Image of Each Gesture')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Log in wandb
    averageImages_filename = f'{testrun_foldername}averageImages_seed{args.seed}_{partition_name}_{formatted_datetime}.png'
    plt.savefig(averageImages_filename, dpi=450)
    wandb.log({f"Average {partition_name.capitalize()} Images": wandb.Image(averageImages_filename)})


def plot_first_fifteen_images(image_data, true, gesture_labels, testrun_foldername, args, formatted_datetime, partition_name):
    # Convert true to numpy for quick indexing
    true_np = np.array(true)

    # Parameters for plotting
    rows_per_gesture = 15
    total_gestures = numGestures  # Replace with the actual number of gestures

    # Create subplots
    fig, axs = plt.subplots(rows_per_gesture, total_gestures, figsize=(20, 20))

    print(f"Plotting first fifteen {partition_name} images...")
    for i in range(total_gestures):
        # Find indices of the first 15 images for gesture i
        gesture_indices = np.where(true_np == i)[0][:rows_per_gesture]
        
        # Select and denormalize only the required images
        gesture_images = denormalize(transforms.Resize((224,224))(image_data[gesture_indices])).cpu().detach().numpy()

        for j in range(len(gesture_images)):  # len(gesture_images) is no more than rows_per_gesture
            ax = axs[j, i]
            # Transpose the image data to match the expected shape (H, W, C) for imshow
            ax.imshow(gesture_images[j].transpose(1, 2, 0))
            if j == 0:
                ax.set_title(gesture_labels[i])
            ax.axis('off')

    fig.suptitle(f'First Fifteen {partition_name.capitalize()} Images of Each Gesture')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and log the figure
    firstThreeImages_filename = f'{testrun_foldername}firstFifteenImages_seed{args.seed}_{partition_name}_{formatted_datetime}.png'
    plt.savefig(firstThreeImages_filename, dpi=300)
    wandb.log({f"First Fifteen {partition_name.capitalize()} Images of Each Gesture": wandb.Image(firstThreeImages_filename)})
