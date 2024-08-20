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
from tqdm import tqdm
import h5py
import os
from scipy.signal import spectrogram, stft
import pywt
import fcwt
import emd 

numGestures = 7
fs = 200.0 #Hz
wLen = 250 # ms
wLenTimesteps = int(wLen / 1000 * fs)
stepLen = 50 #50 ms
stepLen = int(stepLen / 1000 * fs)
numElectrodes = 8
num_subjects = 18
cmap = mpl.colormaps['viridis']
# Gesture Labels
gesture_labels = ["Neutral","Radial Deviation","Wrist Flexion","Ulnar Deviation","Wrist Extension","Hand Close","Hand Open"]

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

def contract(R):
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(len(R), numGestures))
    for x in range(len(R)):
        labels[x][int(R[x]) - 1] = 1.0
    return labels

def filter(emg):
    # sixth-order Butterworth highpass filter
    b, a = butter(N=3, Wn=5, btype='highpass', analog=False, fs=fs)
    emgButter = torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())

    #second-order notch filter at 50 Hz
    b, a = iirnotch(w0=50.0, Q=0.0001, fs=fs)
    return torch.from_numpy(np.flip(filtfilt(b, a, emgButter),axis=0).copy())

# partition data by channel; returns [# samples, # channels]
def format_emg (data):
    """Partition data by channel

    Args:
        data (_type_): _description_

    Returns:
        emg: (SAMPLES, CHANNELS)
    """
    emg = np.zeros((len(data) // numElectrodes, numElectrodes))
    for i in range(len(data) // numElectrodes):
        for j in range(numElectrodes):
            emg[i][j] = data[i * numElectrodes + j]
    return emg

# data is [# samples, # channels]
# target min/max is [# channels, # gestures]
def normalize (data, target_min, target_max, gesture):
    source_min = np.zeros(len(data[0]), dtype=np.float32)
    source_max = np.zeros(len(data[0]), dtype=np.float32)
    for i in range(len(data[0])):
        source_min[i] = np.min(data[:, i])
        source_max[i] = np.max(data[:, i])
    # normalizes each channel's data separately
    for i in range(len(data[0])):
        data[:, i] = ((data[:, i] - source_min[i]) / (source_max[i] 
        - source_min[i])) * (target_max[i][gesture] - target_min[i][gesture]) + target_min[i][gesture]
    return data

def getEMG (args):
    if (type(args) == int):
        n = args
    else:
        n = args[0]
        target_min = args[1]
        target_max = args[2]
        leftout = args[3]

    emg = []
    for i in range(numGestures * 4):
        if (n < 3):
            data = np.fromfile(f'myoarmbanddataset/Female{n-1}/Test1/classe_{i}.dat', dtype=np.int16)
        else:
            data = np.fromfile(f'myoarmbanddataset/Male{n-3}/Test1/classe_{i}.dat', dtype=np.int16)
        data = format_emg(np.array(data, dtype=np.float32))
        if (type(args) != int and leftout != n):
            data = normalize(data, target_min, target_max, i % numGestures)
        emg.append(torch.from_numpy(data).unfold(dimension=0, size=wLenTimesteps, step=stepLen))
    emg = filter(torch.cat(emg, dim=0))
    return emg

def getExtrema (n, proportion):
    """Returns the min/max EMG values for each electrode per gesture over a proportion of the windows of data. 

    Per gesture, accumulates data across each of its repetitions and windows this data. Then takes a proportion of the windows and calculates the min/max values for each electrode over these windows across all trials and time steps. 

    Args:
        n (int): subject number
        proportion: proportion of the windows to consider

    Returns:
        mins, maxes: mins[electrode][gesture] is min value of electrode for gesture across proportion of windows
    """
    mins = np.zeros((numElectrodes, numGestures))
    maxes = np.zeros((numElectrodes, numGestures))

    for i in range(numGestures):
        
        emg = []

        for j in range(4):
            if (n < 3):
                data = np.fromfile(f'myoarmbanddataset/Female{n-1}/Test1/classe_{i + j*numGestures}.dat', dtype=np.int16)
            else:
                data = np.fromfile(f'myoarmbanddataset/Male{n-3}/Test1/classe_{i + j*numGestures}.dat', dtype=np.int16)

            data = format_emg(np.array(data, dtype=np.float32))
            # windowed per repetition (needs to match the windowing in getEMG)
            emg.append(torch.from_numpy(data).unfold(dimension=0, size=wLenTimesteps, step=stepLen)) # (REPETITION, WINDOW, ELECTRODE, TIME STEP) 


        # concatenate across repetitions 
        emg = torch.cat(emg, dim=0) # (WINDOW, ELECTRODE, TIME STEP)

        num_windows = np.round(len(emg)*proportion).astype(int)
        selected_windows = emg[:num_windows]

        for j in range(numElectrodes):
            mins[j][i] = torch.min(selected_windows[:, j, :])
            maxes[j][i] = torch.max(selected_windows[:, j, :])

    return mins, maxes

def getLabels (n):
    labels = []
    for i in range(numGestures * 4):
        if (n < 3):
            data = np.fromfile(f'myoarmbanddataset/Female{n-1}/Test1/classe_{i}.dat', dtype=np.int16)
        else:
            data = np.fromfile(f'myoarmbanddataset/Male{n-3}/Test1/classe_{i}.dat', dtype=np.int16)
        labels.append(torch.from_numpy((i % numGestures) + np.zeros(torch.from_numpy(format_emg(np.array(data, dtype=np.float32))).unfold(dimension=0, size=wLenTimesteps, step=stepLen).shape[0])))
    labels = contract(torch.cat(labels, dim=0))
    return labels

def optimized_makeOneCWTImage(data, length, width, resize_length_factor, native_resnet_size):
    emg_sample = data
    data = data.reshape(length, width)
    # Convert EMG sample to numpy array for CWT computation
    emg_sample_np = data.astype(np.float16)
    highest_cwt_scale = wLenTimesteps
    downsample_factor_for_cwt_preprocessing = 1 # used to make image processing tractable
    scales = np.arange(1, highest_cwt_scale)
    # wavelet = 'morl'
    # Perform Continuous Wavelet Transform (CWT)
    # Note: PyWavelets returns scales and coeffs (coefficients)
    # for i in range(numElectrodes):
    for i in range(length):
        # coefficients, frequencies = pywt.cwt(emg_sample_np[i, ::downsample_factor_for_cwt_preprocessing], scales, wavelet, sampling_period=1/fs*downsample_factor_for_cwt_preprocessing)
        frequencies, coefficients = fcwt.cwt(emg_sample_np[i, ::downsample_factor_for_cwt_preprocessing], int(fs), int(scales[0]), int(scales[-1]), int(highest_cwt_scale))
        # note fcwt.cwt returns frequencies and coefficients with frequencies from most to least
        coefficients_dB = 10 * np.log10(np.abs(coefficients)+1e-12) # Adding a small constant to avoid log(0)
        if i == 0:
            time_frequency_emg = np.zeros((length * coefficients_dB.shape[0], coefficients_dB.shape[1]))
        time_frequency_emg[i*coefficients_dB.shape[0]:(i+1)*coefficients_dB.shape[0], :] = coefficients_dB # flip for low frequency to be at bottom
    # Convert back to PyTorch tensor and reshape
    emg_sample = torch.tensor(time_frequency_emg).float().reshape(-1, time_frequency_emg.shape[-1])
    # Normalization
    emg_sample -= torch.min(emg_sample)
    emg_sample /= torch.max(emg_sample) - torch.min(emg_sample)  # Adjusted normalization to avoid divide-by-zero
    # blocks = emg_sample.reshape(highest_cwt_scale, numElectrodes, -1)
    # emg_sample = blocks.transpose(1,0).reshape(numElectrodes*(highest_cwt_scale), -1)
        
    data = emg_sample

    data_converted = cmap(data)
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))

    resize_length_factor = len(frequencies)
    width_to_transform_to = min(native_resnet_size, time_frequency_emg.shape[-1])
    
    resize = transforms.Resize([length * resize_length_factor, width_to_transform_to],
                           interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp between 0 and 1 using torch.clamp
    image_clamped = torch.clamp(image_resized, 0, 1)

    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    # Since no split occurs, we don't need to concatenate halves back together
    final_image = image_normalized.numpy().astype(np.float16)

    return final_image

def closest_factors(num):
    # Find factors of the number
    factors = [(i, num // i) for i in range(1, int(np.sqrt(num)) + 1) if num % i == 0]
    # Sort factors by their difference, so the closest pair is first
    factors.sort(key=lambda x: abs(x[0] - x[1]))
    return factors[0]

def optimized_makeOneCWTImage(data, length, width, resize_length_factor, native_resnet_size):
    # Reshape and preprocess EMG data
    data = data.reshape(length, width).astype(np.float16)
    highest_cwt_scale = wLenTimesteps
    scales = np.arange(1, highest_cwt_scale)

    # Pre-allocate the array for the CWT coefficients
    grid_width, grid_length = closest_factors(numElectrodes)

    length_to_resize_to = min(native_resnet_size, grid_width * highest_cwt_scale)
    width_to_transform_to = min(native_resnet_size, grid_length * width)

    time_frequency_emg = np.zeros((length * (highest_cwt_scale), width))

    # Perform Continuous Wavelet Transform (CWT)
    for i in range(length):
        frequencies, coefficients = fcwt.cwt(data[i, :], int(fs), int(scales[0]), int(scales[-1]), int(highest_cwt_scale))
        coefficients_abs = np.abs(coefficients) 
        # coefficients_dB = 10 * np.log10(coefficients_abs + 1e-12)  # Avoid log(0)
        time_frequency_emg[i * (highest_cwt_scale):(i + 1) * (highest_cwt_scale), :] = coefficients_abs

    # Convert to PyTorch tensor and normalize
    emg_sample = torch.tensor(time_frequency_emg).float()
    emg_sample = emg_sample.view(numElectrodes, wLenTimesteps, -1)

    # Reshape into blocks
    
    blocks = emg_sample.view(grid_width, grid_length, wLenTimesteps, -1)

    # Combine the blocks into the final image
    rows = [torch.cat([blocks[i, j] for j in range(grid_length)], dim=1) for i in range(grid_width)]
    combined_image = torch.cat(rows, dim=0)

    # Normalize combined image
    combined_image -= torch.min(combined_image)
    combined_image /= torch.max(combined_image) - torch.min(combined_image)

    # Convert to RGB and resize
    data_converted = cmap(combined_image)
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))

    resize = transforms.Resize([length_to_resize_to, width_to_transform_to],
                               interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp and normalize
    image_clamped = torch.clamp(image_resized, 0, 1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    # Return final image as a NumPy array
    final_image = image_normalized.numpy().astype(np.float16)
    return final_image

def optimized_makeOneSpectrogramImage(data, length, width, resize_length_factor,
 native_resnet_size):
    spectrogram_window_size = wLenTimesteps // 4
    emg_sample_unflattened = data.reshape(numElectrodes, -1)
    number_of_frequencies = wLenTimesteps 

    # Pre-allocate the array for the CWT coefficients
    grid_width, grid_length = closest_factors(numElectrodes)

    length_to_resize_to = min(native_resnet_size, grid_width * number_of_frequencies)
    width_to_transform_to = min(native_resnet_size, grid_length * width)
    
    frequencies, times, Sxx = stft(emg_sample_unflattened, fs=fs, nperseg=spectrogram_window_size - 1, noverlap=spectrogram_window_size-2, nfft=number_of_frequencies - 1) # defaults to hann window
    Sxx_abs = np.abs(Sxx) # small constant added to avoid log(0)
    # Sxx_dB = 10 * np.log10(np.abs(Sxx_abs) + 1e-12)
    emg_sample = torch.from_numpy(Sxx_abs)
    emg_sample -= torch.min(emg_sample)
    emg_sample /= torch.max(emg_sample)
    emg_sample = emg_sample.reshape(emg_sample.shape[0]*emg_sample.shape[1], emg_sample.shape[2])
    # flip spectrogram vertically for each electrode
    for i in range(numElectrodes):
        num_frequencies = len(frequencies)
        emg_sample[i*num_frequencies:(i+1)*num_frequencies, :] = torch.flip(emg_sample[i*num_frequencies:(i+1)*num_frequencies, :], dims=[0])

    # Convert to PyTorch tensor and normalize
    emg_sample = torch.tensor(emg_sample).float()
    emg_sample = emg_sample.view(numElectrodes, len(frequencies), -1)

    # Reshape into blocks
    
    blocks = emg_sample.view(grid_width, grid_length, len(frequencies), -1)

    # Combine the blocks into the final image
    rows = [torch.cat([blocks[i, j] for j in range(grid_length)], dim=1) for i in range(grid_width)]
    combined_image = torch.cat(rows, dim=0)

    # Normalize combined image
    combined_image -= torch.min(combined_image)
    combined_image /= torch.max(combined_image) - torch.min(combined_image)

    data = combined_image.numpy()

    data_converted = cmap(data)
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))

    width_to_transform_to = min(native_resnet_size, image.shape[-1])
    
    resize = transforms.Resize([length_to_resize_to, width_to_transform_to],
                           interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp between 0 and 1 using torch.clamp
    image_clamped = torch.clamp(image_resized, 0, 1)

    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    final_image = image_normalized.numpy().astype(np.float32)

    return final_image

def optimized_makeOnePhaseSpectrogramImage(data, length, width, resize_length_factor, native_resnet_size):
    spectrogram_window_size = wLenTimesteps // 4
    emg_sample_unflattened = data.reshape(numElectrodes, -1)
    number_of_frequencies = wLenTimesteps 

    # Pre-allocate the array for the CWT coefficients
    grid_width, grid_length = closest_factors(numElectrodes)

    length_to_resize_to = min(native_resnet_size, grid_width * number_of_frequencies)
    width_to_transform_to = min(native_resnet_size, grid_length * width)
    
    frequencies, times, Sxx = stft(emg_sample_unflattened, fs=fs, nperseg=spectrogram_window_size - 1, noverlap=spectrogram_window_size-2, nfft=number_of_frequencies - 1) # defaults to hann window
    Sxx_abs = np.abs(Sxx) # small constant added to avoid log(0)
    # Sxx_dB = 10 * np.log10(np.abs(Sxx_abs) + 1e-12)

    Sxx_phase = np.angle(Sxx)
    Sxx_phase_normalized = (Sxx_phase + np.pi) / (2 * np.pi)
    emg_sample = torch.from_numpy(Sxx_phase_normalized)
 
    emg_sample = emg_sample.reshape(emg_sample.shape[0]*emg_sample.shape[1], emg_sample.shape[2])
    # flip spectrogram vertically for each electrode
    for i in range(numElectrodes):
        num_frequencies = len(frequencies)
        emg_sample[i*num_frequencies:(i+1)*num_frequencies, :] = torch.flip(emg_sample[i*num_frequencies:(i+1)*num_frequencies, :], dims=[0])

    # Convert to PyTorch tensor and normalize
    emg_sample = torch.tensor(emg_sample).float()
    emg_sample = emg_sample.view(numElectrodes, len(frequencies), -1)

    # Reshape into blocks
    
    blocks = emg_sample.view(grid_width, grid_length, len(frequencies), -1)

    # Combine the blocks into the final image
    rows = [torch.cat([blocks[i, j] for j in range(grid_length)], dim=1) for i in range(grid_width)]
    combined_image = torch.cat(rows, dim=0)

    # Normalize combined image
    combined_image -= torch.min(combined_image)
    combined_image /= torch.max(combined_image) - torch.min(combined_image)

    data = combined_image.numpy()

    data_converted = cmap(data)
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))

    width_to_transform_to = min(native_resnet_size, image.shape[-1])
    
    resize = transforms.Resize([length_to_resize_to, width_to_transform_to],
                           interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp between 0 and 1 using torch.clamp
    image_clamped = torch.clamp(image_resized, 0, 1)

    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    final_image = image_normalized.numpy().astype(np.float32)

    return final_image


def optimized_makeOneImage(data, cmap, length, width, resize_length_factor, native_resnet_size):
     # Contrast normalize and convert data
    data = (data - data.min()) / (data.max() - data.min())
    data_converted = cmap(data)
    rgb_data = data_converted[:, :3]
    image_data = np.reshape(rgb_data, (numElectrodes, width, 3))
    image = np.transpose(image_data, (2, 0, 1))
    
    # Resize the image
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

def optimized_makeOneHilbertHuangImage(data, length, width, resize_length_factor, native_resnet_size):

    emg_sample = data 
    max_imfs = 6

    # Perform Empirical Mode Decomposition (EMD)
    intrinsic_mode_functions = emd.sift.sift(emg_sample, max_imfs=max_imfs-1) 
    instantaneous_phase, instantaneous_frequencies, instantaneous_amplitudes = \
        emd.spectra.frequency_transform(imf=intrinsic_mode_functions, sample_rate=fs, method='nht')
    
    # Pad any missing IMFs with zeros
    if instantaneous_phase.shape[-1] < max_imfs:
        padded_instantaneous_phase = np.zeros((instantaneous_phase.shape[0], max_imfs))

        for electrode_at_time in range(instantaneous_phase.shape[0]):
            missing_imfs = max_imfs - instantaneous_phase.shape[-1]
            padding = np.zeros(missing_imfs)
            padded_instantaneous_phase[electrode_at_time] = np.append(instantaneous_phase[electrode_at_time], padding)
        instantaneous_phase = padded_instantaneous_phase

    # Rearrange to be (WLENTIMESTEP, NUM_ELECTRODES, MAX_IMF+1 (includes a combined IMF))
    instantaneous_phase_norm = instantaneous_phase / (2 * np.pi) 
    emg_sample = np.array_split(instantaneous_phase_norm, numElectrodes, axis=0) 
    emg_sample = [torch.tensor(emg) for emg in emg_sample]
    emg_sample = torch.stack(emg_sample)
    emg_sample = emg_sample.permute(1, 0, 2) 

    # Stack the y axis to be all imfs per electrode
    final_emg = torch.zeros(wLenTimesteps, numElectrodes*(max_imfs))
    for t in range(wLenTimesteps):
        for i in range(numElectrodes):
            final_emg[t, i*(max_imfs):(i+1)*(max_imfs)] = emg_sample[t, i, :]

    combined_image = final_emg 
    combined_image -= torch.min(combined_image)
    combined_image /= torch.max(combined_image) - torch.min(combined_image)

    data = combined_image.numpy()
    data_converted = cmap(data) 
    rgb_data = data_converted[:, :, :3]
    image = np.transpose(rgb_data, (2, 0, 1))

    length_to_transform_to = min(native_resnet_size, image.shape[-2])
    width_to_transform_to = min(native_resnet_size, image.shape[-1])
    
    resize = transforms.Resize([length_to_transform_to, width_to_transform_to],
                           interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image_resized = resize(torch.from_numpy(image))

    # Clamp between 0 and 1 using torch.clamp
    image_clamped = torch.clamp(image_resized, 0, 1)

    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_clamped)

    final_image = image_normalized.numpy().astype(np.float32)

    return final_image
 

def calculate_rms(array_2d):
    # Calculate RMS for 2D array where each row is a window
    return np.sqrt(np.mean(array_2d**2))

def getImages(emg, standardScaler, length, width, turn_on_rms=False, rms_windows=10, turn_on_magnitude=False, turn_on_spectrogram=False, turn_on_phase_spectrogram=False, turn_on_cwt=False, global_min=None, global_max=None, turn_on_hht=False):

    if (standardScaler != None):
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
    resize_length_factor = 6
    if turn_on_magnitude:
        resize_length_factor = 3
    native_resnet_size = 224

    with multiprocessing.Pool(processes=5) as pool:
        args = [(emg[i], cmap, length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_async = pool.starmap_async(optimized_makeOneImage, args)
        images = images_async.get()

    if turn_on_magnitude:
        # with multiprocessing.Pool(processes=5) as pool:
        #     args = [(emg[i], length, width, resize_length_factor, native_resnet_size, global_min, global_max) for i in range(len(emg))]
        #     images_async = pool.starmap_async(optimized_makeOneMagnitudeImage, args)
        #     images_magnitude = images_async.get()
        # images = np.concatenate((images, images_magnitude), axis=2)
        raise NotImplementedError("Magnitude is not implemented")

    elif turn_on_spectrogram:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_spectrogram = []
        for i in tqdm(range(len(emg)), desc="Creating Spectrogram Images"):
            images_spectrogram.append(optimized_makeOneSpectrogramImage(*args[i]))
        images = images_spectrogram

    elif turn_on_phase_spectrogram:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_spectrogram = []
        for i in tqdm(range(len(emg)), desc="Creating Spectrogram Images"):
            images_spectrogram.append(optimized_makeOnePhaseSpectrogramImage(*args[i]))
        images = images_spectrogram


    elif turn_on_hht: 
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_spectrogram = []
        for i in tqdm(range(len(emg)), desc="Creating Phase HHT Images"):
            images_spectrogram.append(optimized_makeOneHilbertHuangImage(*args[i]))
        images = images_spectrogram
    
    elif turn_on_cwt:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_cwt_list = []
        # with multiprocessing.Pool(processes=5) as pool:
        for i in tqdm(range(len(emg)), desc="Creating CWT Images"):
            images_cwt_list.append(optimized_makeOneCWTImage(*args[i]))
        images = images_cwt_list
        
    elif turn_on_hht:
        raise NotImplementedError("HHT is not implemented yet")
    
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
        gesture_images = denormalize(image_data[gesture_indices]).cpu().detach().numpy()
        average_images.append(np.mean(gesture_images, axis=0))

    average_images = np.array(average_images)

    resize_transform = transforms.Resize((224, 224))
    # Plot average image of each gesture
    fig, axs = plt.subplots(2, 9, figsize=(10, 5))
    for i in range(numGestures):
        current_average_image = resize_transform(torch.tensor(average_images[i])).numpy()
        axs[i//9, i%9].imshow(current_average_image.transpose(1,2,0))
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
    fig, axs = plt.subplots(rows_per_gesture, total_gestures, figsize=(20, 15))

    print(f"Plotting first fifteen {partition_name} images...")
    for i in range(total_gestures):
        # Find indices of the first 15 images for gesture i
        gesture_indices = np.where(true_np == i)[0][:rows_per_gesture]
        
        # Select and denormalize only the required images
        gesture_images = denormalize(image_data[gesture_indices]).cpu().detach().numpy()
        resize_transform = transforms.Resize((224, 224))

        for j in range(len(gesture_images)):  # len(gesture_images) is no more than rows_per_gesture
            ax = axs[j, i]
            # Transpose the image data to match the expected shape (H, W, C) for imshow
            # Resize to 224 x 224
            current_gesture_image = resize_transform(torch.tensor(gesture_images[j])).numpy()
            ax.imshow(current_gesture_image.transpose(1, 2, 0))
            if j == 0:
                ax.set_title(gesture_labels[i])
            ax.axis('off')

    fig.suptitle(f'First Fifteen {partition_name.capitalize()} Images of Each Gesture')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and log the figure
    firstThreeImages_filename = f'{testrun_foldername}firstFifteenImages_seed{args.seed}_{partition_name}_{formatted_datetime}.png'
    plt.savefig(firstThreeImages_filename, dpi=300)
    wandb.log({f"First Fifteen {partition_name.capitalize()} Images of Each Gesture": wandb.Image(firstThreeImages_filename)})

