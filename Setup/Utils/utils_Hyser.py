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
from scipy.signal import spectrogram, stft
import pywt
import fcwt
from tqdm import tqdm

numGestures = 10
fs = 2048.0 # Hz 
wLen = 250.0 # ms
wLenTimesteps = int(wLen / 1000 * fs)
stepLen = int(125.0 / 1000 * fs) # 125 ms
numElectrodes = 256
num_subjects = 20
cmap = mpl.colormaps['viridis']
# Gesture Labels
gesture_nums = {'6' : 0, '7' : 1, '8' : 2, '9' : 3, '10' : 4, '11' : 5, '30' : 6, '31' : 7, '32' : 8, '34' : 9}
gesture_labels = ["wrist flexion", "wrist extension", "wrist radial", "wrist ulnar", "wrist pronation", "wrist supination", 
                    "hand close", "hand open", "thumb and index fingers pinch", "thumb and middle fingers pinch"]

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
    # sixth-order Butterworth bandpass filter
    b, a = butter(N=3, Wn=[5.0, 500.0], btype='bandpass', analog=False, fs=fs)
    emgButter = torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy()).to(dtype=torch.float32)

    #second-order notch filter at 50 Hz
    b, a = iirnotch(w0=50.0, Q=0.0001, fs=fs)
    emgNotch = torch.from_numpy(np.flip(filtfilt(b, a, emgButter),axis=0).copy()).to(dtype=torch.float32)

    return emgNotch

# partition data by channel
def format_emg (data):
    emg = np.zeros((len(data) // numElectrodes, numElectrodes))
    for i in range(len(data) // numElectrodes):
        for j in range(numElectrodes):
            emg[i][j] = data[i * numElectrodes + j]
    return emg

# data is [# samples, # channels]
# target min/max is [# channels, # gestures]
def target_normalize (data, target_min, target_max, gesture):
    source_min = np.zeros(len(data[0]), dtype=np.float16)
    source_max = np.zeros(len(data[0]), dtype=np.float16)
    for i in range(len(data[0])):
        source_min[i] = np.min(data[:, i])
        source_max[i] = np.max(data[:, i])

    for i in range(len(data[0])):
        data[:, i] = ((data[:, i] - source_min[i]) / (source_max[i] 
        - source_min[i])) * (target_max[i][gesture] - target_min[i][gesture]) + target_min[i][gesture]
    return data

def getEMG_help (sub, session, target_min=None, target_max=None, leftout=None, unfold=True):
    emg = []

    currFile = 1
    select_gestures = set([])
    curr_gestures = []
    with open(f'hyser/subject{sub}_session{session}/label_dynamic.txt', 'r') as file:
        vals = file.readline().strip().split(',')
        for v in vals:
            if (v in gesture_nums):
                select_gestures.add(currFile)
                curr_gestures.append(gesture_nums[v])
            currFile += 1

    currFile = 1
    while (os.path.isfile(f'hyser/subject{sub}_session{session}/dynamic_raw_sample{currFile}.dat')):
        if (currFile not in select_gestures):
            currFile += 1
            continue

        data = np.fromfile(f'hyser/subject{sub}_session{session}/dynamic_raw_sample{currFile}.dat', dtype=np.int16).reshape((256, -1)).astype(np.float32)

        adjustment = []
        with open(f'hyser/subject{sub}_session{session}/dynamic_raw_sample{currFile}.hea', 'r') as file:
            ignoreFirst = True
            for line in file:
                if (ignoreFirst):
                    ignoreFirst = False
                else:
                    values = line.split(" ")[2].split("(")
                    adjustment.append((float(values[0]), float(values[1].split(")")[0])))

        for col in range(len(data)):
            data[col] = (data[col] - adjustment[col][1]) / (adjustment[col][0])

        # converts data to form [# samples, # channels]
        data = data.transpose((1, 0))
        if (leftout != None and sub != leftout):
            data = target_normalize(data, target_min, target_max, curr_gestures.pop(0))

        if (unfold):
            emg.append(torch.from_numpy(data).unfold(dimension=0, size=wLenTimesteps, step=stepLen))
        else:
            emg.append(torch.from_numpy(data))

        currFile += 1
        
    return emg

def getEMG (args, session_number=1):
    if (type(args) == int):
        n = args
        target_min = None
        target_max = None
        leftout = None
    else:
        n = args[0]
        target_min = args[1]
        target_max = args[2]
        leftout = args[3]
    
    if (n < 10):
        sub = f'0{n}'
    else:
        sub = f'{n}'

    # emg = getEMG_help(sub, "1", target_max, target_min, leftout) + getEMG_help(sub, "2", target_max, target_min, leftout)
    emg = getEMG_help(sub, str(session_number), target_max, target_min, leftout)
    
    return filter(torch.cat(emg, dim=0))

def getEMG_separateSessions(args):
    if (len(args) == 2):
        subject_number = args[0]
        session_number = args[1]
        target_min = None
        target_max = None
        leftout = None
    else:
        subject_number = args[0]
        session_number = args[1]
        target_min = args[2]
        target_max = args[3]
        leftout = args[4]
    
    if (subject_number < 10):
        sub = f'0{subject_number}'
    else:
        sub = f'{subject_number}'

    emg = getEMG_help(sub, str(session_number), target_min, target_max, leftout)
    emg = filter(torch.cat(emg, dim=0))
    return emg
        
    
def getExtrema (n, proportion, lastSessionOnly=False):
    """Returns the min max of the electrode per gesture for a proportion of its windows. 
    
    Used for target normalization.

    Args:
        n: participant
        proportion: proportion of windows to consider
        exercise: exercise
        args_exercises: exercises for the overall program (important for getLabels)

    Returns:
        (ELECTRODE, GESTURE): min and max values for each electrode per gesture

    """
    mins = np.zeros((numElectrodes, numGestures))
    maxes = np.zeros((numElectrodes, numGestures))

    if lastSessionOnly:
        emg = getEMG((n), session_number=2) 
        labels = getLabels(n, unfold=True, session_number=2)
    else:
        emg = getEMG((n), session_number=1) # (WINDOW, CHANNEL, TIME STEP)
        labels = getLabels(n, unfold=True, session_number=1) # (WINDOW, GESTURE)

    # convert labels out of one hot encoding
    labels = torch.argmax(labels, dim=1)

    # Get the proportion of the windows per gesture 
    unique_labels, counts = np.unique(labels, return_counts=True)
    size_per_gesture = np.round(proportion*counts).astype(int)
    gesture_amount = dict(zip(unique_labels, size_per_gesture)) # (GESTURE, NUMBER OF WINDOWS)

    for gesture in gesture_amount.keys():
        size_for_current_gesture = gesture_amount[gesture]

        all_windows = np.where(labels == gesture)[0]
        chosen_windows = all_windows[:size_for_current_gesture] 
        
        # out of these indices, pick the min/max emg values
        for j in range(numElectrodes): 
            # minimum emg value
            mins[j][gesture] = torch.min(emg[chosen_windows, j])
            maxes[j][gesture] = torch.max(emg[chosen_windows, j])
    return mins, maxes

def contract(R):
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(len(R), numGestures))
    for x in range(len(R)):
        labels[x][gesture_nums[R[x]]] = 1.0
    return labels

# returns [# samples, # gestures]
def getLabels (n, unfold=True, session_number=1):
    labels = []

    if (n < 10):
        sub = f'0{n}'
    else:
        sub = f'{n}'

    file = open(f'hyser/subject{sub}_session{session_number}/label_dynamic.txt', 'r')
    vals = file.readline().strip().split(',')
    for v in vals:
        if (v in gesture_nums):
            if (unfold):
                for i in range(7):
                    labels.append(v)
            else:
                labels.append(v)
    file.close()

    return contract(labels)

def getLabels_separateSessions(args, unfold=True):
    subject_number = args[0]
    session_number = args[1]
    
    if (subject_number < 10):
        sub = f'0{subject_number}'
    else:
        sub = f'{subject_number}'
    
    labels = []
    file = open(f'hyser/subject{sub}_session{session_number}/label_dynamic.txt', 'r')
    vals = file.readline().strip().split(',')
    for v in vals:
        if (v in gesture_nums):
            if unfold:
                for i in range(7):
                    labels.append(v)
            else:
                labels.append(v)
                
    file.close()
    return contract(labels)

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
    # NOTE: Should this be contrast normalized? Then only patterns of data will be visible, not absolute values
    data = (data - data.min()) / (data.max() - data.min())
    data_converted = cmap(data)
    rgb_data = data_converted[:, :3]
    image_data = np.reshape(rgb_data, (numElectrodes, width, 3))
    image = np.transpose(image_data, (2, 0, 1))

    # Split image and resize
    imageL, imageR = np.split(image, 2, axis=1)
    imageL, imageR = map(lambda img: torch.from_numpy(img), (imageL, imageR))
    
    # Get max and min values after interpolation
    max_val = max(imageL.max(), imageR.max())
    min_val = min(imageL.min(), imageR.min())
    
    # Contrast normalize again after interpolation
    imageL, imageR = map(lambda img: (img - min_val) / (max_val - min_val), (imageL, imageR))
    
    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imageL, imageR = map(normalize, (imageL, imageR))
    
    return torch.cat([imageL, imageR], dim=1).numpy().astype(np.float32)

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
        coefficients_dB = 10 * np.log10(coefficients_abs + 1e-12)  # Avoid log(0)
        time_frequency_emg[i * (highest_cwt_scale):(i + 1) * (highest_cwt_scale), :] = coefficients_dB

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

def optimized_makeOneSpectrogramImage(data, length, width, resize_length_factor, native_resnet_size):
    spectrogram_window_size = wLenTimesteps // 16
    emg_sample_unflattened = data.reshape(numElectrodes, -1)
    number_of_frequencies = wLenTimesteps 

    # Pre-allocate the array for the CWT coefficients
    grid_width, grid_length = closest_factors(numElectrodes)

    length_to_resize_to = min(native_resnet_size, grid_width * number_of_frequencies)
    width_to_transform_to = min(native_resnet_size, grid_length * width)
    
    frequencies, times, Sxx = stft(emg_sample_unflattened, fs=fs, nperseg=spectrogram_window_size - 1, noverlap=spectrogram_window_size-2, nfft=number_of_frequencies - 1) # defaults to hann window
    Sxx_abs = np.abs(Sxx) # small constant added to avoid log(0)
    Sxx_dB = 10 * np.log10(np.abs(Sxx_abs) + 1e-12)
    emg_sample = torch.from_numpy(Sxx_dB)
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
    spectrogram_window_size = wLenTimesteps // 16
    emg_sample_unflattened = data.reshape(numElectrodes, -1)
    number_of_frequencies = wLenTimesteps 

    # Pre-allocate the array for the CWT coefficients
    grid_width, grid_length = closest_factors(numElectrodes)

    length_to_resize_to = min(native_resnet_size, grid_width * number_of_frequencies)
    width_to_transform_to = min(native_resnet_size, grid_length * width)
    
    frequencies, times, Sxx = stft(emg_sample_unflattened, fs=fs, nperseg=spectrogram_window_size - 1, noverlap=spectrogram_window_size-2, nfft=number_of_frequencies - 1) # defaults to hann window
    
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


def calculate_rms(array_2d):
    # Calculate RMS for 2D array where each row is a window
    return np.sqrt(np.mean(array_2d**2))

def process_chunk(data_chunk):
    return np.apply_along_axis(calculate_rms, -1, data_chunk)

def process_optimized_makeOneImage(args_tuple):
    return optimized_makeOneImage(*args_tuple)

def process_optimized_makeOneMagnitudeImage(args_tuple):
    return optimized_makeOneMagnitudeImage(*args_tuple)

def getImages(emg, standardScaler, length, width, turn_on_rms=False, rms_windows=10, turn_on_magnitude=False, global_min=None, global_max=None,
              turn_on_spectrogram=False, turn_on_phase_spectrogram=False, turn_on_cwt=False, turn_on_hht=False):
    
    
    if standardScaler is not None:
        emg = standardScaler.transform(np.array(emg.view(len(emg), length*width)))
    else:
        emg = np.array(emg.view(len(emg), length*width))

    # Use RMS preprocessing
    if turn_on_rms:
        emg = emg.reshape(len(emg), length, width)
        # Reshape data for RMS calculation: (SAMPLES, 16, 5, 10)
        emg = emg.reshape(len(emg), length, rms_windows, width // rms_windows)
        
        num_splits = multiprocessing.cpu_count()
        data_chunks = np.array_split(emg, num_splits)
        
        emg_rms = process_map(process_chunk, data_chunks, chunksize=1, max_workers=num_splits, desc="Calculating RMS")
        # Apply RMS calculation along the last axis (axis=-1)
        emg = np.concatenate(emg_rms)  # Resulting shape will be (SAMPLES, 16, 5)
        width = rms_windows
        emg = emg.reshape(len(emg), length*width)
        
        del emg_rms
        del data_chunks

    # Parameters that don't change can be set once
    resize_length_factor = 1
    native_resnet_size = 224
    
    # Using process_map instead of multiprocessing.Pool directly
    # images = process_map(process_optimized_makeOneImage, args, chunksize=1, max_workers=4)
    if not turn_on_magnitude and not turn_on_spectrogram and not turn_on_cwt and not turn_on_hht:
        args = [(emg[i], cmap, length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_list = []
        for i in tqdm(range(len(emg)), desc="Creating Images"):
            images_list.append(optimized_makeOneImage(*args[i]))
        images = images_list

    if turn_on_magnitude:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size, global_min, global_max) for i in range(len(emg))]
        images_magnitude = process_map(process_optimized_makeOneMagnitudeImage, args, chunksize=1, max_workers=32)
        images = np.concatenate((images, images_magnitude), axis=2)
    
    elif turn_on_spectrogram:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_spectrogram = []
        for i in tqdm(range(len(emg)), desc="Creating Spectrogram Images"):
            images_spectrogram.append(optimized_makeOneSpectrogramImage(*args[i]))
        images = images_spectrogram

    elif turn_on_phase_spectrogram:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_spectrogram = []
        for i in tqdm(range(len(emg)), desc="Creating Phase Spectrogram Images"):
            images_spectrogram.append(optimized_makeOnePhaseSpectrogramImage(*args[i]))
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
    cf_matrix = confusion_matrix(true, pred, labels=range(numGestures))
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
        current_image = resize_transform(torch.from_numpy(average_images[i])).numpy()
        axs[i//9, i%9].imshow(current_image.transpose(1,2,0))
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
    
    resize_transform = transforms.Resize((224, 224))

    print(f"Plotting first fifteen {partition_name} images...")
    for i in range(total_gestures):
        # Find indices of the first 15 images for gesture i
        gesture_indices = np.where(true_np == i)[0][:rows_per_gesture]
        
        # Select and denormalize only the required images
        gesture_images = denormalize(image_data[gesture_indices]).cpu().detach().numpy()

        for j in range(len(gesture_images)):  # len(gesture_images) is no more than rows_per_gesture
            ax = axs[j, i]
            # Transpose the image data to match the expected shape (H, W, C) for imshow
            current_image = resize_transform(torch.from_numpy(gesture_images[j])).numpy()
            ax.imshow(current_image.transpose(1, 2, 0))
            if j == 0:
                ax.set_title(gesture_labels[i])
            ax.axis('off')

    fig.suptitle(f'First Fifteen {partition_name.capitalize()} Images of Each Gesture')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and log the figure
    firstThreeImages_filename = f'{testrun_foldername}firstFifteenImages_seed{args.seed}_{partition_name}_{formatted_datetime}.png'
    plt.savefig(firstThreeImages_filename, dpi=300)
    wandb.log({f"First Fifteen {partition_name.capitalize()} Images of Each Gesture": wandb.Image(firstThreeImages_filename)})

