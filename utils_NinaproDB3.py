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
from scipy import io
from tqdm.contrib.concurrent import process_map  # Use process_map from tqdm.contrib
from scipy.signal import stft
import fcwt

fs = 2000 # Hz (SEMG signals sampling rate)
wLen = 250 # ms
wLenTimesteps = int(wLen / 1000 * fs)
stepLen = int(50.0 / 1000 * fs) # 50 ms

numElectrodes = 12 # number of EMG columns
num_subjects = 11

MISSING_SUBJECT = 10 # Subject 10 is missing from exercise 3

cmap = mpl.colormaps['viridis']
# Gesture Labels
gesture_labels = {}
gesture_labels['Rest'] = ['Rest'] # Shared between exercises

gesture_labels[1] = ['Thumb Up', 'Index Middle Extension', 'Ring Little Flexion', 'Thumb Opposition', 'Finger Abduction', 'Fist', 'Pointing Index', 'Finger Adduction',
                    'Middle Axis Supination', 'Middle Axis Pronation', 'Little Axis Supination', 'Little Axis Pronation', 'Wrist Flexion', 'Wrist Extension', 'Radial Deviation',
                    'Ulnar Deviation', 'Wrist Extension Fist'] # End exercise B

gesture_labels[2] = ['Large Diameter Grasp', 'Small Diameter Grasp', 'Fixed Hook Grasp', 'Index Finger Extension Grasp', 'Medium Wrap',
                    'Ring Grasp', 'Prismatic Four Fingers Grasp', 'Stick Grasp', 'Writing Tripod Grasp', 'Power Sphere Grasp', 'Three Finger Sphere Grasp', 'Precision Sphere Grasp',
                    'Tripod Grasp', 'Prismatic Pinch Grasp', 'Tip Pinch Grasp', 'Quadrupod Grasp', 'Lateral Grasp', 'Parallel Extension Grasp', 'Extension Type Grasp', 'Power Disk Grasp',
                    'Open A Bottle With A Tripod Grasp', 'Turn A Screw', 'Cut Something'] # End exercise C

gesture_labels[3] = ['Flexion Of The Little Finger', 'Flexion Of The Ring Finger', 'Flexion Of The Middle Finger', 'Flexion Of The Index Finger', 
                     'Abduction Of The Thumb', 'Flexion Of The Thumb', 'Flexion Of Index And Little Finger', 'Flexion Of Ring And Middle Finger', 
                     'Flexion Of Index Finger And Thumb'] # End exercise D

partial_gesture_labels = ['Rest', 'Finger Abduction', 'Fist', 'Finger Adduction', 'Middle Axis Supination', 
                          'Middle Axis Pronation', 'Wrist Flexion', 'Wrist Extension', 'Radial Deviation', 'Ulnar Deviation']
partial_gesture_indices = [0] + [gesture_labels[1].index(g) + len(gesture_labels['Rest']) for g in partial_gesture_labels[1:]] # 0 is for rest
class CustomDataset_swav(Dataset):
    def __init__(self, data, labels=None, transform=None):
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
    
class CustomDataset_Simclr(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x_i = self.transform(x)
            x_j = self.transform(x)
        else:
            x_i = x_j = x
        
        if self.labels is not None:
            y = self.labels[idx]
            return (x_i, x_j), y
        return (x_i, x_j)
    
class CustomDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
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
    
def getPartialEMG (args):
    n, exercise = args
    restim = getRestim(n, exercise)
    emg = torch.from_numpy(io.loadmat(f'./NinaproDB3/DB3_s{n}/S{n}_E{exercise}_A1.mat')['emg']).to(torch.float16)
    return filter(emg.unfold(dimension=0, size=wLenTimesteps, step=stepLen)[balance(restim)])

def getPartialLabels (args):
    n, exercise = args
    restim = getRestim(n, exercise)
    return contract(restim[balance(restim)])
        
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
    """ Balances distribution of restimulus by minimizing zero (rest) gestures.

    Args:
        restimulus (tensor): restimulus tensor
    """

    numZero = 0
    indices = []
    count_dict = {}
    
    # First pass: count the occurrences of each unique tensor
    for x in range(len(restimulus)):
        unique_elements = torch.unique(restimulus[x])
        if len(unique_elements) == 1:
            element = unique_elements.item()
            if element in count_dict:
                count_dict[element] += 1
            else:
                count_dict[element] = 1
    
    # Calculate average count of non-zero elements
    non_zero_counts = [count for key, count in count_dict.items() if key != 0]
    if non_zero_counts:
        avg_count = sum(non_zero_counts) / len(non_zero_counts)
    else:
        avg_count = 0  # Handle case where there are no non-zero unique elements

    # Second pass: apply the threshold logic
    for x in range(len(restimulus)):
        unique_elements = torch.unique(restimulus[x])
        if len(unique_elements) == 1:
            if unique_elements.item() == 0:
                if numZero < avg_count:
                    indices.append(x)
                numZero += 1
            else:
                indices.append(x)
                
    return indices

def contract(restim, unfold=True):
    """Converts restimulus tensor to one-hot encoded tensor.

    Args:
        restim (tensor): restimulus data tensor
        unfold (bool, optional): whether data was unfolded according to time steps. Defaults to True.

    Returns:
        labels: restimulus data now one-hot encoded
    """
    numGestures = restim.max() + 1 # + 1 to account for rest gesture
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(len(restim), numGestures))
    if unfold:
        for x in range(len(restim)):
            labels[x][int(restim[x][0][0])] = 1.0
    else:
        for x in range(len(restim)):
            labels[x][int(restim[x][0])] = 1.0
    return labels

def filter(emg):
    # sixth-order Butterworth highpass filter
    b, a = butter(N=1, Wn=999.0, btype='lowpass', analog=False, fs=2000.0)
    return torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())

def getRestim (n: int, exercise: int, unfold=True):
    """
    Returns a restiumulus (label) tensor for participant n and exercise exercise and if unfold, unfolded across time. 

    (Unfold=False is needed in getEMG for target normalization)

    Args:
        n (int): participant 
        exercise (int): exercise. 
        unfold (bool, optional): whether or not to unfold data across time steps. Defaults to True.
    """
    restim = torch.from_numpy(io.loadmat(f'./NinaproDB3/DB3_s{n}/S{n}_E{exercise}_A1.mat')['restimulus'])

    if unfold:
        return restim.unfold(dimension=0, size=wLenTimesteps, step=stepLen)
    return restim

def target_normalize (data, target_min, target_max, restim):
    source_min = np.zeros(numElectrodes, dtype=np.float32)
    source_max = np.zeros(numElectrodes, dtype=np.float32)

    resize = min(len(data), len(restim))
    data = data[:resize]
    restim = restim[:resize]
    
    for i in range(numElectrodes):
        source_min[i] = np.min(data[:, i])
        source_max[i] = np.max(data[:, i])
        if source_min[i] == source_max[i]:
            source_max[i] = source_min[i] + 1

    data_norm = np.zeros(data.shape, dtype=np.float32)
    for gesture in range(target_min.shape[1]):
        if target_min[0][gesture] == 0 and target_max[0][gesture] == 0:
            continue
        for i in range(numElectrodes):
            data_norm[:, i] = data_norm[:, i] + (restim[:, 0] == gesture) * (((data[:, i] - source_min[i]) / (source_max[i] 
            - source_min[i])) * (target_max[i][gesture] - target_min[i][gesture]) + target_min[i][gesture])
    return data_norm

def getEMG (input):
    """Returns EMG data for a given participant and exercise. EMG data is balanced (reduced rest gestures), target normalized (if toggled), filtered (butterworth), and unfolded across time. 

    Args:
        n (int): participant number
        exercise (int): exercise number
        target_min (np.array): minimum target values for each electrode
        target_max (np.array): maximum target values for each electrode
        leftout (int): participant number to leave out
        args: argument parser object (needed for DB3 to ignore subject 10)

    Returns:
        (WINDOW, ELECTRODE, TIME STEP): EMG data
    """

    if (len(input) == 3):
        n, exercise, args = input
        leftout = None
        is_target_normalize = False
    else:
        n, exercise, target_min, target_max, leftout, args = input
        is_target_normalize = True

    if args.force_regression and n == MISSING_SUBJECT: 
        return None 
    
    emg = io.loadmat(f'./NinaproDB3/DB3_s{n}/S{n}_E{exercise}_A1.mat')['emg'] # (TOTAL TIME STEPS, ELECTRODE)

    # normalize data for non leftout participants 
    if (is_target_normalize and n != leftout):
        emg = target_normalize(emg, target_min, target_max, np.array(getRestim(n, exercise, unfold=False)))

    restim = getRestim(n, exercise, unfold=True)
    emg = torch.from_numpy(emg).to(torch.float16)
    return filter(emg.unfold(dimension=0, size=wLenTimesteps, step=stepLen)[balance(restim)]) # (WINDOWS, ELECTRODE, TIME STEP)

def get_decrements(args):
    """
    Calculates how much gestures from exercise 1, 2, and 3 should be decremented by to make them sequential.

    Args:
        args: args parser object

    Returns:
        (d1, d2, d3): decrements for each exercise
    """
    
    decrements = {(1,): [0, 0, 0], (2,): [0, 17, 0], (3,): [0, 0, 40], (1,2): [0, 0, 0], (1,3): [0, 0, 23], (2,3): [0, 17, 17], (1,2,3): [0, 0, 0]}
    exercises = tuple(args.exercises)
    return decrements[exercises]

def make_gestures_sequential(balanced_restim, args):
    """
    Removes missing gaps between gestures depending on which exercises are selected.

    Ex: If args.exercises = [1, 3], gesture labels in exercise 1 are kept the same while gesture labels in exercise 3 are decremented by 23. 

    Doing so prevents out of bound array accesses in train_test_split. 

    Returns:
        balanced_restim: restim but with gestures now sequential
    """
   
    exercise_starts = {1: 1, 2: 18, 3: 41}
    decrements = get_decrements(args)
    for x in range(len(balanced_restim)): 
        value = balanced_restim[x][0][0] # TODO: break here and check why its [x][0][0]

        if value != 0:
            exercise = (max(ex for ex in exercise_starts if exercise_starts[ex] <= value))-1
            d = decrements[exercise]
    
            balanced_restim[x][0][0] = value - d

    return balanced_restim

def getLabels (input):
    """Returns one-hot-encoding labels for a given participant and exercise. Labels are balanced (reduced rest gestures) and are sequential (no gaps between gestures of different exercises).

    Args:
        n (int): participant number
        exercise (int): exercise number
        args: argument parser object

    Returns:
        (TIME STEP, GESTURE): one-hot-encoded labels for participant n and exercise exercise
    """

    n, exercise, args = input

    if args.force_regression and n == MISSING_SUBJECT:
        return None

    restim = getRestim(n, exercise)             
    balanced_restim = restim[balance(restim)]   # (WINDOW, GESTURE, TIME STEP) 
    ordered_restim = make_gestures_sequential(balanced_restim, args) 
    return contract(ordered_restim)

def getExtrema (n, proportion, exercise, args):
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

    # Windowed data (must be windowed and balanced so that it matches the splitting in train_test_split)
    emg = getEMG((n, exercise, args))       # (WINDOW, ELECTRODE, TIME STEP)
    labels = getLabels((n, exercise, args))  # (TIME STEP, LABEL)

    # need to convert labels out of one-hot encoding
    numGestures = labels.shape[1]
    labels = torch.argmax(labels, dim=1) 
    
    # Create new arrays to hold data
    mins = np.zeros((numElectrodes, numGestures))   
    maxes = np.zeros((numElectrodes, numGestures))

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
           
def getForces(input):
    """Returns force data for a given participant and exercise. Forces are balanced (reduced rest gestures) and sequential (no gaps between gestures of different exercises).

    Args:
        (n, exercise): participant number and exercise number

    Returns:
        _type_: _description_
    """
    n, exercise = input

    if n == MISSING_SUBJECT: 
        return None
        # implicitly, args.force_regression 

    assert exercise == 3, "Only exercise 3 has force data"
    force = torch.from_numpy(io.loadmat(f'./NinaproDB3/DB3_s{n}/S{n}_E{exercise}_A1.mat')['force'])
    force = force.unfold(dimension=0, size=wLenTimesteps, step=stepLen)
    restim = getRestim(n, exercise)
    balanced_indices = balance(restim)
    return force[balanced_indices]
    
def optimized_makeOneMagnitudeImage(data, length, width, resize_length_factor, native_resnet_size, global_min, global_max):
    # Normalize with global min and max
    data = (data - global_min) / (global_max - global_min)
    data_converted = cmap(data)
    rgb_data = data_converted[:, :3]
    image_data = np.reshape(rgb_data, (numElectrodes, width, 3))
    image = np.transpose(image_data, (2, 0, 1))
    
    # Split image and resize
    imageL, imageR = np.split(image, 2, axis=2)
    #resize = transforms.Resize([length * resize_length_factor, native_resnet_size // 2],
    #                           interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    #imageL, imageR = map(lambda img: resize(torch.from_numpy(img)), (imageL, imageR))
    imageL, imageR = map(lambda img: torch.from_numpy(img), (imageL, imageR))
    
    # Clamp between 0 and 1 using torch.clamp
    imageL, imageR = map(lambda img: torch.clamp(img, 0, 1), (imageL, imageR))
    
    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imageL, imageR = map(normalize, (imageL, imageR))
    
    return torch.cat([imageL, imageR], dim=2).numpy().astype(np.float32)

def optimized_makeOneImage(data, cmap, length, width, resize_length_factor, native_resnet_size, index, display_interval=1000):
    # Normalize and convert data to a usable color map
    data = (data - data.min()) / (data.max() - data.min())
    data_converted = cmap(data)
    rgb_data = data_converted[:, :3]
    image_data = np.reshape(rgb_data, (length, width, 3))
    image = np.transpose(image_data, (2, 0, 1))

    imageL, imageR = np.split(image, 2, axis=1)
    # print(imageL.shape)
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

def optimized_makeOneSpectrogramImage(data, length, width, resize_length_factor, native_resnet_size):
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

def calculate_rms(array_2d):
    # Calculate RMS for 2D array where each row is a window
    return np.sqrt(np.mean(array_2d**2, axis=-1))

def process_chunk(data_chunk):
    return np.apply_along_axis(calculate_rms, -1, data_chunk)

def process_optimized_makeOneImage(args_tuple):
    return optimized_makeOneImage(*args_tuple)

def process_optimized_makeOneMagnitudeImage(args_tuple):
    return optimized_makeOneMagnitudeImage(*args_tuple)

def process_optimized_makeOneImageChunk(args_tuple):
    images = [None] * len(args_tuple)
    for i in range(len(args_tuple)):
        images[i] = optimized_makeOneImage(*args_tuple[i])
    return images

def process_optimized_makeOneMagnitudeImageChunk(args_tuple):
    images = [None] * len(args_tuple)
    for i in range(len(args_tuple)):
        images[i] = optimized_makeOneMagnitudeImage(*args_tuple[i])
    return images

def getImages(emg, standardScaler, length, width, turn_on_rms=False, rms_windows=10, turn_on_magnitude=False, global_min=None, global_max=None,
              turn_on_spectrogram=False, turn_on_cwt=False, turn_on_hht=False):
    
    
    if standardScaler is not None:
        emg = standardScaler.transform(np.array(emg.view(len(emg), length*width)))
    else:
        emg = np.array(emg.view(len(emg), length*width))

    # Use RMS preprocessing
    if turn_on_rms:
        emg = emg.reshape(len(emg), length, width)
        # Reshape data for RMS calculation: (SAMPLES, 16, 5, 10)
        emg = emg.reshape(len(emg), length, rms_windows, width // rms_windows)
        
        num_splits = multiprocessing.cpu_count() // 2
        data_chunks = np.array_split(emg, num_splits)
        
        emg_rms = process_map(process_chunk, data_chunks, chunksize=1, max_workers=num_splits, desc="Calculating RMS")
        # Apply RMS calculation along the last axis (axis=-1)
        # emg_rms = np.apply_along_axis(calculate_rms, -1, emg)
        emg = np.concatenate(emg_rms)  # Resulting shape will be (SAMPLES, 16, 5)
        width = rms_windows
        emg = emg.reshape(len(emg), length*width)
        
        del emg_rms
        del data_chunks

    # Parameters that don't change can be set once
    resize_length_factor = 1
    if turn_on_magnitude:
        resize_length_factor = 1
    native_resnet_size = 224
    
    args = [(emg[i], cmap, length, width, resize_length_factor, native_resnet_size, i) for i in range(len(emg))]
    chunk_size = len(args) // (multiprocessing.cpu_count() // 2)
    arg_chunks = [args[i:i + chunk_size] for i in range(0, len(args), chunk_size)]
    images = []
    for i in tqdm(range(len(arg_chunks)), desc="Creating Images in Chunks"):
        images.extend(process_optimized_makeOneImageChunk(arg_chunks[i]))

    if turn_on_magnitude:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size, global_min, global_max) for i in range(len(emg))]
        chunk_size = len(args) // (multiprocessing.cpu_count() // 2)
        arg_chunks = [args[i:i + chunk_size] for i in range(0, len(args), chunk_size)]
        images_magnitude = []
        for i in tqdm(range(len(arg_chunks)), desc="Creating Magnitude Images in Chunks"):
            images_magnitude.extend(process_optimized_makeOneMagnitudeImageChunk(arg_chunks[i]))
        images = np.concatenate((images, images_magnitude), axis=2)
    
    elif turn_on_spectrogram:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_spectrogram = []
        for i in tqdm(range(len(emg)), desc="Creating Spectrogram Images"):
            images_spectrogram.append(optimized_makeOneSpectrogramImage(*args[i]))
        images = images_spectrogram
    
    elif turn_on_cwt:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_cwt_list = []
        for i in tqdm(range(len(emg)), desc="Creating CWT Images"):
            images_cwt_list.append(optimized_makeOneCWTImage(*args[i]))
        images = images_cwt_list
    
    if turn_on_hht:
        raise NotImplementedError("HHT is not implemented yet.")
    
    return images
        

    # with multiprocessing.Pool(processes=32) as pool:
    #     args = [(emg[i], cmap, length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
    #     images_async = pool.starmap_async(optimized_makeOneImage, args)
    #     images = images_async.get()

    # if turn_on_magnitude:
    #     with multiprocessing.Pool(processes=32) as pool:
    #         args = [(emg[i], length, width, resize_length_factor, native_resnet_size, global_min, global_max) for i in range(len(emg))]
    #         images_async = pool.starmap_async(optimized_makeOneMagnitudeImage, args)
    #         images_magnitude = images_async.get()
    #     images = np.concatenate((images, images_magnitude), axis=2)
        
    # if turn_on_cwt:
    #     NotImplementedError("CWT is not implemented yet.")
        
    # if turn_on_spectrogram:
    #     NotImplementedError("Spectrogram is not implemented yet.")
        
    # if turn_on_hht:
    #     NotImplementedError("HHT is not implemented yet.")
    
    # return images

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
    
    if args.force_regression:
        return 
    
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
    numGestures = len(gesture_labels)

    for i in range(numGestures):
        # Find indices
        gesture_indices = np.where(true_np == i)[0]

        # Select and denormalize only the required images
        gesture_images = denormalize(image_data[gesture_indices]).cpu().detach().numpy()
        average_images.append(np.mean(gesture_images, axis=0))

    average_images = np.array(average_images)

    # resize average images to 224 x 224
    resize = transforms.Resize([224, 224], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    average_images = np.array([resize(torch.from_numpy(img)).numpy() for img in average_images])

    # Plot average image of each gesture
    fig, axs = plt.subplots(2, 9, figsize=(15, 5))
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
    total_gestures = len(gesture_labels)  # Replace with the actual number of gestures

    # Create subplots
    fig, axs = plt.subplots(rows_per_gesture, total_gestures, figsize=(20, 15))

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



