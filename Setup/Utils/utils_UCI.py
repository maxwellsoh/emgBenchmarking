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

numGestures = 6 # 7 total, but not all subjects have 7
fs = 1000 #Hz (device sampling frequency is 200Hz but raw data is collected at 1000Hz)
wLen = 250 # ms
wLenTimesteps = int(wLen / 1000 * fs)
stepLen = 50 # 125 ms (increased from 50ms due to high number of subjects)
stepLen = int(stepLen / 1000 * fs)
numElectrodes = 8
num_subjects = 36
cmap = mpl.colormaps['viridis']
# Gesture Labels
gesture_labels = ["hand at rest","hand clenched in a fist","wrist flexion","wrist extension","radial deviations","ulnar deviations","extended palm"]
gesture_labels = gesture_labels[:numGestures]

include_transitions = False # Whether or not to include mixed gestures and label them as their last. Defined by Setup/Setup.py

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

def balance (restimulus):
    indices = []
    for x in range (len(restimulus)):
        unique_gestures = len(torch.unique(restimulus[x]))
        if unique_gestures == 1: 
            gesture = restimulus[x][0] - 1 # 0 is unmarked data, 1 rest, etc
            if gesture >= 0 and gesture <= 6: 
                indices.append(x)

        else: 
            if include_transitions:
                start_gesture = restimulus[x][0] - 1
                end_gesture = restimulus[x][-1] - 1 
            
                if end_gesture >= 0 and end_gesture <= 6:
                    indices.append(x)

                    # Uncertain what Unmarked represents. Will include windows that go from Unmarked -> Gesture but not windows that go from Gesture -> Unmarked. (Unmarked is -1)
          
    return indices

def contract(R, unfold=True):
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(len(R), numGestures))
    if (unfold):
        for x in range(len(R)):
            if include_transitions: 
                gesture = int(R[x][-1]) -1 # take the last gesture of the window, subtract by 1 because 0 is unmarked data
            else: 
                gesture = int(R[x][0]) - 1 # take the first gesture of the window, subtract by 1 because 0 is unmarked data
            labels[x][gesture] = 1.0
    else:
        for x in range(len(R)):
            labels[x][int(R[x]) - 1] = 1.0
    return labels

def filter(emg):
    # sixth-order Butterworth highpass filter
    b, a = butter(N=3, Wn=[5.0, 500.0], btype='bandpass', analog=False, fs=fs)
    emgButter = torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())

    #second-order notch filter at 50 Hz
    b, a = iirnotch(w0=50.0, Q=0.0001, fs=fs)
    return torch.from_numpy(np.flip(filtfilt(b, a, emgButter),axis=0).copy())

def getRestim (n, unfold=True, session_number=1):
    restim = []
    n = "{:02d}".format(n)
    for file in os.listdir(f"uciEMG/{n}/"):
        try:
            if file[0] == str(session_number):
                print("Reading file", file, "Subject", n)
            else:
                continue

            if (unfold):

                file_path = os.path.join(f"uciEMG/{n}/", file)

                data_numpy = np.loadtxt(file_path, dtype=np.float32, skiprows=1)[:, 1:] # ignore first row (header) and first column (time)
                data_tensor = torch.from_numpy(data_numpy)
                data_unfolded = data_tensor.unfold(dimension=0, size=wLenTimesteps, step=stepLen)
                gesture_col = data_unfolded[:, -1] # take the gesture column
                balanced_data = gesture_col[balance(gesture_col)]
                restim.append(balanced_data)

            else:
                data = torch.from_numpy(np.loadtxt(os.path.join(f"uciEMG/{n}/", file), dtype=np.float32, skiprows=1)[:, 1:])
                restim.append(data[:, -1])
        except:
            print("Error reading file", file, "Subject", n)
    if numGestures == 6 and unfold:
        for i in range(len(restim)):
            restim[i] = restim[i][torch.all(restim[i] != 7, axis=1)]
    return torch.cat(restim, dim=0)

def target_normalize (data, target_min, target_max):
    source_min = np.zeros(numElectrodes, dtype=np.float32)
    source_max = np.zeros(numElectrodes, dtype=np.float32)
    
    for i in range(numElectrodes):
        source_min[i] = np.min(data[:, i])
        source_max[i] = np.max(data[:, i])

    data_norm = np.zeros(data.shape, dtype=np.float32)
    for gesture in range(numGestures):
        for i in range(numElectrodes):
            data_norm[:, i] = data_norm[:, i] + (data[:, -1] == (gesture+1)) * (((data[:, i] - source_min[i]) / (source_max[i] 
            - source_min[i])) * (target_max[i][gesture] - target_min[i][gesture]) + target_min[i][gesture])
    data_norm[:, -1] = data[:, -1]
    return data_norm

def getEMG (args, unfold=True, session_number=1):
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

    emg = []
    restim = []
    n = "{:02d}".format(n)
    for file in os.listdir(f"uciEMG/{n}/"):
        try: 
            # data = np.loadtxt(os.path.join(f"uciEMG/{n}/", file), dtype=np.float32, skiprows=1)[:, 1:]
            
            if file[0] == str(session_number):
                data = np.loadtxt(os.path.join(f"uciEMG/{n}/", file), dtype=np.float32, skiprows=1)[:, 1:]

                if (leftout != None and n != leftout):
                    data = target_normalize(data, target_min, target_max)

                data = torch.from_numpy(data)
                if (unfold):
                    data = data.unfold(dimension=0, size=wLenTimesteps, step=stepLen)
                    data = data[balance(data[:, -1])]

                emg.append(data[:, :-1])
                restim.append(data[:, -1])
        except Exception as e:
            print("Error reading file", file, "Subject", n)
            print(e)

    if numGestures == 6 and unfold:
        for i in range(len(restim)):
            emg[i] = emg[i][torch.all(restim[i] != 7, axis=1)]

    return torch.cat(emg, dim=0)

def getEMG_separateSessions(args, unfold=True):
    if (len(args) == 2):
        subject_number, session_number = args
        target_min = None
        target_max = None
        leftout = None
    else:
        subject_number, session_number, target_min, target_max, leftout = args
    
    emg = [] 
    restim = []  
    n = "{:02d}".format(subject_number)
    for file in os.listdir(f"uciEMG/{n}/"):
        try: 
            if file[0] == str(session_number):
                data = np.loadtxt(os.path.join(f"uciEMG/{n}/", file), dtype=np.float32, skiprows=1)[:, 1:]
                
                if (leftout != None and n != leftout):
                    data = target_normalize(data, target_min, target_max)

                data = torch.from_numpy(data)
                if (unfold):
                    data = data.unfold(dimension=0, size=wLenTimesteps, step=stepLen)
                    data = data[balance(data[:, -1])]
                
                emg.append(data[:, :-1])
                restim.append(data[:, -1])
        except Exception as e:
            print("Error reading file", file, "Subject", n)
            print(e)
    if numGestures == 6 and unfold:
        for i in range(len(restim)):
            emg[i] = emg[i][torch.all(restim[i] != 7, axis=1)]
        return torch.cat(emg, dim=0)
    return torch.cat(emg, dim=0)

def getExtrema (n, proportion, lastSessionOnly=False):
    mins = np.zeros((numElectrodes, numGestures))
    maxes = np.zeros((numElectrodes, numGestures))

    if lastSessionOnly:
        emg = getEMG_separateSessions((n, 2), unfold=True) 
        labels = getLabels_separateSessions((n, 2), unfold=True)
    else:
        emg = getEMG(n, unfold=True) # (WINDOW, ELECTRODE, TIMESTEP)
        labels = getLabels(n, unfold=True) #(WINDOW, GESTURE), one hot encoded
        
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


def getRestim_separateSessions(args, unfold=True):
    subject_number, session_number = args
    restim = []
    n = "{:02d}".format(subject_number)
    for file in os.listdir(f"uciEMG/{n}/"):
        try:
            if file[0] == str(session_number):
                if unfold:
                    data = torch.from_numpy(np.loadtxt(os.path.join(f"uciEMG/{n}/", file), dtype=np.float32, skiprows=1)[:, 1:]).unfold(dimension=0, size=wLenTimesteps, step=stepLen)
                    restim.append(data[:, -1][balance(data[:, -1])])
                else:
                    data = torch.from_numpy(np.loadtxt(os.path.join(f"uciEMG/{n}/", file), dtype=np.float32, skiprows=1)[:, 1:])
                    restim.append(data[:, -1])
        except:
            print("Error reading file", file, "Subject", n)
    if numGestures == 6 and unfold:
        for i in range(len(restim)):
            restim[i] = restim[i][torch.all(restim[i] != 7, axis=1)]
        return torch.cat(restim, dim=0)
    return torch.cat(restim, dim=0)

def getLabels (n, unfold=True):
    return contract(getRestim(n, unfold), unfold)

def getLabels_separateSessions(args, unfold=True):
    subject_number, session_number = args
    return contract(getRestim_separateSessions((subject_number, session_number), unfold), unfold)

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
        emg = standardScaler.transform(np.array(emg.view(len(emg), length * width)))
    else:
        emg = np.array(emg.view(len(emg), length * width))
        
    # Use RMS preprocessing
    if turn_on_rms:
        emg = emg.reshape(len(emg), length, width)
        # Reshape data for RMS calculation: (SAMPLES, 16, 5, 10)
        emg = emg.reshape(len(emg), length, rms_windows, width // rms_windows)
        
        # Apply RMS calculation along the last axis (axis=-1)
        emg_rms = np.apply_along_axis(calculate_rms, -1, emg)
        emg = emg_rms  # Resulting shape will be (SAMPLES, 16, 5)
        width = rms_windows
        emg = emg.reshape(len(emg), length * width)

    # Parameters that don't change can be set once
    resize_length_factor = 1
    native_resnet_size = 224

    args = [(emg[i], cmap, length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]

    with multiprocessing.Pool(processes=5) as pool:
        images = list(tqdm(pool.starmap(optimized_makeOneImage, args), total=len(args), desc="Creating Images"))

    if turn_on_magnitude:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size, global_min, global_max) for i in range(len(emg))]
        with multiprocessing.Pool(processes=5) as pool:
            images_magnitude = list(tqdm(pool.starmap(optimized_makeOneMagnitudeImage, args), total=len(args), desc="Creating Magnitude Images"))
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
        # with multiprocessing.Pool(processes=5) as pool:
        for i in tqdm(range(len(emg)), desc="Creating CWT Images"):
            images_cwt_list.append(optimized_makeOneCWTImage(*args[i]))
        images = images_cwt_list

    # elif turn_on_phase: 
    #     # TODO: update the phase here 
        
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

