import torch
import numpy as np
import pandas as pd
import random
import h5py
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
from scipy.signal import spectrogram, stft
import pywt
from tqdm.contrib.concurrent import process_map  # Use process_map from tqdm.contrib
import glob

numGestures = 10
fs = 4000 #Hz
wLen = 250 # ms
wLenTimesteps = int(wLen / 1000 * fs)
stepLen = int(50.0 / 1000.0 * fs) #50 ms
numElectrodes = 64
num_subjects = 13
cmap = mpl.colormaps['viridis']
gesture_labels = ['abduct_p1', 'adduct_p1', 'extend_p1', 'grip_p1', 'pronate_p1', 'rest_p1', 'supinate_p1', 'tripod_p1', 'wextend_p1', 'wflex_p1']
participants = [8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22]
participants_with_online_data = [8, 9, 11, 15, 18, 21, 22]

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

def highpassFilter (emg):
    b, a = butter(N=1, Wn=120.0, btype='highpass', analog=False, fs=fs)
    # what axis should the filter apply to? other datasets have axis=0
    return torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=-1).copy())

def target_normalize (data, target_min, target_max, gesture):
    source_min = np.zeros(len(data[0]), dtype=np.float32)
    source_max = np.zeros(len(data[0]), dtype=np.float32)
    for i in range(len(data[0])):
        source_min[i] = np.min(data[:, i])
        source_max[i] = np.max(data[:, i])

    for i in range(len(data[0])):
        data[:, i] = ((data[:, i] - source_min[i]) / (source_max[i] 
        - source_min[i])) * (target_max[i][gesture] - target_min[i][gesture]) + target_min[i][gesture]
    return data

# returns array with dimensions (# of samples)x64x10x100 [SAMPLES, CHANNELS, GESTURES, TIME]
def getData(n, gesture, target_max=None, target_min=None, leftout=None, session_number=1):
    session_number_mapping = {1: 'initial', 2: 'recalibration'}
    if (n<10):
        file = h5py.File('./Jehan_Dataset/p00' + str(n) + f'/data_allchannels_{session_number_mapping[session_number]}.h5', 'r')
    else:
        file = h5py.File('./Jehan_Dataset/p0' + str(n) + f'/data_allchannels_{session_number_mapping[session_number]}.h5', 'r')
    data = np.array(file[gesture])
    if (leftout is not None and n != leftout):
        data = target_normalize(data, target_min, target_max, gesture_labels.index(gesture))
    data = highpassFilter(torch.from_numpy(data).unfold(dimension=-1, size=wLenTimesteps, step=stepLen))

    return torch.cat([data[i] for i in range(len(data))], axis=1).permute([1, 0, 2])

def getOnlineUnlabeledData(subject_number):
    subject_number = participants[subject_number-1]
    assert subject_number in participants_with_online_data, "Subject number does not have online data."
    folder_paths = glob.glob(f"./Jehan_Unlabeled_Dataset/p{subject_number:03}_online_part-*/")
    data_all = []
    for folder_path in folder_paths:
        file_paths = glob.glob(f"{folder_path}/*.h5")
        for file_path in file_paths:
            file = h5py.File(file_path, 'r')
            data = np.array(file['realtime_emgdata']) # shape: (num_channels, num_samples)
            data = highpassFilter(torch.from_numpy(data).unfold(dimension=-1, size=wLenTimesteps, step=stepLen))
            data_all.append(data)
    return torch.cat(data_all, axis=1).permute([1, 0, 2])

def getEMG(args):
    if (type(args) == int):
        n = participants[args-1]
        target_max = None
        target_min = None
        leftout = None
    else:
        n = participants[args[0]-1]
        target_max = args[1]
        target_min = args[2]
        leftout = args[3]

    return torch.cat([getData(n, name, target_max, target_min, leftout) for name in gesture_labels], axis=0)

def getEMG_separateSessions(args):
    subject_number, session_number = args
    if (len(args) == 2):
        subject_number = participants[subject_number-1]
        session_number = session_number
        target_max = None
        target_min = None
        leftout = None
    else:
        subject_number = participants[subject_number-1]
        session_number = session_number
        target_max = args[2]
        target_min = args[3]
        leftout = args[4]
        
    return torch.cat([getData(subject_number, name, target_max, target_min, leftout, session_number) for name in gesture_labels], axis=0)

# assumes first 1/12 of target domain accessed
def getExtrema (n):
    mins = np.zeros((numElectrodes, numGestures))
    maxes = np.zeros((numElectrodes, numGestures))
    
    if (n<10):
        file = h5py.File('./Jehan_Dataset/p00' + str(n) +'/data_allchannels_initial.h5', 'r')
    else:
        file = h5py.File('./Jehan_Dataset/p0' + str(n) +'/data_allchannels_initial.h5', 'r')
    
    for i in range(numGestures):    
        data = np.array(file[gesture_labels[0]])

        for j in range(numElectrodes):
            mins[j][i] = np.min(data[:len(data)//12, j])
            maxes[j][i] = np.max(data[:len(data)//12, j])
    return mins, maxes

def getGestures(n):
    gesture_count = []
    for gesture in gesture_labels: 
        data = getData(n, gesture)
        gesture_count.append(len(data))
    return gesture_count

def getLabels (n):
    n = participants[n-1]
    gesture_count = getGestures(n)
    emg_len = sum(gesture_count)
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(emg_len, numGestures))
    curr = 0
    for x in range (numGestures):
        #labels[curr:curr+gesture_count[x]][x] = 1.0
        for y in range(gesture_count[x]):
            labels[curr + y][x] = 1.0
        curr += gesture_count[x]
    return labels

def getGestures_separateSessions(args):
    subject_number, session_number = args
    subject_number = participants[int(subject_number)-1]
    session_number_mapping = {1: 'initial', 2: 'recalibration'}
    file = h5py.File(f'./Jehan_Dataset/p{subject_number:03}/data_allchannels_{session_number_mapping[session_number]}.h5', 'r')
    gesture_count = []
    for gesture in gesture_labels: 
        data = np.array(file[gesture])
        gesture_count.append(len(data))
    return gesture_count

def getLabels_separateSessions(args):
    subject_number, session_number = args
    gesture_count = getGestures_separateSessions((subject_number, session_number))
    emg_len = sum(gesture_count)
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(emg_len, numGestures))
    curr = 0
    for x in range (numGestures):
        for y in range(gesture_count[x]):
            labels[curr + y][x] = 1.0
        curr += gesture_count[x]
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
    resize_length_factor = 6
    if turn_on_magnitude:
        resize_length_factor = 3
    native_resnet_size = 224
    
    args = [(emg[i], cmap, length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
    # Using process_map instead of multiprocessing.Pool directly
    images = process_map(process_optimized_makeOneImage, args, chunksize=1, max_workers=4)

    if turn_on_magnitude:
        args = [(emg[i], length, width, resize_length_factor, native_resnet_size, global_min, global_max) for i in range(len(emg))]
        images_magnitude = process_map(process_optimized_makeOneMagnitudeImage, args, chunksize=1, max_workers=32)
        images = np.concatenate((images, images_magnitude), axis=2)
    
    if turn_on_cwt:
        raise NotImplementedError("CWT is not implemented yet.")
    
    if turn_on_spectrogram:
        raise NotImplementedError("Spectrogram is not implemented yet.")
    
    if turn_on_hht:
        raise NotImplementedError("HHT is not implemented yet.")
    
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

