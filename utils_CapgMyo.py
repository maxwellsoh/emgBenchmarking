import torch
import numpy as np
import pandas as pd
import random
from scipy import io
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
from tqdm import tqdm
import fcwt

numGestures = 8
fs = 1000 #Hz
wLen = 250 # ms
wLenTimesteps = int(wLen / 1000 * fs)
stepLen = 50 #50 ms
numElectrodes = 128
num_subjects = 10 # 10 subjects with 2 sessions each
cmap = mpl.colormaps['viridis']
gesture_labels = ["thumb up", "extension of index and middle, flexion of the others", "flexion of ring and little finger, extension of the others", 
"thumb opposing base of little finger", "abduction of all fingers", "fingers flexed together in fist", "pointing index", "adduction of extended fingers"]
participants_first_session_index = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
participants_second_session_index = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
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

# data is [# samples, # channels]
# target min/max is [# channels, # gestures]
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

def window (e):
    return e.unfold(dimension=0, size=wLen, step=50)

def filter(emg):
    # Butterworth bandpass filter
    b, a = butter(N=3, Wn=[20.0, 380.0], btype='bandpass', analog=False, fs=fs)
    emgButter = torch.from_numpy(filtfilt(b, a, emg, axis=2).copy())

    # second-order notch filter at 50 Hz
    b, a = iirnotch(w0=50.0, Q=0.0001, fs=fs)
    return torch.from_numpy(filtfilt(b, a, emgButter, axis=2).copy())

def fft_plot(signal):
    T = 1/fs  # Sampling interval
    L = len(signal)  # Length of the signal

    # Compute FFT
    fft_values = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(L, T)

    # Only take the positive half of the spectrum, since it's symmetric for real signals
    half_point = L // 2
    fft_values = fft_values[:half_point]
    frequencies = frequencies[:half_point]

    # Plot FFT with true frequencies
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies, np.abs(fft_values))
    plt.title('FFT of the Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

def getData (subject, gesture, trial): 

    if (isinstance(subject, int)):  
        sub = str(subject)
        if subject < 10:
            sub = '0' + sub
    else:
        sub = str(subject[0])
        if subject[0] < 10:
            sub = '0' + sub
        target_min = subject[1]
        target_max = subject[2]
        leftout = subject[3]

    name = '0' + sub + '-00' + str(gesture) + '-00' +str(trial)
    if trial == 10:
        name = '0' + sub + '-00' + str(gesture) + '-010'
    mat_data = io.loadmat('./CapgMyo_B/dbb-preprocessed-0' + sub + '/' + name + '.mat')
    # [# timesteps, # channels]
    mat_array = mat_data['data']

    if (not isinstance(subject, int) and leftout != subject[0]):
        mat_array = target_normalize(mat_array, target_min, target_max, gesture - 1)

    # stacks data per gesture for all trials
    tensor_data = torch.from_numpy(mat_array)
    if trial < 10:
        return torch.cat((window(tensor_data), getData(subject, gesture, trial + 1)), dim=0)
    elif gesture < 8:
        return torch.cat((window(tensor_data), getData(subject, gesture + 1, 1)), dim=0)
    else:
        return window(tensor_data)

def getEMG (x, session_number=1):
    #return torch.cat((getData(x-1,1,1), getData(x,1,1)), dim=0)
    subject_number = x[0] if isinstance(x, tuple) else x
    data_index = participants_first_session_index[subject_number-1] if session_number == 1 else participants_second_session_index[subject_number-1]
    if isinstance(x, tuple):
        return filter(getData((data_index, x[1], x[2], x[3]), 1, 1))
    return filter(getData(data_index, 1, 1))

def getEMG_separateSessions(args):
    if (len(args) == 2):
        subject_number, session_number = args
        mins = None
        maxes = None
        leftout = None
    else:
        subject_number, session_number, mins, maxes, leftout = args
    data_index = participants_first_session_index[subject_number-1] if session_number == 1 else participants_second_session_index[subject_number-1]
    if (len(args) == 2):
        return filter(getData(data_index, 1, 1))
    else:
        return filter(getData((data_index, mins, maxes, leftout), 1, 1))

def getExtrema (n, proportion, lastSessionOnly=False):
    """Returns the min/max EMG values for each electrode per gesture over a proportion of the windows of data. 

    Per gesture, accumulates data across each of its trials and windows this data. Then takes a proportion of the windows and calculates the min/max values for each electrode over these windows across all trials and time steps. 

    Args:
        n (int): subject number
        proportion: proportion of the windows to consider
        lastSessionOnly (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    mins = np.zeros((numElectrodes, numGestures))
    maxes = np.zeros((numElectrodes, numGestures))

    if lastSessionOnly:
        n = participants_second_session_index[n-1]
    else:
        n = participants_first_session_index[n-1]
    
    sub = str(n)
    if (n < 10):
        sub = '0' + sub

    for i in range(numGestures):
        data = []

        for trial in range(10):
            name = '0' + sub + '-00' + str(i+1) + '-00' +str(trial+1)
            if trial == 9:
                name = '0' + sub + '-00' + str(i+1) + '-010'
            mat_data = io.loadmat('./CapgMyo_B/dbb-preprocessed-0' + sub + '/' + name + '.mat')
            # data.append(mat_data['data'].transpose()) 

            data.append(mat_data['data']) # (TRIAL, TIME STEP, CHANNEL)

        tensor_data = torch.from_numpy(np.concatenate(data, axis=0)) # 
        # (TIME STEP, CHANNEL)
        windowed_data = window(tensor_data) # (WINDOW, CHANNEL, TIME STEP)

        num_windows = int(len(windowed_data) * proportion)
        selected_windows = windowed_data[:num_windows]

        for j in range(numElectrodes):
            mins[j][i] = torch.min(selected_windows[:, j, :])
            maxes[j][i] = torch.max(selected_windows[:, j, :])
            
    return mins, maxes


def getLabels (n):
    emg_len = len(getEMG(n))
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(emg_len, 8))
    for x in range (8):
        for y in range (int(emg_len / 8)):
            labels[int(x * (emg_len / 8) + y)][x] = 1.0
    return labels

def getLabels_separateSessions(args):
    subject_number, session_number = args
    data_index = participants_first_session_index[subject_number-1] if session_number == 1 else participants_second_session_index[subject_number-1]
    emg_len = len(getData(data_index, 1, 1))
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(emg_len, 8))
    for x in range (8):
        for y in range (int(emg_len / 8)):
            labels[int(x * (emg_len / 8) + y)][x] = 1.0
    return labels

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

