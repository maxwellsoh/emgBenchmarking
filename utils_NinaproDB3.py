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

fs = 2000 # Hz (SEMG signals samplign rate)

# not really sure what these three values are... were the same in both though
wLen = 250 # ms
wLenTimesteps = int(wLen / 1000 * fs)

# this is commented as 50 ms on DB2 and DB5 but have different values
stepLen = 100 

numElectrodes = 12 # number of EMG columns
num_subjects = 11


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
    numZero = 0
    indices = []
    #print(restimulus.shape)
    for x in range (len(restimulus)):
        L = torch.chunk(restimulus[x], 2, dim=1)
        if torch.equal(L[0], L[1]):
            if L[0][0][0] == 0:
                if (numZero < 550):
                    #print("working")
                    indices += [x]
                numZero += 1
            else:
                indices += [x]
    return indices

def contract(R):
    numGestures = R.max() + 1
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(len(R), numGestures))
    for x in range(len(R)):
        labels[x][int(R[x][0][0])] = 1.0
    return labels

def filter(emg):
    # sixth-order Butterworth highpass filter
    b, a = butter(N=1, Wn=999.0, btype='lowpass', analog=False, fs=2000.0)
    return torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())

def getRestim (n: int, exercise: int = 2):
    """
    Returns a restiumulus tensor for participant n and exercise exercise unfolded across time. 

    Args:
        n (int): _description_
        exercise (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    restim = torch.from_numpy(io.loadmat(f'./NinaproDB3/DB3_s{n}/S{n}_E{exercise}_A1.mat')['restimulus'])
    return restim.unfold(dimension=0, size=wLenTimesteps, step=stepLen)

def normalize (data, target_min, target_max, gesture):
    source_min = np.zeros(len(data[0]), dtype=np.float32)
    source_max = np.zeros(len(data[0]), dtype=np.float32)
    for i in range(len(data[0])):
        source_min[i] = np.min(data[:, i])
        source_max[i] = np.max(data[:, i])

    for i in range(len(data[0])):
        data[:, i] = ((data[:, i] - source_min[i]) / (source_max[i] 
        - source_min[i])) * (target_max[i][gesture] - target_min[i][gesture]) + target_min[i][gesture]
    return data

def getEMG (args):
    
    n, exercise = args
    restim = getRestim(n, exercise)
    #emg = pd.read_hdf(f'DatasetsProcessed_hdf5/NinaproDB5/s{n}/emgS{n}_E2.hdf5')
    #emg = torch.tensor(emg.values)
    emg = torch.from_numpy(io.loadmat(f'./NinaproDB3/DB3_s{n}/S{n}_E{exercise}_A1.mat')['emg']).to(torch.float16)

    return filter(emg.unfold(dimension=0, size=wLenTimesteps, step=stepLen)[balance(restim)])

def getLabels (args):
    n, exercise = args
    restim = getRestim(n, exercise)
    return contract(restim[balance(restim)])

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

    '''
    resize = transforms.Resize([length * resize_length_factor, native_resnet_size],
                               interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    image = resize(torch.from_numpy(image))
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image)

    # Displaying the image periodically
    if index is not None and index % display_interval == 0:
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))  # Adjusting to HxWxC for display
        plt.title(f"Image at index {index}")
        plt.show(block=False)

    return image.numpy().astype(np.float32)
    '''

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
    
    if turn_on_cwt:
        raise NotImplementedError("CWT is not implemented yet.")
    
    if turn_on_spectrogram:
        raise NotImplementedError("Spectrogram is not implemented yet.")
    
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

    # resize average images to 224 x 224
    resize = transforms.Resize([224, 224], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

    print(f"Plotting first fifteen {partition_name} images...")
    for i in range(total_gestures):
        # Find indices of the first 15 images for gesture i
        gesture_indices = np.where(true_np == i)[0][:rows_per_gesture]

        current_images = torch.tensor(np.array([resize(img) for img in image_data[gesture_indices]]))
        
        # Select and denormalize only the required images
        gesture_images = denormalize(current_images).cpu().detach().numpy()

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



