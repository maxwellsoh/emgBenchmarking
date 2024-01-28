# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from functools import partial
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from scipy.signal import butter,filtfilt
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import scipy
import h5py
import random
from random import gauss
import math
import multiprocessing
import time
import gc
from itertools import chain
import argparse
from tqdm import tqdm
import matplotlib as mpl
from tqdm import tqdm
mpl.use('Agg')  # Use a non-interactive backend like 'Agg'
import matplotlib.pyplot as plt
import zarr
import os
import timm
import utils_NinaproDB2 as ut_NDB2
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.model_selection import train_test_split
import logging
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed

logging.basicConfig(filename='error_log.log', level=logging.DEBUG, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
## Argument parser with optional argumenets

# Create the parser
parser = argparse.ArgumentParser(description="Include arguments for running different trials")

# Add an optional argument
parser.add_argument('--leftout_subject', type=int, help='number of subject that is left out for cross validation. Set to 0 to run standard random held-out test. Set to 0 by default.', default=0)
parser.add_argument('--seed', type=int, help='number of seed that is used for randomization. Set to 0 by default.', default=0)
parser.add_argument('--save_images', type=ut_NDB2.str2bool, help='whether to save the RMS images. Set to false by default.', default=False)
parser.add_argument('--model', type=str, help='model to use. Set to resnet50 by default.', default='resnet50')
parser.add_argument('--epochs', type=int, help='number of epochs to train for. Set to 50 by default.', default=50)
parser.add_argument('--turn_on_rms', type=ut_NDB2.str2bool, help='whether to use RMS images. Set to false by default.', default=False)
parser.add_argument('--rms_input_windowsize', type=int, help='RMS input window size. Set to 1000 by default.', default=1000)
parser.add_argument('--window_size_in_ms', type=int, help='window size in ms. Set to 250 by default.', default=250)
parser.add_argument('--downsample_factor', type=int, help='downsample factor, should be multiple of 1. Set to 1 by default.', default=1)
parser.add_argument('--freeze_model', type=ut_NDB2.str2bool, help='whether to freeze the model. Set to false by default.', default=False)
parser.add_argument('--number_hidden_classifier_layers', type=int, help='number of hidden classifier layers. Set to 0 by default.', default=0)
parser.add_argument('--hidden_classifier_layer_size', type=int, help='size of hidden classifier layer. Set to 256 by default.', default=256)

# Parse the arguments
args = parser.parse_args()

# Use the arguments
print(f"The value of --leftout_subject is {args.leftout_subject}")
print(f"The value of --seed is {args.seed}")
print(f"The value of --save_images is {args.save_images}")
print(f"The value of --model is {args.model}")
print(f"The value of --epochs is {args.epochs}")
print(f"The value of --turn_on_rms is {args.turn_on_rms}")
print(f"The value of --rms_input_windowsize is {args.rms_input_windowsize}")
print(f"The value of --window_size_in_ms is {args.window_size_in_ms}")
print(f"The value of --downsample_factor is {args.downsample_factor}")
print(f"The value of --freeze_model is {args.freeze_model}")
print("\n")

# %%
# 0 for no LOSO; participants here are 1-13
leaveOut = int(args.leftout_subject)

# root mean square instances per channel per image
#RMS_input_windowsize = 500 # must be a factor of 1000
RMS_input_windowsize = args.rms_input_windowsize # must be a factor of 1000

# image width - must be multiple of 64
width = 64

number_channels = 64

# gaussian Noise signal-to-noise ratio
SNR = 15

# magnitude warping std
std = 0.05

window_length_in_milliseconds = args.window_size_in_ms #ms
step_length_in_milliseconds = 50 #ms
sampling_frequency = 4000 #Hz

# Set seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

milliseconds_in_second = 1000
window_size_in_timesteps = int(window_length_in_milliseconds/milliseconds_in_second*sampling_frequency)
step_size_in_timesteps = int(step_length_in_milliseconds/milliseconds_in_second*sampling_frequency)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

### Data Extraction
class DataExtract:
    gestures = ['abduct_p1', 'adduct_p1', 'extend_p1', 'grip_p1', 'pronate_p1', 'rest_p1', 'supinate_p1', 'tripod_p1', 'wextend_p1', 'wflex_p1']

    def highpassFilter (self, emg):
        b, a = butter(N=1, Wn=120.0, btype='highpass', analog=False, fs=sampling_frequency)
        # what axis should the filter apply to? other datasets have axis=0
        return torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=-1).copy())

    # returns array with dimensions (# of samples)x64x10x100 [SAMPLES, CHANNELS, GESTURES, TIME]
    def getData(self, n, gesture):
        milliseconds_in_second = 1000
        if (n<10):
            file = h5py.File('./Jehan_Dataset/p00' + str(n) +'/data_allchannels_initial.h5', 'r')
        else:
            file = h5py.File('./Jehan_Dataset/p0' + str(n) +'/data_allchannels_initial.h5', 'r')
        data = self.highpassFilter(torch.from_numpy(np.array(file[gesture])).unfold( # [SAMPLE, CHANNEL, TIME]
                        dimension=-1, 
                        size=int(window_length_in_milliseconds/milliseconds_in_second*sampling_frequency), # window length in time steps
                        step=int(step_length_in_milliseconds/milliseconds_in_second*sampling_frequency))) # [SAMPLE, CHANNEL, WINDOW, TIME]
        if (args.turn_on_rms):
            data = data.unfold(
                    dimension=-1, 
                    size=int(window_length_in_milliseconds/(milliseconds_in_second*RMS_input_windowsize)*sampling_frequency), # RMS window length
                    step=int(window_length_in_milliseconds/(milliseconds_in_second*RMS_input_windowsize)*sampling_frequency)) # [SAMPLE, CHANNEL, WINDOW, RMS_WINDOW, TIME]
        else: 
            data = data.unsqueeze(3) # [SAMPLE, CHANNEL, WINDOW, 1, TIME]

        # Downsample tensor by downsample_factor
        if args.downsample_factor != 1:
            data = data[:, :, :, :, ::args.downsample_factor]

        return torch.cat([data[i] for i in range(len(data))], axis=1).permute([1, 0, 2, 3]) # [SAMPLE, CHANNEL, RMS_WINDOW or 1, TIME]


    def getEMG(self, n):
        if args.turn_on_rms:
            return torch.cat([torch.sqrt(torch.mean(self.getData(n, name) ** 2, dim=3)) for name in self.gestures], axis=0)
        else: 
            return torch.cat([self.getData(n, name) for name in self.gestures], axis=0).squeeze()

    def getGestures(self, n):
        if (n<10):
            file = h5py.File('./Jehan_Dataset/p00' + str(n) +'/data_allchannels_initial.h5', 'r')
        else:
            file = h5py.File('./Jehan_Dataset/p0' + str(n) +'/data_allchannels_initial.h5', 'r')

        numGestures = []
        milliseconds_in_second = 1000
        for gesture in self.gestures: 
            data = self.highpassFilter(torch.from_numpy(np.array(file[gesture]))\
                .unfold(dimension=-1, 
                        size=int(window_length_in_milliseconds/milliseconds_in_second*sampling_frequency), 
                        step=int(step_length_in_milliseconds/milliseconds_in_second*sampling_frequency)))
            if args.turn_on_rms:
                data = data.unfold(dimension=-1,
                        size=int(window_length_in_milliseconds/(milliseconds_in_second*RMS_input_windowsize)*sampling_frequency), 
                        step=int(window_length_in_milliseconds/(milliseconds_in_second*RMS_input_windowsize)*sampling_frequency))
            else:
                data = data.unsqueeze(3)
            numGestures += [len(data)]
        return numGestures

### Data Augmentation
class DataAugment:

    # gaussian noise
    def addNoise (emg):
        for i in range(len(emg)):
            emg[i] += gauss(0.0, math.sqrt((emg[i] ** 2) / SNR))
        return emg

    # magnitude warping
    def magWarp (emg):
        '''
        if (len(data_noRMS) == 0):
            data_noRMS = torch.cat([getData(currParticipant, name) for name in gestures], axis=0)
        emg = data_noRMS[n].view(64, window_length_in_milliseconds
    *4)
        '''

        cs = scipy.interpolate.CubicSpline([i*25 for i in range(RMS_input_windowsize//25+1)], [gauss(1.0, std) for i in range(RMS_input_windowsize//25+1)])
        scaleFact = cs([i for i in range(RMS_input_windowsize)])
        for i in range(RMS_input_windowsize):
            for j in range(64):
                emg[i*64 + j] = emg[i*64 + j] * scaleFact[i]
                #emg[i + j*RMS_input_windowsize] = emg[i + j*RMS_input_windowsize] * scaleFact[i]
        '''
        for i in range(len(scaleFact)):
            emg[:, i] = emg[:, i] * scaleFact[i]
        '''
        return emg
        #return torch.sqrt(torch.mean(emg.unfold(dimension=-1, size=int(window_length_in_milliseconds
        #/(1000*RMS_input_windowsize)*sampling_frequency), step=int(window_length_in_milliseconds
        #/(1000*RMS_input_windowsize)*sampling_frequency)) ** 2, dim=2)).view([64*RMS_input_windowsize])

    # electrode offseting
    def shift_up (batch):
        batch_up = batch.view(4, 16, RMS_input_windowsize).clone()
        for k in range(len(batch_up)):
            for j in range(len(batch_up[k])-1):
                batch_up[k][j] = batch_up[k][j+1]
        return batch_up

    def shift_down (batch):
        batch_down = batch.view(4, 16, RMS_input_windowsize).clone()
        for k in range(len(batch_down)):
            for j in range(len(batch_down[k])-1):
                batch_down[k][len(batch_down[k])-j-1] = batch_down[k][len(batch_down[k])-j-2]
        return batch_down

### Data Processing
class DataProcessing:
    # raw emg data -> 64x(RMS_input_windowsize) image
    cmap = mpl.colormaps['viridis']
    order = list(chain.from_iterable([[[k for k in range(64)][(i+j*16+32) % 64] for j in range(4)] for i in range(16)]))
    
    def dataToImage(self, emg_sample):
        try:
            emg_sample = emg_sample.squeeze()
            emg_sample -= torch.min(emg_sample)
            emg_sample /= torch.max(emg_sample)

            window_size = RMS_input_windowsize if args.turn_on_rms else window_size_in_timesteps

            emg_sample = emg_sample.view(64, window_size)
            emg_sample = torch.stack([emg_sample[i] for i in self.order])

            # Preallocate frames array for efficiency
            frames = [None] * window_size
            for i in range(window_size):
                frame = np.array(list(map(lambda x: self.cmap(x[i]), emg_sample.numpy())), dtype=np.float32)
                frames[i] = np.transpose(frame, axes=[1, 0])[:3]

            image = torch.from_numpy(np.transpose(np.stack(frames), axes=[1, 2, 0]))
            norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            resize_transform = transforms.Resize(size=[width, window_size], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

            image = resize_transform(norm_transform(image))

            return image.numpy().astype(np.float32)

        except Exception as e:
            logging.exception("Error occurred in dataToImage")
            raise e  # Re-raise the exception for further handling if necessary


    # image generation with augmentation
    dataCopies = 1

    def oneWindowImages(self, emg):
        """ Would usually include a list of augmented images, but for now, only returns the original image """
        combinedImages = []

        combinedImages.append(self.dataToImage(emg))
        #allImages.append(dataToImage(shift_up(batch)))
        #allImages.append(dataToImage(shift_down(batch)))
        #for j in range(3):
            #combinedImages.append(dataToImage(addNoise(emg)))
            #combinedImages.append(dataToImage(magWarp(emg)))

        return combinedImages

    # curr = 0
    def getImages(self, emg):
        pbar = tqdm(total=len(emg), desc="Augmented Image Generation")
        allImages = []

        # Define a callback function to collect results and update the progress bar
        def collect_result(result):
            allImages.append(result)
            pbar.update()

        with multiprocessing.Pool() as pool:
            for i in range(len(emg)):
                # Use collect_result as the callback to append results
                pool.apply_async(self.oneWindowImages, args=(emg[i],), callback=collect_result)
                    
            pool.close()  # Close the pool to any more tasks
            pool.join()   # Wait for all worker processes to exit

        

        pbar.close()

        '''
        if i % 1000 == 0:
            print("progress: " + str(i) + "/" + str(len(emg)))
            #print(labels[i])
            plt.imshow(allImages[i*dataCopies].T, origin='lower')
            plt.axis('off')
            plt.show()
        '''
        return allImages

    # no augmentation image generation

    # def getImages_noAugment(self, emg_sample):
    #     allImages = []
    #     pbar = tqdm(total=len(emg_sample), desc="Non-Augmented Image Generation")

    #     # def collect_result(result):
    #     #     allImages.append(result)
    #     #     print("progress: " + str(len(allImages)) + "/" + str(len(emg_sample)))
    #         #pbar.update()

    #     # with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as pool:  # Use half of available CPU cores
    #     #     results = [pool.apply_async(self.dataToImage, args=(sample,), callback=collect_result) for sample in emg_sample]

    #     #     pool.close()  # Close the pool to any more tasks
    #     #     pool.join()   # Wait for all worker processes to exit

    #         # pbar.update(1)
    #     for i in range(len(emg_sample)):
    #         allImages.append(self.dataToImage(emg_sample[i]))
    #         pbar.update(1)

    #     return allImages
    def getImages_noAugment(self, emg_sample):
        # Parallel processing using Joblib
        
        # Wrap emg_sample with tqdm for progress reporting
        emg_sample_wrapped = tqdm(emg_sample, desc="Processing", total=len(emg_sample))

        # Use Joblib to parallel process the data
        results = Parallel(n_jobs=-1)(delayed(self.dataToImage)(sample) for sample in emg_sample_wrapped)

        return results
    
# extracting raw EMG data

participants = [8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22]  # Example participant IDs
data_extract = DataExtract()  # Your existing DataExtract object

# Initialize the progress bars
pbar_emg = tqdm(total=len(participants), desc="EMG data extraction")
pbar_gestures = tqdm(total=len(participants), desc="Number of gestures data extraction", leave=False)

# Define callback functions to update the progress bars
def update_pbar_emg(result):
    pbar_emg.update()

def update_pbar_gestures(result):
    pbar_gestures.update()

# Start the multiprocessing pool
with multiprocessing.Pool() as pool:
    # Asynchronously apply tasks for EMG data extraction and update pbar_emg
    emg_results = [pool.apply_async(data_extract.getEMG, args=(participant,), callback=update_pbar_emg) for participant in participants] 
    
    # Asynchronously apply tasks for Gestures data extraction and update pbar_gestures
    gesture_results = [pool.apply_async(data_extract.getGestures, args=(participant,), callback=update_pbar_gestures) for participant in participants]

    # Wait for all EMG data extraction tasks to complete
    emg = [result.get() for result in emg_results]
    pbar_emg.close()

    # Wait for all Gestures data extraction tasks to complete
    numGestures = [result.get() for result in gesture_results]
    pbar_gestures.close()

print("\nData extraction complete")
print("\n")

# generating labels

labels = []

windowsPerSample = math.ceil((8000 - window_size_in_timesteps) / (step_size_in_timesteps - 1))  

for nums in tqdm(numGestures, desc="Label Generation"):
    sub_labels = torch.tensor(()).new_zeros(size=(sum(nums)*windowsPerSample, 10))
    subGestures = [(i * windowsPerSample) for i in nums]
    index = 0
    count = 0

    for i in range(len(sub_labels)):
        sub_labels[i][index] = 1.0
        count += 1
        if (count >= subGestures[index]):
            index += 1
            count = 0

    labels += [sub_labels]
labels = list(labels)
print("Labels generated")
print("\n")

# LOSO-CV data processing

data_process = DataProcessing()

if leaveOut != 0:
    foldername_zarr = 'LOSOimages_zarr/JehanDataset/LOSO_subject' + str(leaveOut) + '/'
    
    emg_subject_leftout = emg[leaveOut-1]
    emg_leftin = emg.copy()
    emg_leftin.pop(leaveOut-1)
    emg_scaling_subjects_leftin = np.concatenate([np.array(i.view(len(i), 64*RMS_input_windowsize)) for i in emg_leftin], axis=0, dtype=np.float16)
    
    labels_subject_leftout = labels[leaveOut-1]

    standard_scalar = preprocessing.StandardScaler().fit(emg_scaling_subjects_leftin )
    emg_scaled_subject_leftout = torch.from_numpy(standard_scalar.transform(np.array(emg_subject_leftout.view(len(emg_subject_leftout), 64*RMS_input_windowsize))))
    del emg_scaling_subjects_leftin 
    del emg_subject_leftout
    
    # load zarr images if they exist
    if (os.path.exists(foldername_zarr + 'val_data_LOSO' + str(leaveOut) + '.zarr')):
        X_validation_np = zarr.load(foldername_zarr + 'val_data_LOSO' + str(leaveOut) + '.zarr')
        Y_validation_np = zarr.load(foldername_zarr + 'val_labels_LOSO' + str(leaveOut) + '.zarr')
        X_validation = torch.from_numpy(X_validation_np).to(torch.float16)
        Y_validation = torch.from_numpy(Y_validation_np).to(torch.float16)
        print("Validation images found. Validation images loaded for left out subject " + str(leaveOut))
    else:
        print("Generating validation images for left out subject " + str(leaveOut))
        X_validation = torch.tensor(np.array(data_process.getImages_noAugment(emg_scaled_subject_leftout))).to(torch.float16)
        Y_validation = torch.from_numpy(np.array(labels_subject_leftout)).to(torch.float16)
        # Convert the PyTorch tensors to NumPy and ensure the type is compatible withf Zarr
        X_validation_np = X_validation.numpy().astype(np.float16)
        Y_validation_np = Y_validation.numpy().astype(np.float16)

        # Save the numpy arrays using Zarr
        if (args.save_images):
            zarr.save(foldername_zarr + 'val_data_LOSO' + str(leaveOut) + '.zarr', X_validation_np)
            zarr.save(foldername_zarr + 'val_labels_LOSO' + str(leaveOut) + '.zarr', Y_validation_np)
            print("Validation images generated for left out subject " + str(leaveOut) + " and saved")
    print("\n")
    
    del X_validation_np
    del Y_validation_np

    del participants[leaveOut-1]
    del emg_scaled_subject_leftout

    X_train_all = []
    Y_train_all = []
    
    # load zarr images if they exist
    for subject in range(len(emg)):
        if subject + 1 == leaveOut:
            continue
        
        if os.path.exists(foldername_zarr + 'train_data_LOSO_subject' + str(subject+1) + '.zarr'):
            X_train_np = zarr.load(foldername_zarr + 'train_data_LOSO_subject' + str(subject+1) + '.zarr')
            Y_train_np = zarr.load(foldername_zarr + 'train_labels_LOSO_subject' + str(subject+1) + '.zarr')
            print("Training images found. Training images loaded for subject", subject+1)
        else:
            print("Generating training images for subject", subject+1)
            X_train_subject = torch.from_numpy(np.array(data_process.getImages(torch.from_numpy(standard_scalar.transform(np.array(emg[subject].view(len(emg[subject]), 64*RMS_input_windowsize))))))
                                            .astype(np.float16)).to(torch.float16)
            Y_train_subject = torch.from_numpy(np.repeat(np.array(labels[subject]), data_process.dataCopies, axis=0)).to(torch.float16)
            X_train_np = X_train_subject.numpy().astype(np.float16)
            Y_train_np = Y_train_subject.numpy().astype(np.float16)
            if (args.save_images):
                zarr.save(foldername_zarr + 'train_data_LOSO_subject' + str(subject+1) + '.zarr', X_train_np)
                zarr.save(foldername_zarr + 'train_labels_LOSO_subject' + str(subject+1) + '.zarr', Y_train_np)
                print("Training images generated and saved for subject", subject+1)

        # Append the subject's data to the accumulating lists
        X_train_all.append(X_train_np)
        Y_train_all.append(Y_train_np)
        
    print("\n")

    # Concatenate all the subject data into single arrays
    X_train = np.concatenate(X_train_all, axis=0).astype(np.float16)
    Y_train = np.concatenate(Y_train_all, axis=0).astype(np.float16)
    
    # Optionally convert back to PyTorch tensors if you will continue processing with PyTorch
    X_train = torch.from_numpy(X_train).to(torch.float16).squeeze()
    Y_train = torch.from_numpy(Y_train).to(torch.float16).squeeze()

    del X_train_np
    del Y_train_np
    del X_train_all
    del Y_train_all
    
# non-LOSO data processing

else:
    # emg is of dimensions [SUBJECT, SAMPLE, CHANNEL, TIME]
    # Split the dataset into training, validation, and test sets
    # You'll need to define your own splitting logic here based on your dataset
    test_split_ratio = 0.2  # For example, 20% for testing

    # Flatten and concatenate all EMG data
    all_emg = np.concatenate([np.array(i.view(len(i), -1)) for i in emg], axis=0)

    # Flatten and concatenate all labels
    all_labels = np.concatenate(labels, axis=0)

    # Split the data into training and testing sets
    train_emg, test_emg, train_labels, test_labels = train_test_split(all_emg, all_labels, test_size=test_split_ratio, shuffle=True, random_state=args.seed)
    test_emg, val_emg, test_labels, val_labels = train_test_split(test_emg, test_labels, test_size=0.5, shuffle=True, random_state=args.seed)
    
    # Standardize the data
    standard_scalar = preprocessing.StandardScaler().fit(train_emg)
    
    # Apply the scaler to training, validation, and test sets
    if args.turn_on_rms:
        train_emg_scaled = [torch.from_numpy(standard_scalar.transform(subject.reshape(-1, 64*RMS_input_windowsize))) for subject in train_emg]
        val_emg_scaled = [torch.from_numpy(standard_scalar.transform(subject.reshape(-1, 64*RMS_input_windowsize))) for subject in val_emg]
        test_emg_scaled = [torch.from_numpy(standard_scalar.transform(subject.reshape(-1, 64*RMS_input_windowsize)))for subject in test_emg]
    else:
        train_emg_scaled = [torch.from_numpy(standard_scalar.transform(subject.reshape(-1, 64*window_size_in_timesteps))) for subject in train_emg]
        val_emg_scaled = [torch.from_numpy(standard_scalar.transform(subject.reshape(-1, 64*window_size_in_timesteps))) for subject in val_emg]
        test_emg_scaled = [torch.from_numpy(standard_scalar.transform(subject.reshape(-1, 64*window_size_in_timesteps)))for subject in test_emg]
    
    debug_number = int(1e2)

    # Generate images (or your specific data processing) for training, validation, and test sets
    X_train = torch.tensor(np.array(data_process.getImages_noAugment(train_emg_scaled[:debug_number]))).to(torch.float16)
    X_validation = torch.tensor(np.array(data_process.getImages_noAugment(val_emg_scaled[:debug_number]))).to(torch.float16)
    X_test = torch.tensor(np.array(data_process.getImages_noAugment(test_emg_scaled[:debug_number]))).to(torch.float16)

    # Convert labels to tensors and possibly perform additional processing
    Y_train = torch.stack([torch.tensor(labels) for labels in train_labels[:debug_number]])
    Y_validation = torch.stack([torch.tensor(labels) for labels in val_labels[:debug_number]])
    Y_test = torch.stack([torch.tensor(labels) for labels in test_labels[:debug_number]])

    # Concatenate data for each set if necessary
    # This step depends on how you want to structure your data for training
    # X_train = torch.concat(X_train).to(torch.float16).squeeze()
    # Y_train = torch.concat(Y_train).to(torch.float16).squeeze()
    # X_val = torch.concat(X_val).to(torch.float16).squeeze()
    # Y_val = torch.concat(Y_val).to(torch.float16).squeeze()
    # X_test = torch.concat(X_test).to(torch.float16).squeeze()
    # Y_test = torch.concat(Y_test).to(torch.float16).squeeze()

    # At this point, you have your data ready for a standard training/validation/test procedure.


print("Size of X_train:     ", X_train.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
print("Size of Y_train:     ", Y_train.size()) # (SAMPLE, GESTURE)
print("Size of X_validation:", X_validation.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
print("Size of Y_validation:", Y_validation.size()) # (SAMPLE, GESTURE)
if leaveOut == 0:
    print("Size of X_test:      ", X_test.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_test:      ", Y_test.size()) # (SAMPLE, GESTURE)

numGestureTypes = len(labels[0][0])

# %% Referencing: https://medium.com/exemplifyml-ai/image-classification-with-resnet-convnext-using-pytorch-f051d0d7e098
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

n_inputs = 768
hidden_size = 128 # default is 2048
n_outputs = 10

if args.model == 'resnet50_custom':
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-4])
    # #model = nn.Sequential(*list(model.children())[:-4])
    num_features = model[-1][-1].conv3.out_channels
    # #num_features = model.fc.in_features
    dropout = 0.5
    model.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
    model.add_module('flatten', nn.Flatten())
    '''
    model.add_module('fc1', nn.Linear(num_features, 1024))
    model.add_module('relu', nn.ReLU())
    model.add_module('dropout1', nn.Dropout(dropout))
    model.add_module('fc2', nn.Linear(1024, 1024))
    model.add_module('relu2', nn.ReLU())
    model.add_module('dropout2', nn.Dropout(dropout))
    model.add_module('fc3', nn.Linear(1024, ut_NDB2.numGestures))
    '''
    model.add_module('fc1', nn.Linear(num_features, 512))
    model.add_module('relu', nn.ReLU())
    model.add_module('dropout1', nn.Dropout(dropout))
    model.add_module('fc3', nn.Linear(512, numGestureTypes))
    model.add_module('softmax', nn.Softmax(dim=1))
elif args.model == 'resnet50':
    # Load the pre-trained ResNet50 model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Replace the last fully connected layer
    num_ftrs = model.fc.in_features  # Get the number of input features of the original fc layer
    model.fc = nn.Linear(num_ftrs, numGestureTypes)  # Replace with a new linear layer
    
elif args.model == 'convnext_tiny_custom':
    # %% Referencing: https://medium.com/exemplifyml-ai/image-classification-with-resnet-convnext-using-pytorch-f051d0d7e098
    class LayerNorm2d(nn.LayerNorm):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 3, 1)
            x = x.permute(0, 3, 1, 2)
            return x

    n_inputs = 256
    hidden_size = 256 # default is 2048
    n_outputs = numGestureTypes
    
    model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    model.features = model.features[:-4]
    norm_layer = partial(LayerNorm2d, eps=1e-6)

    # model = timm.create_model(model_name, pretrained=True, num_classes=10)
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    #model = nn.Sequential(*list(model.children())[:-4])
    #model = nn.Sequential(*list(model.children())[:-3])
    #num_features = model[-1][-1].conv3.out_channels
    #num_features = model.fc.in_features
    dropout = 0.5 # was 0.5

    sequential_layers = nn.Sequential(
        norm_layer(n_inputs),
        nn.Flatten(1),
        nn.Linear(n_inputs, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, n_outputs),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = sequential_layers

else: 
    # model_name = 'efficientnet_b0'  # or 'efficientnet_b1', ..., 'efficientnet_b7'
    # model_name = 'tf_efficientnet_b3.ns_jft_in1k'
    model = timm.create_model(args.model, pretrained=True, num_classes=numGestureTypes)
    # # Load the Vision Transformer model
    # model_name = 'vit_base_patch16_224'  # This is just one example, many variations exist
    # model = timm.create_model(model_name, pretrained=True, num_classes=ut_NDB2.numGestures)
    
def find_last_layer(module):
    children = list(module.children())
    if len(children) == 0:
        return module
    else: 
        return find_last_layer(children[-1])

last_layer = find_last_layer(model)
if args.freeze_model:
    for param in model.parameters():
        param.requires_grad = False
    
    print("Last layer: ", last_layer)

    # Unfreeze the last layer if it has parameters
    if hasattr(last_layer, 'parameters'):
        for param in last_layer.parameters():
            param.requires_grad = True
            
if args.number_hidden_classifier_layers > 0:
    # Determine in_features for the last layer
    if isinstance(last_layer, nn.Linear):
        in_features = last_layer.in_features
    else:
        raise Exception("Last layer is not a linear layer. Please check the model architecture.")
    
    def remove_last_layer(module):
        children = list(module.children())
        if len(children) == 0:
            # Base case: module has no children
            return module
        else:
            # If the child is a leaf (has no children), remove it
            if len(list(children[-1].children())) == 0:
                return nn.Sequential(*children[:-1])
            else:
                # Otherwise, recursively remove the last layer from the last child
                new_children = children[:-1] + [remove_last_layer(children[-1])]
                return nn.Sequential(*new_children)

    # Remove the last layer
    model = remove_last_layer(model)

    layers = []
    for hidden_size in range(args.number_hidden_classifier_layers):
        layers.append(nn.Linear(in_features, args.hidden_classifier_layer_size))
        layers.append(nn.ReLU())
        in_features = args.hidden_classifier_layer_size
        
    # Add the last layer
    layers.append(nn.Linear(in_features, numGestureTypes))
    
    new_layers = nn.Sequential(*layers)
        
    model.add_module('classifier_layers', new_layers)
    
    print(model)
    
class Data(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing the image
    # Add any other transformations you need here
])

# Apply the transform to your datasets
train_dataset = ut_NDB2.CustomDataset(X_train, Y_train, transform=transform)
val_dataset = ut_NDB2.CustomDataset(X_validation, Y_validation, transform=transform)
test_dataset = ut_NDB2.CustomDataset(X_test, Y_test, transform=transform) if leaveOut == 0 else None

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=ut_NDB2.seed_worker, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, worker_init_fn=ut_NDB2.seed_worker, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, worker_init_fn=ut_NDB2.seed_worker, pin_memory=True)


print("number of batches: ", len(train_loader))

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
learn = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learn)

# %%
# Training loop
gc.collect()
torch.cuda.empty_cache()

wandb_runname = 'CNN_seed-' + str(args.seed)
if leaveOut != 0:
    wandb_runname += '_LOSO-' + str(args.leftout_subject)     
wandb_runname += '_' + args.model
if args.freeze_model:
    wandb_runname += '_freeze'
if args.number_hidden_classifier_layers > 0:
    wandb_runname += '_hidden-' + str(args.number_hidden_classifier_layers) + '-' + str(args.hidden_classifier_layer_size)
    
if leaveOut != 0:
    run = wandb.init(name=wandb_runname, project='emg_benchmarking_LOSO_JehanDataset', entity='jehanyang')
else:
    run = wandb.init(name=wandb_runname, project='emg_benchmarking_heldout_JehanDataset', entity='jehanyang')
wandb.config.lr = learn

num_epochs = args.epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

wandb.watch(model)

for epoch in tqdm(range(num_epochs), desc="Epoch"):
    model.train()
    train_acc = 0.0
    train_loss = 0.0
    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(device).to(torch.float32)
        Y_batch = Y_batch.to(device).to(torch.long)

        optimizer.zero_grad()
        #output = model(X_batch).logits
        output = model(X_batch)
        loss = criterion(output, Y_batch)
        train_loss += loss.item()

        train_acc += np.mean(np.argmax(np.argmax(output.cpu().detach().numpy(),
                                       axis=1), axis=1) == np.argmax(Y_batch.cpu().detach().numpy(), axis=1))

        loss.backward()
        optimizer.step()

        del X_batch, Y_batch
        torch.cuda.empty_cache()

    # Validation
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device).to(torch.float32)
            Y_batch = Y_batch.to(device).to(torch.long)

            #output = model(X_batch).logits
            output = model(X_batch)
            val_loss += criterion(output, Y_batch).item()

            val_acc += np.mean(np.argmax(np.argmax(output.cpu().detach().numpy(), axis=1), axis=1) == np.argmax(Y_batch.cpu().detach().numpy(), axis=1))

            del X_batch, Y_batch
            torch.cuda.empty_cache()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")
    #print(f"{val_acc:.4f}")
    wandb.log({
        "Epoch": epoch,
        "Train Loss": train_loss,
        "Train Acc": train_acc,
        "Valid Loss": val_loss,
        "Valid Acc": val_acc, })

#run.finish()

if (leaveOut == 0):
    # Testing
    pred = []
    true = []

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device).to(torch.float32)
            Y_batch = Y_batch.to(device).to(torch.long)

            output = model(X_batch)
            test_loss += criterion(output, Y_batch).item()

            test_acc += np.mean(np.argmax(np.argmax(output.cpu().detach().numpy(), axis=1), axis=1) == np.argmax(Y_batch.cpu().detach().numpy(), axis=1))

            output = np.argmax(output.cpu().detach().numpy(), axis=1)
            pred.extend(output)
            labels = np.argmax(Y_batch.cpu().detach().numpy(), axis=1)
            true.extend(labels)

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    wandb.log({        
        "Test Loss": test_loss,
        "Test Acc": test_acc, })

    cf_matrix = confusion_matrix(true, np.argmax(pred, axis=-1))
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = np.arange(1, 11, 1),
                        columns = np.arange(1, 11, 1))
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, fmt=".3f")
    plt.savefig('output.png')
    wandb.log({"Confusion Matrix": wandb.Image(plt)})


