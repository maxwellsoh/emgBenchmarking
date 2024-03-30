import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from scipy.signal import butter,filtfilt,iirnotch
#from PyEMD import EMD
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import multiprocessing
from tqdm import tqdm
import argparse
import random 
import utils_OzdemirEMG as utils
from sklearn.model_selection import StratifiedKFold
import os
import datetime
import matplotlib as mpl
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import timm
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import zarr
import diffusion_generated_zarr_loading as dgzl

## Argument parser with optional argumenets

# Create the parser
parser = argparse.ArgumentParser(description="Include arguments for running different trials")

# Add argument for dataset
parser.add_argument('--dataset', help='dataset to test. Set to OzdemirEMG by default', default="OzdemirEMG")
# Add argument for leftout subject
parser.add_argument('--leftout_subject', type=int, help='number of subject that is left out for cross validation. Set to 0 to run standard random held-out test. Set to 0 by default.', default=0)
# Add parser for seed
parser.add_argument('--seed', type=int, help='seed for reproducibility. Set to 0 by default.', default=0)
# Add number of epochs to train for
parser.add_argument('--epochs', type=int, help='number of epochs to train for. Set to 25 by default.', default=25)
# Add whether or not to use k folds (leftout_subject must be 0)
parser.add_argument('--turn_on_kfold', type=utils.str2bool, help='whether or not to use k folds cross validation. Set to False by default.', default=False)
# Add argument for stratified k folds cross validation
parser.add_argument('--kfold', type=int, help='number of folds for stratified k-folds cross-validation. Set to 5 by default.', default=5)
# Add argument for checking the index of the fold
parser.add_argument('--fold_index', type=int, help='index of the fold to use for cross validation (should be from 1 to --kfold). Set to 1 by default.', default=1)
# Add argument for whether or not to use cyclical learning rate
parser.add_argument('--turn_on_cyclical_lr', type=utils.str2bool, help='whether or not to use cyclical learning rate. Set to False by default.', default=False)
# Add argument for whether or not to use cosine annealing with warm restartfs
parser.add_argument('--turn_on_cosine_annealing', type=utils.str2bool, help='whether or not to use cosine annealing with warm restarts. Set to False by default.', default=False)
# Add argument for whether or not to use RMS
parser.add_argument('--turn_on_rms', type=utils.str2bool, help='whether or not to use RMS. Set to False by default.', default=False)
# Add argument for RMS input window size (resulting feature dimension to classifier)
parser.add_argument('--rms_input_windowsize', type=int, help='RMS input window size. Set to 1000 by default.', default=1000)
# Add argument for whether or not to concatenate magnitude image
parser.add_argument('--turn_on_magnitude', type=utils.str2bool, help='whether or not to concatenate magnitude image. Set to False by default.', default=False)
# Add argument for model to use
parser.add_argument('--model', type=str, help='model to use (e.g. \'convnext_tiny_custom\', \'convnext_tiny\', \'davit_tiny.msft_in1k\', \'efficientnet_b3.ns_jft_in1k\', \'vit_tiny_path16_224\', \'efficientnet_b0\'). Set to resnet50 by default.', default='resnet50')
# Add argument for project suffix
parser.add_argument('--project_name_suffix', type=str, help='suffix for project name. Set to empty string by default.', default='')
# Add argument for full or partial dataset for Ozdemir EMG dataset
parser.add_argument('--full_dataset_ozdemir', type=utils.str2bool, help='whether or not to use the full dataset for Ozdemir EMG Dataset. Set to False by default.', default=False)
# Add argument for using spectrogram transform
parser.add_argument('--turn_on_spectrogram', type=utils.str2bool, help='whether or not to use spectrogram transform. Set to False by default.', default=False)
# Add argument for using cwt
parser.add_argument('--turn_on_cwt', type=utils.str2bool, help='whether or not to use cwt. Set to False by default.', default=False)
# Add argument for using Hilbert Huang Transform
parser.add_argument('--turn_on_hht', type=utils.str2bool, help='whether or not to use HHT. Set to False by default.', default=False)
# Add argument for saving images
parser.add_argument('--save_images', type=utils.str2bool, help='whether or not to save images. Set to False by default.', default=False)
# Add argument to turn off scaler normalization
parser.add_argument('--turn_off_scaler_normalization', type=utils.str2bool, help='whether or not to turn off scaler normalization. Set to False by default.', default=False)
# Add argument to change learning rate
parser.add_argument('--learning_rate', type=float, help='learning rate. Set to 1e-4 by default.', default=1e-4)
# Add argument to specify which gpu to use (if any gpu exists)
parser.add_argument('--gpu', type=int, help='which gpu to use. Set to 0 by default.', default=0)
# Add argument to specify whether to load diffusion generated images
parser.add_argument('--load_diffusion_generated_images', type=utils.str2bool, help='whether or not to load diffusion generated images. Set to False by default.', default=False)
# Add argument to specify guidance scales for diffusion generated images
parser.add_argument('--guidance_scales', type=str, help='guidance scales for diffusion generated images. Set to 5,15,25,50 by default.', default="5,15,25,50")
# Add argument for loading just a few images from dataset for debugging
parser.add_argument('--load_few_images', type=utils.str2bool, help='whether or not to load just a few images from dataset for debugging. Set to False by default.', default=False)
# Add argument for reducing training data size while remaining stratified in terms of gestures and amount of data from each subject
parser.add_argument('--reduce_training_data_size', type=utils.str2bool, help='whether or not to reduce training data size while remaining stratified in terms of gestures and amount of data from each subject. Set to False by default.', default=False)
# Add argument for size of reduced training data
parser.add_argument('--reduced_training_data_size', type=int, help='size of reduced training data. Set to 56000 by default.', default=56000)

# Parse the arguments
args = parser.parse_args()

if (args.dataset == "uciEMG"):
    import utils_UCI as utils
    print(f"The dataset being tested is uciEMG")
    project_name = 'emg_benchmarking_uci'

elif (args.dataset == "ninapro-db2"):
    import utils_NinaproDB2 as utils
    print(f"The dataset being tested is ninapro-db2")
    project_name = 'emg_benchmarking_ninapro-db2'

elif (args.dataset == "ninapro-db5"):
    import utils_NinaproDB5 as utils
    print(f"The dataset being tested is ninapro-db5")
    project_name = 'emg_benchmarking_ninapro-db5'

else:
    print(f"The dataset being tested is OzdemirEMG")
    project_name = 'emg_benchmarking_ozdemir'
    if args.full_dataset_ozdemir:
        print(f"Using the full dataset for Ozdemir EMG")
        utils.gesture_labels = utils.gesture_labels_full
        utils.numGestures = len(utils.gesture_labels)
    else: 
        print(f"Using the partial dataset for Ozdemir EMG")
        utils.gesture_labels = utils.gesture_labels_partial
        utils.numGestures = len(utils.gesture_labels)

# Use the arguments
print(f"The value of --leftout_subject is {args.leftout_subject}")
print(f"The value of --seed is {args.seed}")
print(f"The value of --epochs is {args.epochs}")
print(f"The model to use is {args.model}")
if args.turn_on_kfold:
    if args.leftout_subject == 0:
        print(f"The value of --turn_on_kfold is {args.turn_on_kfold}")
    else: 
        print("Cannot turn on kfold if leftout_subject is not 0")
        exit()
    print(f"The value of --kfold is {args.kfold}")
    print(f"The value of --fold_index is {args.fold_index}")
    
if args.turn_on_cyclical_lr:
    print(f"The value of --turn_on_cyclical_lr is {args.turn_on_cyclical_lr}")
if args.turn_on_cosine_annealing:
    print(f"The value of --turn_on_cosine_annealing is {args.turn_on_cosine_annealing}")
if args.turn_on_cyclical_lr and args.turn_on_cosine_annealing:
    print("Cannot turn on both cyclical learning rate and cosine annealing")
    exit()
if args.turn_on_rms:
    print(f"The value of --turn_on_rms is {args.turn_on_rms}")
    print(f"The value of --rms_input_windowsize is {args.rms_input_windowsize}")
if args.turn_on_magnitude:
    print(f"The value of --turn_on_magnitude is {args.turn_on_magnitude}")
print(f"The value of --project_name_suffix is {args.project_name_suffix}")
print(f"The value of --turn_on_spectrogram is {args.turn_on_spectrogram}")
print(f"The value of --turn_on_cwt is {args.turn_on_cwt}")
print(f"The value of --turn_on_hht is {args.turn_on_hht}")

print(f"The value of --save_images is {args.save_images}")
print(f"The value of --turn_off_scaler_normalization is {args.turn_off_scaler_normalization}")
print(f"The value of --learning_rate is {args.learning_rate}")
print(f"The value of --gpu is {args.gpu}")
print(f"The value of --load_diffusion_generated_images is {args.load_diffusion_generated_images}")
print(f"The value of --guidance_scales is {args.guidance_scales}")

print(f"The value of --load_few_images is {args.load_few_images}")
print(f"The value of --reduce_training_data_size is {args.reduce_training_data_size}")
print(f"The value of --reduced_training_data_size is {args.reduced_training_data_size}")

args.guidance_scales = args.guidance_scales.split(",")
    
# Add date and time to filename
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

print("------------------------------------------------------------------------------------------------------------------------")
print("Starting run at", formatted_datetime)
print("------------------------------------------------------------------------------------------------------------------------")

# %%
# 0 for no LOSO; participants here are 1-13
leaveOut = int(args.leftout_subject)

# Set seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with  multiprocessing.Pool(processes=16) as pool:
    emg_async = pool.map_async(utils.getEMG, [(i+1) for i in range(utils.num_subjects)])
    emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
    
    labels_async = pool.map_async(utils.getLabels, [(i+1) for i in range(utils.num_subjects)])
    labels = labels_async.get()

length = len(emg[0][0])
width = len(emg[0][0][0])
print("Number of Electrode Channels: ", length)
print("Number of Timesteps per Trial:", width)

# These can be tuned to change the normalization
# This is the coefficient for the standard deviation
# used for the magnitude images. In practice, these
# should be fairly small so that images have more
# contrast
if args.turn_on_rms:
    # This tends to be small because in pracitice
    # the RMS is usually much smaller than the raw EMG
    # NOTE: Should check why this is the case
    sigma_coefficient = 0.1
else:
    # This tends to be larger because the raw EMG
    # is usually much larger than the RMS
    sigma_coefficient = 0.5

if (leaveOut == 0):
    if args.turn_on_kfold:
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        
        emg_in = np.concatenate([np.array(i.reshape(-1, length*width)) for i in emg], axis=0, dtype=np.float32)
        labels_in = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)
        
        labels_for_folds = np.argmax(labels_in, axis=1)
        
        fold_count = 1
        for train_index, test_index in skf.split(emg_in, labels_for_folds):
            if fold_count == args.fold_index:
                train_indices = train_index
                validation_indices = test_index
                break
            fold_count += 1

        # Normalize by electrode
        emg_in_by_electrode = emg_in[train_indices].reshape(-1, length, width)
        # s = preprocessing.StandardScaler().fit(emg_in[train_indices])
        global_min = emg_in[train_indices].mean() - sigma_coefficient*emg_in[train_indices].std()
        global_max = emg_in[train_indices].mean() + sigma_coefficient*emg_in[train_indices].std()

        # Assuming emg is your initial data of shape (SAMPLES, 16, 50)
        # Reshape data to (SAMPLES*50, 16)
        emg_reshaped = emg_in_by_electrode.reshape(-1, utils.numElectrodes)

        # Initialize and fit the scaler on the reshaped data
        # This will compute the mean and std dev for each electrode across all samples and features
        scaler = preprocessing.StandardScaler()
        scaler.fit(emg_reshaped)
        
        # Repeat means and std_devs for each time point using np.repeat
        scaler.mean_ = np.repeat(scaler.mean_, width)
        scaler.scale_ = np.repeat(scaler.scale_, width)
        scaler.var_ = np.repeat(scaler.var_, width)
        scaler.n_features_in_ = width*utils.numElectrodes

        del emg_in
        del labels_in

        del emg_in_by_electrode
        del emg_reshaped

    else: 
        # Reshape and concatenate EMG data
        # Flatten each subject's data from (TRIAL, CHANNEL, TIME) to (TRIAL, CHANNEL*TIME)
        # Then concatenate along the subject dimension (axis=0)
        emg_in = np.concatenate([np.array(i.reshape(-1, length*width)) for i in emg], axis=0, dtype=np.float16)
        labels_in = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)
        indices = np.arange(emg_in.shape[0])
        train_indices, validation_indices = model_selection.train_test_split(indices, test_size=0.2, stratify=labels_in)
        train_emg_in = emg_in[train_indices]  # Select only the train indices
        # s = preprocessing.StandardScaler().fit(train_emg_in)

        # Normalize by electrode
        emg_in_by_electrode = train_emg_in.reshape(-1, length, width)
        global_min = emg_in[train_indices].mean() - sigma_coefficient*emg_in[train_indices].std()
        global_max = emg_in[train_indices].mean() + sigma_coefficient*emg_in[train_indices].std()

        # Assuming emg is your initial data of shape (SAMPLES, 16, 50)
        # Reshape data to (SAMPLES*50, 16)
        emg_reshaped = emg_in_by_electrode.reshape(-1, utils.numElectrodes)

        # Initialize and fit the scaler on the reshaped data
        # This will compute the mean and std dev for each electrode across all samples and features
        scaler = preprocessing.StandardScaler()
        scaler.fit(emg_reshaped)
        
        # Repeat means and std_devs for each time point using np.repeat
        scaler.mean_ = np.repeat(scaler.mean_, width)
        scaler.scale_ = np.repeat(scaler.scale_, width)
        scaler.var_ = np.repeat(scaler.var_, width)
        scaler.n_features_in_ = width*utils.numElectrodes

        del emg_in
        del labels_in

        del train_emg_in
        del indices

        del emg_in_by_electrode
        del emg_reshaped

else: # Running LOSO
    emg_in = np.concatenate([np.array(i.view(len(i), length*width)) for i in emg[:(leaveOut-1)]] + [np.array(i.view(len(i), length*width)) for i in emg[leaveOut:]], axis=0, dtype=np.float32)
    # s = preprocessing.StandardScaler().fit(emg_in)
    global_min = emg_in.mean() - sigma_coefficient*emg_in.std()
    global_max = emg_in.mean() + sigma_coefficient*emg_in.std()

    # Normalize by electrode
    emg_in_by_electrode = emg_in.reshape(-1, length, width)

    # Assuming emg is your initial data of shape (SAMPLES, 16, 50)
    # Reshape data to (SAMPLES*50, 16)
    emg_reshaped = emg_in_by_electrode.reshape(-1, utils.numElectrodes)

    # Initialize and fit the scaler on the reshaped data
    # This will compute the mean and std dev for each electrode across all samples and features
    scaler = preprocessing.StandardScaler()
    scaler.fit(emg_reshaped)
    
    # Repeat means and std_devs for each time point using np.repeat
    scaler.mean_ = np.repeat(scaler.mean_, width)
    scaler.scale_ = np.repeat(scaler.scale_, width)
    scaler.var_ = np.repeat(scaler.var_, width)
    scaler.n_features_in_ = width*utils.numElectrodes

    del emg_in
    del emg_in_by_electrode
    del emg_reshaped


data = []

# add tqdm to show progress bar
print("Width of EMG data: ", width)
print("Length of EMG data: ", length)
base_foldername_zarr = f'LOSOimages_zarr/{args.dataset}/LOSO_subject' + str(leaveOut) + '/'
if args.turn_off_scaler_normalization:
        base_foldername_zarr = f'LOSOimages_zarr/{args.dataset}/LOSO_no_scaler_normalization/'
        scaler = None
if args.turn_on_rms:
    base_foldername_zarr += 'RMS_input_windowsize_' + str(args.RMS_input_windowsize) + '/'
elif args.turn_on_spectrogram:
    base_foldername_zarr += 'spectrogram/'
elif args.turn_on_cwt:
    base_foldername_zarr += 'cwt/'
elif args.turn_on_hht:
    base_foldername_zarr += 'hht/'
if args.save_images: 
    if not os.path.exists(base_foldername_zarr):
        os.makedirs(base_foldername_zarr)

for x in tqdm(range(len(emg)), desc="Number of Subjects "):
    subject_folder = f'LOSO_subject{x}/'
    foldername_zarr = base_foldername_zarr + subject_folder
    
    # Check if the folder (dataset) exists, load if yes, else create and save
    if os.path.exists(foldername_zarr):
        # Load the dataset
        dataset = zarr.open(foldername_zarr, mode='r')
        print(f"Loaded dataset for subject {x} from {foldername_zarr}")
        if args.load_few_images:
            data += [dataset[:10]]
        else: 
            data += [dataset[:]]
    else:
        # Get images and create the dataset
        images = utils.getImages(emg[x], scaler, length, width, 
                                 turn_on_rms=args.turn_on_rms, rms_windows=args.rms_input_windowsize, 
                                 turn_on_magnitude=args.turn_on_magnitude, global_min=global_min, global_max=global_max, 
                                 turn_on_spectrogram=args.turn_on_spectrogram, turn_on_cwt=args.turn_on_cwt, 
                                 turn_on_hht=args.turn_on_hht)
        images = np.array(images, dtype=np.float16)
        
        # Save the dataset
        if args.save_images:
            os.makedirs(foldername_zarr, exist_ok=True)
            dataset = zarr.open(foldername_zarr, mode='w', shape=images.shape, dtype=images.dtype, chunks=True)
            dataset[:] = images
            print(f"Saved dataset for subject {x} at {foldername_zarr}")
        else:
            print(f"Did not save dataset for subject {x} at {foldername_zarr} because save_images is set to False")
        data += [images]

print("------------------------------------------------------------------------------------------------------------------------")
print("NOTE: The width 224 is natively used in Resnet50, height is currently integer multiples of number of electrode channels ")
print("------------------------------------------------------------------------------------------------------------------------")
if leaveOut == 0:
    combined_labels = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)
    combined_images = np.concatenate([np.array(i) for i in data], axis=0, dtype=np.float16)
    X_train = combined_images[train_indices]
    Y_train = combined_labels[train_indices]
    X_validation = combined_images[validation_indices]
    Y_validation = combined_labels[validation_indices]
    X_validation, X_test, Y_validation, Y_test = model_selection.train_test_split(X_validation, Y_validation, test_size=0.5, stratify=Y_validation)
    del combined_images
    del combined_labels
    del data
    del emg

    X_train = torch.from_numpy(X_train).to(torch.float16)
    Y_train = torch.from_numpy(Y_train).to(torch.float16)
    X_validation = torch.from_numpy(X_validation).to(torch.float16)
    Y_validation = torch.from_numpy(Y_validation).to(torch.float16)
    X_test = torch.from_numpy(X_test).to(torch.float16)
    Y_test = torch.from_numpy(Y_test).to(torch.float16)
    print("Size of X_train:     ", X_train.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_train:     ", Y_train.size()) # (SAMPLE, GESTURE)
    print("Size of X_validation:", X_validation.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_validation:", Y_validation.size()) # (SAMPLE, GESTURE)
    print("Size of X_test:      ", X_test.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_test:      ", Y_test.size()) # (SAMPLE, GESTURE)
else:
    if args.reduce_training_data_size:
        reduced_size_per_subject = args.reduced_training_data_size // (utils.num_subjects - 1)
        
    X_validation = np.array(data.pop(leaveOut-1))
    Y_validation = np.array(labels.pop(leaveOut-1))
    for i in range(len(data)):
        current_data = np.array(data[i])
        current_labels = np.array(labels[i])
        
        if args.reduce_training_data_size:
            proportion_to_keep = reduced_size_per_subject / current_data.shape[0]
            current_data, _, current_labels, _ = model_selection.train_test_split(current_data, current_labels, 
                                                                                        train_size=proportion_to_keep, stratify=current_labels, 
                                                                                        random_state=args.seed, shuffle=True)
            
        if i == 0:
            X_train = current_data
            Y_train = current_labels
        else:
            X_train = np.concatenate((X_train, current_data), axis=0)
            Y_train = np.concatenate((Y_train, current_labels), axis=0)
        print("Appended subject", i+1, "to training data")

    X_train = torch.from_numpy(X_train).to(torch.float16)
    Y_train = torch.from_numpy(Y_train).to(torch.float16)
    X_validation = torch.from_numpy(X_validation).to(torch.float16)
    Y_validation = torch.from_numpy(Y_validation).to(torch.float16)
    print("Size of X_train:", X_train.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_train:", Y_train.size()) # (SAMPLE, GESTURE)
    print("Size of X_validation:", X_validation.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_validation:", Y_validation.size()) # (SAMPLE, GESTURE)
    
    del data
    del emg

model_name = args.model
if args.model == 'resnet50_custom':
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-4])
    # #model = nn.Sequential(*list(model.children())[:-4])
    num_features = model[-1][-1].conv3.out_channels
    # #num_features = model.fc.in_features
    dropout = 0.5
    model.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
    model.add_module('flatten', nn.Flatten())
    model.add_module('fc1', nn.Linear(num_features, 512))
    model.add_module('relu', nn.ReLU())
    model.add_module('dropout1', nn.Dropout(dropout))
    model.add_module('fc3', nn.Linear(512, utils.numGestures))
    model.add_module('softmax', nn.Softmax(dim=1))
elif args.model == 'resnet50':
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Replace the last fully connected layer
    num_ftrs = model.fc.in_features  # Get the number of input features of the original fc layer
    model.fc = nn.Linear(num_ftrs, utils.numGestures)  # Replace with a new linear layer
elif args.model == 'convnext_tiny_custom':
    # %% Referencing: https://medium.com/exemplifyml-ai/image-classification-with-resnet-convnext-using-pytorch-f051d0d7e098
    class LayerNorm2d(nn.LayerNorm):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 3, 1)
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x

    n_inputs = 768
    hidden_size = 128 # default is 2048
    n_outputs = utils.numGestures

    # model = timm.create_model(model_name, pretrained=True, num_classes=10)
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    #model = nn.Sequential(*list(model.children())[:-4])
    #model = nn.Sequential(*list(model.children())[:-3])
    #num_features = model[-1][-1].conv3.out_channels
    #num_features = model.fc.in_features
    dropout = 0.1 # was 0.5

    sequential_layers = nn.Sequential(
        LayerNorm2d((n_inputs,), eps=1e-06, elementwise_affine=True),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(n_inputs, hidden_size, bias=True),
        nn.BatchNorm1d(hidden_size),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, n_outputs),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = sequential_layers

else: 
    # model_name = 'efficientnet_b0'  # or 'efficientnet_b1', ..., 'efficientnet_b7'
    # model_name = 'tf_efficientnet_b3.ns_jft_in1k'
    model = timm.create_model(model_name, pretrained=True, num_classes=utils.numGestures)
    # # Load the Vision Transformer model
    # model_name = 'vit_base_patch16_224'  # This is just one example, many variations exist
    # model = timm.create_model(model_name, pretrained=True, num_classes=utils.numGestures)

num = 0
for name, param in model.named_parameters():
    num += 1
    if (num > 0):
    #if (num > 72): # for -3
    #if (num > 33): # for -4
        param.requires_grad = True
    else:
        param.requires_grad = False
        
if args.load_diffusion_generated_images:
    generated_images_grouped, generated_group_labels = dgzl.load_images(args.leftout_subject, args.guidance_scales, utils.gesture_labels)
    
    # Because images and labels are stored as tensors in a list, we need to append them to X_train and Y_train
    for i in range(len(generated_images_grouped)):
        X_train = torch.cat((X_train, generated_images_grouped[i].transpose(1, 3).to(torch.float16)))
        Y_train = torch.cat((Y_train, generated_group_labels[i]))

batch_size = 64
train_loader = DataLoader(list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=utils.seed_worker, pin_memory=True)
val_loader = DataLoader(list(zip(X_validation, Y_validation)), batch_size=batch_size, num_workers=4, worker_init_fn=utils.seed_worker, pin_memory=True)
if (leaveOut == 0):
    test_loader = DataLoader(list(zip(X_test, Y_test)), batch_size=batch_size, num_workers=4, worker_init_fn=utils.seed_worker, pin_memory=True)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
learn = args.learning_rate
optimizer = torch.optim.Adam(model.parameters(), lr=learn)

num_epochs = args.epochs
if args.turn_on_cosine_annealing:
    number_cycles = 5
    annealing_multiplier = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=utils.periodLengthForAnnealing(num_epochs, annealing_multiplier, number_cycles),
                                                                        T_mult=annealing_multiplier, eta_min=1e-5, last_epoch=-1)
elif args.turn_on_cyclical_lr:
    # Define the cyclical learning rate scheduler
    step_size = len(train_loader) * 6  # Number of iterations in half a cycle
    base_lr = 1e-4  # Minimum learning rate
    max_lr = 1e-3  # Maximum learning rate
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=step_size, mode='triangular2', cycle_momentum=False)

# Training loop
import gc
import datetime
gc.collect()
torch.cuda.empty_cache()

wandb_runname = 'CNN_seed-'+str(args.seed)
if args.turn_on_kfold:
    wandb_runname += '_k-fold-'+str(args.kfold)+'_fold-index-'+str(args.fold_index)
if args.turn_on_cyclical_lr:
    wandb_runname += '_cyclical-lr'
if args.turn_on_cosine_annealing: 
    wandb_runname += '_cosine-annealing'
if args.turn_on_rms:
    wandb_runname += '_rms-windows-'+str(args.rms_input_windowsize)
if args.turn_on_magnitude:  
    wandb_runname += '_magnitude'
if args.leftout_subject != 0:
    wandb_runname += '_LOSO-'+str(args.leftout_subject)
wandb_runname += '_' + model_name
if args.dataset == "OzdemirEMG":
    if args.full_dataset_ozdemir:
        wandb_runname += '_full-dataset'
    else:
        wandb_runname += '_partial-dataset'
if args.turn_on_spectrogram:
    wandb_runname += '_spectrogram'
if args.turn_on_cwt:
    wandb_runname += '_cwt'
if args.turn_on_hht:
    wandb_runname += '_hht'
if args.learning_rate != 1e-4:
    wandb_runname += '_lr-'+str(args.learning_rate)

if (leaveOut == 0):
    if args.turn_on_kfold:
        project_name += '_k-fold-'+str(args.kfold)
    else:
        project_name += '_heldout'
else:
    project_name += '_LOSO'
if args.load_diffusion_generated_images:
    project_name += '_diffusion-generated' + '_guidance-scales-' + '_'.join(args.guidance_scales)
    
if args.reduce_training_data_size:
    project_name += '_reduced-training-data-size-' + str(args.reduced_training_data_size)

project_name += args.project_name_suffix

run = wandb.init(name=wandb_runname, project=project_name, entity='jehanyang')
wandb.config.lr = learn


device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)

wandb.watch(model)

testrun_foldername = f'test/{project_name}/{wandb_runname}/{formatted_datetime}/'
# Make folder if it doesn't exist
if not os.path.exists(testrun_foldername):
    os.makedirs(testrun_foldername)
model_filename = f'{testrun_foldername}model_{formatted_datetime}.pth'

if leaveOut == 0:
    # Plot and log images
    utils.plot_average_images(X_test, np.argmax(Y_test.cpu().detach().numpy(), axis=1), utils.gesture_labels, testrun_foldername, args, formatted_datetime, 'test')
    utils.plot_first_fifteen_images(X_test, np.argmax(Y_test.cpu().detach().numpy(), axis=1), utils.gesture_labels, testrun_foldername, args, formatted_datetime, 'test')

utils.plot_average_images(X_validation, np.argmax(Y_validation.cpu().detach().numpy(), axis=1), utils.gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')
utils.plot_first_fifteen_images(X_validation, np.argmax(Y_validation.cpu().detach().numpy(), axis=1), utils.gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')

utils.plot_average_images(X_train, np.argmax(Y_train.cpu().detach().numpy(), axis=1), utils.gesture_labels, testrun_foldername, args, formatted_datetime, 'train')
utils.plot_first_fifteen_images(X_train, np.argmax(Y_train.cpu().detach().numpy(), axis=1), utils.gesture_labels, testrun_foldername, args, formatted_datetime, 'train')
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    model.train()
    train_acc = 0.0
    train_loss = 0.0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as t:
        for X_batch, Y_batch in t:
            X_batch = X_batch.to(device).to(torch.float32)
            Y_batch = Y_batch.to(device).to(torch.float32)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            Y_batch_long = torch.argmax(Y_batch, dim=1)
            train_acc += torch.mean((preds == Y_batch_long).type(torch.float)).item()

            # Optional: You can use tqdm's set_postfix method to display loss and accuracy for each batch
            # Update the inner tqdm loop with metrics
            # Only set_postfix every 10 batches to avoid slowing down the loop
            if t.n % 10 == 0:
                t.set_postfix({"Batch Loss": loss.item(), "Batch Acc": torch.mean((preds == Y_batch_long).type(torch.float)).item()})

            del X_batch, Y_batch, output, preds
            torch.cuda.empty_cache()

    # Validation
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device).to(torch.float32)
            Y_batch = Y_batch.to(device).to(torch.float32)

            #output = model(X_batch).logits
            output = model(X_batch)
            val_loss += criterion(output, Y_batch).item()
            preds = torch.argmax(output, dim=1)
            Y_batch_long = torch.argmax(Y_batch, dim=1)

            val_acc += torch.mean((preds == Y_batch_long).type(torch.float)).item()

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
        "Valid Acc": val_acc, 
        "Learning Rate": optimizer.param_groups[0]['lr']})

torch.save(model.state_dict(), model_filename)
wandb.save(f'model/modelParameters_{formatted_datetime}.pth')

# Testing
if (leaveOut == 0):
    pred = []
    true = []

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device).to(torch.float32)
            Y_batch = Y_batch.to(device).to(torch.float32)

            output = model(X_batch)
            test_loss += criterion(output, Y_batch).item()

            test_acc += np.mean(np.argmax(output.cpu().detach().numpy(), axis=1) == np.argmax(Y_batch.cpu().detach().numpy(), axis=1))

            output = np.argmax(output.cpu().detach().numpy(), axis=1)
            pred.extend(output)
            labels = np.argmax(Y_batch.cpu().detach().numpy(), axis=1)
            true.extend(labels)

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    wandb.log({
        "Test Loss": test_loss,
        "Test Acc": test_acc}) 
    
    
    # %% Confusion Matrix
    # Plot and log confusion matrix in wandb
    utils.plot_confusion_matrix(true, pred, utils.gesture_labels, testrun_foldername, args, formatted_datetime, 'test')

# Load validation in smaller batches for memory purposes
torch.cuda.empty_cache()  # Clear cache if needed

model.eval()
with torch.no_grad():
    validation_predictions = []
    for i, batch in tqdm(enumerate(torch.split(X_validation, split_size_or_sections=batch_size)), desc="Validation Batch Loading"):  # Or some other number that fits in memory
        batch = batch.to(device).to(torch.float32)
        outputs = model(batch)
        preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        validation_predictions.extend(preds)

utils.plot_confusion_matrix(np.argmax(Y_validation.cpu().detach().numpy(), axis=1), np.array(validation_predictions), utils.gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')   

# Load training in smaller batches for memory purposes
torch.cuda.empty_cache()  # Clear cache if needed

model.eval()
with torch.no_grad():
    train_predictions = []
    for i, batch in tqdm(enumerate(torch.split(X_train, split_size_or_sections=batch_size)), desc="Training Batch Loading"):  # Or some other number that fits in memory
        batch = batch.to(device).to(torch.float32)
        outputs = model(batch)
        preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        train_predictions.extend(preds)

utils.plot_confusion_matrix(np.argmax(Y_train.cpu().detach().numpy(), axis=1), np.array(train_predictions), utils.gesture_labels, testrun_foldername, args, formatted_datetime, 'train')
    
run.finish()
