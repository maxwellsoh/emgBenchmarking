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
import cross_validation_utilities.train_test_split as tts # custom train test split to split stratified without shuffling
from torchvision.utils import save_image
import json
import shutil
import gc
import datetime
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer, split_ssl_data, BasicDataset
from semilearn.core.utils import send_model_cuda
from PIL import Image

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

## Argument parser with optional argumenets

# Create the parser
parser = argparse.ArgumentParser(description="Include arguments for running different trials")

# Add argument for dataset
parser.add_argument('--dataset', help='dataset to test. Set to OzdemirEMG by default', default="OzdemirEMG")
# Add argument for doing leave-one-subject-out
parser.add_argument('--leave_one_subject_out', type=utils.str2bool, help='whether or not to do leave one subject out. Set to False by default.', default=False)
# Add argument for leftout subject
parser.add_argument('--leftout_subject', type=int, help='number of subject that is left out for cross validation, starting from subject 1', default=0)
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
# Add argument for whether or not to use cosine annealing with warm restarts
parser.add_argument('--turn_on_cosine_annealing', type=utils.str2bool, help='whether or not to use cosine annealing with warm restarts. Set to False by default.', default=False)
# Add argument for whether or not to use RMS
parser.add_argument('--turn_on_rms', type=utils.str2bool, help='whether or not to use RMS. Set to False by default.', default=False)
# Add argument for RMS input window size (resulting feature dimension to classifier)
parser.add_argument('--rms_input_windowsize', type=int, help='RMS input window size. Set to 1000 by default.', default=1000)
# Add argument for whether or not to concatenate magnitude image
parser.add_argument('--turn_on_magnitude', type=utils.str2bool, help='whether or not to concatenate magnitude image. Set to False by default.', default=False)
# Add argument for model to use
parser.add_argument('--model', type=str, help='model to use (e.g. \'convnext_tiny_custom\', \'convnext_tiny\', \'davit_tiny.msft_in1k\', \'efficientnet_b3.ns_jft_in1k\', \'vit_tiny_patch16_224\', \'efficientnet_b0\'). Set to resnet50 by default.', default='resnet50')
# Add argument for exercises to include
parser.add_argument('--exercises', type=list_of_ints, help='List the exercises of the 3 to load. The most popular for benchmarking seem to be 2 and 3. Can format as \'--exercises 1,2,3\'', default=[1, 2, 3])
# Add argument for project suffix
parser.add_argument('--project_name_suffix', type=str, help='suffix for project name. Set to empty string by default.', default='')
# Add argument for full or partial dataset for Ozdemir EMG dataset
parser.add_argument('--full_dataset_ozdemir', type=utils.str2bool, help='whether or not to use the full dataset for Ozdemir EMG Dataset. Set to False by default.', default=False)
# Add argument for partial dataset for Ninapro DB2 and DB5
parser.add_argument('--partial_dataset_ninapro', type=utils.str2bool, help='whether or not to use the partial dataset for Ninapro DB2 and DB5. Set to False by default.', default=False)
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
# Add argument for loading just a few images from dataset for debugging
parser.add_argument('--load_few_images', type=utils.str2bool, help='whether or not to load just a few images from dataset for debugging. Set to False by default.', default=False)
# Add argument for reducing training data size while remaining stratified in terms of gestures and amount of data from each subject
parser.add_argument('--reduce_training_data_size', type=utils.str2bool, help='whether or not to reduce training data size while remaining stratified in terms of gestures and amount of data from each subject. Set to False by default.', default=False)
# Add argument for size of reduced training data
parser.add_argument('--reduced_training_data_size', type=int, help='size of reduced training data. Set to 56000 by default.', default=56000)
# Add argument to leve n subjects out randomly
parser.add_argument('--leave_n_subjects_out_randomly', type=int, help='number of subjects to leave out randomly. Set to 0 by default.', default=0)
# use target domain for normalization
parser.add_argument('--target_normalize', type=utils.str2bool, help='use a leftout window for normalization. Set to False by default.', default=False)
# Test with transfer learning by using some data from the validation dataset
parser.add_argument('--transfer_learning', type=utils.str2bool, help='use some data from the validation dataset for transfer learning. Set to False by default.', default=False)
# Add argument for cross validation for time series
parser.add_argument('--cross_validation_for_time_series', type=utils.str2bool, help='whether or not to use cross validation for time series. Set to False by default.', default=False)
# Add argument for proportion of left-out-subject data to use for transfer learning
parser.add_argument('--proportion_transfer_learning', type=float, help='proportion of left-out-subject data to use for transfer learning. Set to 0.25 by default.', default=0.25)
# Add argument for amount for reducing number of data to generate for transfer learning
parser.add_argument('--reduce_data_for_transfer_learning', type=int, help='amount for reducing number of data to generate for transfer learning. Set to 1 by default.', default=1)
# Add argument for whether to do leave-one-session-out
parser.add_argument('--leave_one_session_out', type=utils.str2bool, help='whether or not to leave one session out. Set to False by default.', default=False)
# Add argument for whether to do held_out test
parser.add_argument('--held_out_test', type=utils.str2bool, help='whether or not to do held out test. Set to False by default.', default=False)
# Add argument for whether to use only the subject left out for training in leave out session test
parser.add_argument('--one_subject_for_training_set_for_session_test', type=utils.str2bool, help='whether or not to use only the subject left out for training in leave out session test. Set to False by default.', default=False)
# Add argument for pretraining on all data from other subjects, and fine-tuning on some data from left out subject
parser.add_argument('--pretrain_and_finetune', type=utils.str2bool, help='whether or not to pretrain on all data from other subjects, and fine-tune on some data from left out subject. Set to False by default.', default=False)
# Add argument for finetuning epochs
parser.add_argument('--finetuning_epochs', type=int, help='number of epochs to fine-tune for. Set to 25 by default.', default=25)
# Add argument for whether or not to turn on unlabeled domain adaptation
parser.add_argument('--turn_on_unlabeled_domain_adaptation', type=utils.str2bool, help='whether or not to turn on unlabeled domain adaptation methods. Set to False by default.', default=False)
# Add argument to specify algorithm to use for unlabeled domain adaptation
parser.add_argument('--unlabeled_algorithm', type=str, help='algorithm to use for unlabeled domain adaptation. Set to "flexmatch" by default.', default="flexmatch")
# Add argument to specify proportion from left-out-subject to keep as unlabeled data
parser.add_argument('--proportion_unlabeled_data', type=float, help='proportion of data from left-out-subject to keep as unlabeled data. Set to 0.75 by default.', default=0.75)
# Add argument to specify batch size
parser.add_argument('--batch_size', type=int, help='batch size. Set to 64 by default.', default=64)

# Parse the arguments
args = parser.parse_args()

exercises = False

if (args.dataset == "uciEMG"):
    import utils_UCI as utils
    print(f"The dataset being tested is uciEMG")
    project_name = 'emg_benchmarking_uci'

elif (args.dataset == "ninapro-db2"):
    import utils_NinaproDB2 as utils
    print(f"The dataset being tested is ninapro-db2")
    project_name = 'emg_benchmarking_ninapro-db2'
    exercises = True
    if args.leave_one_session_out:
        ValueError("leave-one-session-out not implemented for ninapro-db2; only one session exists")

elif (args.dataset == "ninapro-db5"):
    import utils_NinaproDB5 as utils
    print(f"The dataset being tested is ninapro-db5")
    project_name = 'emg_benchmarking_ninapro-db5'
    exercises = True
    if args.leave_one_session_out:
        ValueError("leave-one-session-out not implemented for ninapro-db5; only one session exists")

elif (args.dataset == "M_dataset"):
    import utils_M_dataset as utils
    print(f"The dataset being tested is M_dataset")
    project_name = 'emg_benchmarking_M_dataset'
    if args.leave_one_session_out:
        ValueError("leave-one-session-out not implemented for M_dataset; only one session exists")

elif (args.dataset == "hyser"):
    import utils_hyser as utils
    print(f"The dataset being tested is hyser")
    project_name = 'emg_benchmarking_hyser'

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
    if args.leave_one_session_out:
        ValueError("leave-one-session-out not implemented for OzdemirEMG; only one session exists")

# Use the arguments
print(f"The value of --leftout_subject is {args.leftout_subject}")
print(f"The value of --seed is {args.seed}")
print(f"The value of --epochs is {args.epochs}")
print(f"The model to use is {args.model}")
if args.turn_on_kfold:
    print(f"The value of --turn_on_kfold is {args.turn_on_kfold}")
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
if exercises:
    print(f"The value of --exercises is {args.exercises}")
print(f"The value of --project_name_suffix is {args.project_name_suffix}")
print(f"The value of --turn_on_spectrogram is {args.turn_on_spectrogram}")
print(f"The value of --turn_on_cwt is {args.turn_on_cwt}")
print(f"The value of --turn_on_hht is {args.turn_on_hht}")

print(f"The value of --save_images is {args.save_images}")
print(f"The value of --turn_off_scaler_normalization is {args.turn_off_scaler_normalization}")
print(f"The value of --learning_rate is {args.learning_rate}")
print(f"The value of --gpu is {args.gpu}")

print(f"The value of --load_few_images is {args.load_few_images}")
print(f"The value of --reduce_training_data_size is {args.reduce_training_data_size}")
print(f"The value of --reduced_training_data_size is {args.reduced_training_data_size}")

print(f"The value of --leave_n_subjects_out_randomly is {args.leave_n_subjects_out_randomly}")
print(f"The value of --target_normalize is {args.target_normalize}")
print(f"The value of --transfer_learning is {args.transfer_learning}")
print(f"The value of --cross_validation_for_time_series is {args.cross_validation_for_time_series}")
print(f"The value of --proportion_transfer_learning is {args.proportion_transfer_learning}")
print(f"The value of --reduce_data_for_transfer_learning is {args.reduce_data_for_transfer_learning}")
print(f"The value of --leave_one_session_out is {args.leave_one_session_out}")
print(f"The value of --held_out_test is {args.held_out_test}")
print(f"The value of --one_subject_for_training_set_for_session_test is {args.one_subject_for_training_set_for_session_test}")
print(f"The value of --pretrain_and_finetune is {args.pretrain_and_finetune}")
print(f"The value of --finetuning_epochs is {args.finetuning_epochs}")

print(f"The value of --turn_on_unlabeled_domain_adaptation is {args.turn_on_unlabeled_domain_adaptation}")
print(f"The value of --unlabeled_algorithm is {args.unlabeled_algorithm}")
print(f"The value of --proportion_unlabeled_data is {args.proportion_unlabeled_data}")

print(f"The value of --batch_size is {args.batch_size}")
    
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

    
if exercises:
    emg = []
    labels = []

    if args.partial_dataset_ninapro:
        if args.dataset == "ninapro-db2":
            args.exercises = [1]
        elif args.dataset == "ninapro-db5":
            args.exercises = [2]

    with multiprocessing.Pool() as pool:
        for exercise in args.exercises:
            emg_async = pool.map_async(utils.getEMG, list(zip([(i+1) for i in range(utils.num_subjects)], exercise*np.ones(utils.num_subjects).astype(int))))
            emg.append(emg_async.get()) # (EXERCISE SET, SUBJECT, TRIAL, CHANNEL, TIME)
            
            labels_async = pool.map_async(utils.getLabels, list(zip([(i+1) for i in range(utils.num_subjects)], exercise*np.ones(utils.num_subjects).astype(int))))
            labels.append(labels_async.get())
            
            assert len(emg[-1]) == len(labels[-1]), "Number of trials for EMG and labels do not match"
            
    # Append exercise sets together and add dimensions to labels if necessary

    new_emg = []  # This will store the concatenated data for each subject
    new_labels = []  # This will store the concatenated labels for each subject
    numGestures = 0 # This will store the number of gestures for each subject

    for subject in range(utils.num_subjects): 
        subject_trials = []  # List to store trials for this subject across all exercise sets
        subject_labels = []  # List to store labels for this subject across all exercise sets
        
        for exercise_set in range(len(emg)):  
            # Append the trials of this subject in this exercise set
            subject_trials.append(emg[exercise_set][subject])
            subject_labels.append(labels[exercise_set][subject])

        concatenated_trials = np.concatenate(subject_trials, axis=0)  # Concatenate trials across exercise sets
        
        total_number_labels = 0
        for i in range(len(subject_labels)):
            total_number_labels += subject_labels[i].shape[1]
            
        # Convert from one hot encoding to labels
        # Assuming labels are stored separately and need to be concatenated end-to-end
        labels_set = []
        index_to_start_at = 0
        for i in range(len(subject_labels)):
            subject_labels_to_concatenate = [x + index_to_start_at if x != 0 else 0 for x in np.argmax(subject_labels[i], axis=1)]
            if args.dataset == "ninapro-db5":
                index_to_start_at = max(subject_labels_to_concatenate)
            labels_set.append(subject_labels_to_concatenate)

        if args.partial_dataset_ninapro:
            desired_gesture_labels = utils.partial_gesture_indices
        
        # Assuming labels are stored separately and need to be concatenated end-to-end
        concatenated_labels = np.concatenate(labels_set, axis=0) # (TRIAL)

        if args.partial_dataset_ninapro:
            indices_for_partial_dataset = np.array([indices for indices, label in enumerate(concatenated_labels) if label in desired_gesture_labels])
            concatenated_labels = concatenated_labels[indices_for_partial_dataset]
            concatenated_trials = concatenated_trials[indices_for_partial_dataset]
            # convert labels to indices
            label_to_index = {label: index for index, label in enumerate(desired_gesture_labels)}
            concatenated_labels = [label_to_index[label] for label in concatenated_labels]
        
        numGestures = len(np.unique(concatenated_labels))

        # Convert to one hot encoding
        concatenated_labels = np.eye(np.max(concatenated_labels) + 1)[concatenated_labels] # (TRIAL, GESTURE)

        # Append the concatenated trials to the new_emg list
        new_emg.append(concatenated_trials)
        new_labels.append(concatenated_labels)

    emg = [torch.from_numpy(emg_np) for emg_np in new_emg]
    labels = [torch.from_numpy(labels_np) for labels_np in new_labels]

else: # Not exercises
    if (args.target_normalize):
        mins, maxes = utils.getExtrema(args.leftout_subject + 1)
        with multiprocessing.Pool() as pool:
            if args.leave_one_session_out:
                NotImplementedError("leave-one-session-out not implemented with target_normalize yet")
            emg_async = pool.map_async(utils.getEMG, [(i+1, mins, maxes, args.leftout_subject + 1) for i in range(utils.num_subjects)])
            emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
            
            labels_async = pool.map_async(utils.getLabels, [(i+1) for i in range(utils.num_subjects)])
            labels = labels_async.get()

    else: # Not target_normalize
        #with multiprocessing.pool.ThreadPool() as pool:
        with multiprocessing.Pool() as pool:
            if args.leave_one_session_out: # based on 2 sessions for each subject
                total_number_of_sessions = 2
                emg = []
                labels = []
                for i in range(1, total_number_of_sessions+1):
                    emg_async = pool.map_async(utils.getEMG_separateSessions, [(j+1, i) for j in range(utils.num_subjects)])
                    emg.extend(emg_async.get())
                    
                    labels_async = pool.map_async(utils.getLabels_separateSessions, [(j+1, i) for j in range(utils.num_subjects)])
                    labels.extend(labels_async.get())
                
            else: # Not leave one session out
                emg_async = pool.map_async(utils.getEMG, [(i+1) for i in range(utils.num_subjects)])
                emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
                
                labels_async = pool.map_async(utils.getLabels, [(i+1) for i in range(utils.num_subjects)])
                labels = labels_async.get()

    print("subject 1 mean", torch.mean(emg[0]))
    numGestures = utils.numGestures

length = emg[0].shape[1]
width = emg[0].shape[2]
print("Number of Samples (across all participants): ", sum([e.shape[0] for e in emg]))
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
    
leaveOutIndices = []
# Generate scaler for normalization
if args.leave_n_subjects_out_randomly != 0 and (not args.turn_off_scaler_normalization and not args.target_normalize):
    leaveOut = args.leave_n_subjects_out_randomly
    print(f"Leaving out {leaveOut} subjects randomly")
    # subject indices to leave out randomly
    leaveOutIndices = np.random.choice(range(utils.num_subjects), leaveOut, replace=False)
    print(f"Leaving out subjects {np.sort(leaveOutIndices)}")
    emg_in = np.concatenate([np.array(i.view(len(i), length*width)) for i in emg if i not in leaveOutIndices], axis=0, dtype=np.float32)
    
    global_low_value = emg_in.mean() - sigma_coefficient*emg_in.std()
    global_high_value = emg_in.mean() + sigma_coefficient*emg_in.std()
    
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

else: # Not leave n subjects out randomly
    if (args.held_out_test):
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
            global_low_value = emg_in[train_indices].mean() - sigma_coefficient*emg_in[train_indices].std()
            global_high_value = emg_in[train_indices].mean() + sigma_coefficient*emg_in[train_indices].std()

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
            emg_in = np.concatenate([np.array(i.reshape(-1, length*width)) for i in emg], axis=0, dtype=np.float32)
            labels_in = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)
            indices = np.arange(emg_in.shape[0])
            train_indices, validation_indices = model_selection.train_test_split(indices, test_size=0.2, stratify=labels_in)
            train_emg_in = emg_in[train_indices]  # Select only the train indices
            # s = preprocessing.StandardScaler().fit(train_emg_in)

            # Normalize by electrode
            emg_in_by_electrode = train_emg_in.reshape(-1, length, width)
            global_low_value = emg_in[train_indices].mean() - sigma_coefficient*emg_in[train_indices].std()
            global_high_value = emg_in[train_indices].mean() + sigma_coefficient*emg_in[train_indices].std()

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

    elif (not args.turn_off_scaler_normalization and not args.target_normalize): # Running LOSO standardization
        emg_in = np.concatenate([np.array(i.view(len(i), length*width)) for i in emg[:(leaveOut-1)]] + [np.array(i.view(len(i), length*width)) for i in emg[leaveOut:]], axis=0, dtype=np.float32)
        # s = preprocessing.StandardScaler().fit(emg_in)
        global_low_value = emg_in.mean() - sigma_coefficient*emg_in.std()
        global_high_value = emg_in.mean() + sigma_coefficient*emg_in.std()

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

    else: 
        global_low_value = None
        global_high_value = None
        scaler = None
        
data = []

# add tqdm to show progress bar
print("Width of EMG data: ", width)
print("Length of EMG data: ", length)

if args.leave_n_subjects_out_randomly != 0:
    base_foldername_zarr = f'leave_n_subjects_out_randomly_images_zarr/{args.dataset}/leave_{args.leave_n_subjects_out_randomly}_subjects_out_randomly_seed-{args.seed}/'
else:
    if args.held_out_test:
        base_foldername_zarr = f'heldout_images_zarr/{args.dataset}/'
    elif args.leave_one_session_out:
        base_foldername_zarr = f'Leave_one_session_out_images_zarr/{args.dataset}/'
    elif args.turn_off_scaler_normalization:
        base_foldername_zarr = f'LOSOimages_zarr/{args.dataset}/'
    elif args.leave_one_subject_out:
        base_foldername_zarr = f'LOSOimages_zarr/{args.dataset}/'

if args.turn_off_scaler_normalization:
    if args.leave_n_subjects_out_randomly != 0:
        base_foldername_zarr = base_foldername_zarr + 'leave_n_subjects_out_randomly_no_scaler_normalization/'
    else: 
        if args.held_out_test:
            base_foldername_zarr = base_foldername_zarr + 'no_scaler_normalization/'
        else: 
            base_foldername_zarr = base_foldername_zarr + 'LOSO_no_scaler_normalization/'
    scaler = None
else:
    base_foldername_zarr = base_foldername_zarr + 'LOSO_subject' + str(leaveOut) + '/'

if args.turn_on_rms:
    base_foldername_zarr += 'RMS_input_windowsize_' + str(args.rms_input_windowsize) + '/'
elif args.turn_on_spectrogram:
    base_foldername_zarr += 'spectrogram/'
elif args.turn_on_cwt:
    base_foldername_zarr += 'cwt/'
elif args.turn_on_hht:
    base_foldername_zarr += 'hht/'

if exercises:
    if args.partial_dataset_ninapro:
        base_foldername_zarr += 'partial_dataset_ninapro/'
    else:
        exercises_numbers_filename = '-'.join(map(str, args.exercises))
        base_foldername_zarr += f'exercises{exercises_numbers_filename}/'
    
if args.save_images: 
    if not os.path.exists(base_foldername_zarr):
        os.makedirs(base_foldername_zarr)

for x in tqdm(range(len(emg)), desc="Number of Subjects "):
    if args.held_out_test:
        subject_folder = f'subject{x}/'
    else:
        subject_folder = f'LOSO_subject{x}/'
    foldername_zarr = base_foldername_zarr + subject_folder
    
    print("Attempting to load dataset for subject", x, "from", foldername_zarr)

    print("Looking in folder: ", foldername_zarr)
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
        print(f"Could not find dataset for subject {x} at {foldername_zarr}")
        # Get images and create the dataset
        if (args.target_normalize):
            scaler = None
        images = utils.getImages(emg[x], scaler, length, width, 
                                 turn_on_rms=args.turn_on_rms, rms_windows=args.rms_input_windowsize, 
                                 turn_on_magnitude=args.turn_on_magnitude, global_min=global_low_value, global_max=global_high_value, 
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

if args.leave_n_subjects_out_randomly != 0:
    
    # Instead of the below code, leave n subjects out randomly to be used as the 
    # validation set and the rest as the training set using leaveOutIndices
    
    X_validation = np.concatenate([np.array(data[i]) for i in range(utils.num_subjects) if i in leaveOutIndices], axis=0, dtype=np.float16)
    Y_validation = np.concatenate([np.array(labels[i]) for i in range(utils.num_subjects) if i in leaveOutIndices], axis=0, dtype=np.float16)
    X_validation = torch.from_numpy(X_validation).to(torch.float16)
    Y_validation = torch.from_numpy(Y_validation).to(torch.float16)
    
    X_train = np.concatenate([np.array(data[i]) for i in range(utils.num_subjects) if i not in leaveOutIndices], axis=0, dtype=np.float16)
    Y_train = np.concatenate([np.array(labels[i]) for i in range(utils.num_subjects) if i not in leaveOutIndices], axis=0, dtype=np.float16)
    X_train = torch.from_numpy(X_train).to(torch.float16)
    Y_train = torch.from_numpy(Y_train).to(torch.float16)
    
    print("Size of X_train:", X_train.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_train:", Y_train.size()) # (SAMPLE, GESTURE)
    print("Size of X_validation:", X_validation.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_validation:", Y_validation.size()) # (SAMPLE, GESTURE)
    
    del data
    del emg
    del labels

else: 
    if args.held_out_test:
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
    elif args.leave_one_session_out:
        total_number_of_sessions = 2
        left_out_subject_last_session_index = (total_number_of_sessions - 1) * utils.num_subjects + leaveOut-1
        left_out_subject_first_n_sessions_indices = [i for i in range(total_number_of_sessions * utils.num_subjects) if i % utils.num_subjects == (leaveOut-1) and i != left_out_subject_last_session_index]
        print("left_out_subject_last_session_index:", left_out_subject_last_session_index)
        print("left_out_subject_first_n_sessions_indices:", left_out_subject_first_n_sessions_indices)
        X_pretrain = np.concatenate([np.array(data[i]) for i in range(total_number_of_sessions * utils.num_subjects) if i != left_out_subject_last_session_index and i not in left_out_subject_first_n_sessions_indices], axis=0, dtype=np.float16)
        Y_pretrain = np.concatenate([np.array(labels[i]) for i in range(total_number_of_sessions * utils.num_subjects) if i != left_out_subject_last_session_index and i not in left_out_subject_first_n_sessions_indices], axis=0, dtype=np.float16)
        X_finetune = np.concatenate([np.array(data[i]) for i in left_out_subject_first_n_sessions_indices], axis=0, dtype=np.float16)
        Y_finetune = np.concatenate([np.array(labels[i]) for i in left_out_subject_first_n_sessions_indices], axis=0, dtype=np.float16)
        X_validation = np.array(data[left_out_subject_last_session_index])
        Y_validation = np.array(labels[left_out_subject_last_session_index])
        
        X_train = torch.from_numpy(X_pretrain).to(torch.float16)
        Y_train = torch.from_numpy(Y_pretrain).to(torch.float16)
        X_train_finetuning = torch.from_numpy(X_finetune).to(torch.float16)
        Y_train_finetuning = torch.from_numpy(Y_finetune).to(torch.float16)
        X_validation = torch.from_numpy(X_validation).to(torch.float16)
        Y_validation = torch.from_numpy(Y_validation).to(torch.float16)

        del X_finetune
        del Y_finetune
        
        if args.turn_on_unlabeled_domain_adaptation: # while in leave one session out
            if args.proportion_unlabeled_data>0:
                proportion_unlabeled_of_proportion_to_keep = args.proportion_unlabeled_data
                X_train_labeled_leftout_subject, X_train_unlabeled_leftout_subject, Y_train_labeled_leftout_subject, Y_train_unlabeled_leftout_subject = tts.train_test_split(
                    X_train_finetuning, Y_train_finetuning, train_size=1-proportion_unlabeled_of_proportion_to_keep, stratify=Y_finetune, random_state=args.seed, shuffle=False)
                X_train_finetuning = torch.tensor(X_train_labeled_leftout_subject)
                Y_train_finetuning = torch.tensor(Y_train_labeled_leftout_subject)
                X_train_unlabeled = torch.tensor(X_train_unlabeled_leftout_subject)
                Y_train_unlabeled = torch.tensor(Y_train_unlabeled_leftout_subject)
                
                print("Size of X_train_finetuning:     ", X_train_finetuning.shape)
                print("Size of Y_train_finetuning:     ", Y_train_finetuning.shape)
                print("Size of X_train_unlabeled:     ", X_train_unlabeled.shape)
                print("Size of Y_train_unlabeled:     ", Y_train_unlabeled.shape)
            else: 
                X_train_finetuning = torch.tensor(X_train_finetuning)
                Y_train_finetuning = torch.tensor(Y_train_finetuning)
                X_train_unlabeled = None
                Y_train_unlabeled = None
                
                print("Size of X_train_finetuning:     ", X_train_finetuning.shape)
                print("Size of Y_train_finetuning:     ", Y_train_finetuning.shape)
                
            if not args.pretrain_and_finetune:
                X_train = torch.concat((X_train, X_train_finetuning), axis=0)
                Y_train = torch.concat((Y_train, Y_train_finetuning), axis=0)
                
        else: 
            if not args.pretrain_and_finetune:
                X_train = torch.concat((X_train, X_train_finetuning), axis=0)
                Y_train = torch.concat((Y_train, Y_train_finetuning), axis=0)
            
        print("Size of X_train:     ", X_train.size())
        print("Size of Y_train:     ", Y_train.size())
        if not args.turn_on_unlabeled_domain_adaptation:
            print("Size of X_train_finetuning:  ", X_train_finetuning.size())
            print("Size of Y_train_finetuning:  ", Y_train_finetuning.size())
        print("Size of X_validation:", X_validation.size())
        print("Size of Y_validation:", Y_validation.size())
        
        del data
        del emg
        del labels
        
    elif args.leave_one_subject_out: # Running LOSO
        if args.reduce_training_data_size:
            reduced_size_per_subject = args.reduced_training_data_size // (utils.num_subjects - 1)

        X_validation = np.array(data[leaveOut-1])
        Y_validation = np.array(labels[leaveOut-1])

        X_train_list = []
        Y_train_list = []
        
        for i in range(len(data)):
            if i == leaveOut-1:
                continue
            current_data = np.array(data[i])
            current_labels = np.array(labels[i])

            if args.reduce_training_data_size:
                proportion_to_keep = reduced_size_per_subject / current_data.shape[0]
                current_data, _, current_labels, _ = model_selection.train_test_split(current_data, current_labels, 
                                                                                        train_size=proportion_to_keep, stratify=current_labels, 
                                                                                        random_state=args.seed, shuffle=True)

            X_train_list.append(current_data)
            Y_train_list.append(current_labels)
            
        X_train = torch.from_numpy(np.concatenate(X_train_list, axis=0)).to(torch.float16)
        Y_train = torch.from_numpy(np.concatenate(Y_train_list, axis=0)).to(torch.float16)
        X_validation = torch.from_numpy(X_validation).to(torch.float16)
        Y_validation = torch.from_numpy(Y_validation).to(torch.float16)

        if args.transfer_learning: # while in leave one subject out
            proportion_to_keep = args.proportion_transfer_learning
            proportion_unlabeled_of_proportion_to_keep = args.proportion_unlabeled_data
            
            if args.cross_validation_for_time_series:
                X_train_partial_leftout_subject, X_validation_partial_leftout_subject, Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject = tts.train_test_split(
                    X_validation, Y_validation, train_size=proportion_to_keep, stratify=Y_validation, random_state=args.seed, shuffle=False)
            else:
                # Split the validation data into train and validation subsets
                X_train_partial_leftout_subject, X_validation_partial_leftout_subject, Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject = tts.train_test_split(
                    X_validation, Y_validation, train_size=proportion_to_keep, stratify=Y_validation, random_state=args.seed, shuffle=True)
                
            if args.turn_on_unlabeled_domain_adaptation and proportion_unlabeled_of_proportion_to_keep>0:
                if args.cross_validation_for_time_series:
                    X_train_labeled_partial_leftout_subject, X_train_unlabeled_partial_leftout_subject, \
                    Y_train_labeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject = tts.train_test_split(
                        X_train_partial_leftout_subject, Y_train_partial_leftout_subject, train_size=1-proportion_unlabeled_of_proportion_to_keep, stratify=Y_train_partial_leftout_subject, random_state=args.seed, shuffle=False)
                else:
                    X_train_labeled_partial_leftout_subject, X_train_unlabeled_partial_leftout_subject, \
                    Y_train_labeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject = tts.train_test_split(
                        X_train_partial_leftout_subject, Y_train_partial_leftout_subject, train_size=1-proportion_unlabeled_of_proportion_to_keep, stratify=Y_train_partial_leftout_subject, random_state=args.seed, shuffle=True)
                    
            print("Size of X_train_partial_leftout_subject:     ", X_train_partial_leftout_subject.shape) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
            print("Size of Y_train_partial_leftout_subject:     ", Y_train_partial_leftout_subject.shape) # (SAMPLE, GESTURE)

            if not args.turn_on_unlabeled_domain_adaptation:
                # Append the partial validation data to the training data
                if not args.pretrain_and_finetune:
                    X_train = np.concatenate((X_train, X_train_partial_leftout_subject), axis=0)
                    Y_train = np.concatenate((Y_train, Y_train_partial_leftout_subject), axis=0)
                else:
                    X_train_finetuning = torch.tensor(X_train_partial_leftout_subject)
                    Y_train_finetuning = torch.tensor(Y_train_partial_leftout_subject)
            else: # unlabeled domain adaptation
                if proportion_unlabeled_of_proportion_to_keep>0:
                    if not args.pretrain_and_finetune:
                        X_train = torch.tensor(np.concatenate((X_train, X_train_labeled_partial_leftout_subject), axis=0))
                        Y_train = torch.tensor(np.concatenate((Y_train, Y_train_labeled_partial_leftout_subject), axis=0))
                        X_train_unlabeled = torch.tensor(X_train_unlabeled_partial_leftout_subject)
                        Y_train_unlabeled = torch.tensor(Y_train_unlabeled_partial_leftout_subject)
                    else:
                        X_train_finetuning = torch.tensor(X_train_labeled_partial_leftout_subject)
                        Y_train_finetuning = torch.tensor(Y_train_labeled_partial_leftout_subject)
                        X_train_unlabeled = torch.tensor(X_train_unlabeled_partial_leftout_subject)
                        Y_train_unlabeled = torch.tensor(Y_train_unlabeled_partial_leftout_subject)
                        
                else:
                    if not args.pretrain_and_finetune:
                        X_train = torch.tensor(np.concatenate((X_train, X_train_partial_leftout_subject), axis=0))
                        Y_train = torch.tensor(np.concatenate((Y_train, Y_train_partial_leftout_subject), axis=0))
                    else: 
                        X_train_finetuning = torch.tensor(X_train_partial_leftout_subject)
                        Y_train_finetuning = torch.tensor(Y_train_partial_leftout_subject)

            # Update the validation data
            X_train = torch.tensor(X_train).to(torch.float16)
            Y_train = torch.tensor(Y_train).to(torch.float16)
            X_validation = torch.tensor(X_validation_partial_leftout_subject).to(torch.float16)
            Y_validation = torch.tensor(Y_validation_partial_leftout_subject).to(torch.float16)
            
            del X_train_partial_leftout_subject, X_validation_partial_leftout_subject, Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject

        print("Size of X_train:     ", X_train.shape) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
        print("Size of Y_train:     ", Y_train.shape) # (SAMPLE, GESTURE)
        print("Size of X_validation:", X_validation.shape) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
        print("Size of Y_validation:", Y_validation.shape) # (SAMPLE, GESTURE)
        
        if args.turn_on_unlabeled_domain_adaptation and proportion_unlabeled_of_proportion_to_keep>0:
            print("Size of X_train_unlabeled:     ", X_train_unlabeled.shape)
            print("Size of Y_train_unlabeled:     ", Y_train_unlabeled.shape)
            
        if args.pretrain_and_finetune:
            print("Size of X_train_finetuning:     ", X_train_finetuning.shape)
            print("Size of Y_train_finetuning:     ", Y_train_finetuning.shape)
            
        
            
    else: 
        ValueError("Please specify the type of test you want to run")

model_name = args.model

if args.model == "vit_tiny_patch2_32":
    pretrain_path = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth"
elif args.model == "resnet50":
    pretrain_path = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
else:
    pretrain_path = f"https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/{model_name}_mlp_im_1k_224.pth"

if args.turn_on_unlabeled_domain_adaptation:
    print("Number of total batches in training data:", X_train.shape[0] // args.batch_size)
    current_date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    assert (args.transfer_learning and args.cross_validation_for_time_series) or args.leave_one_session_out, \
        "Unlabeled Domain Adaptation requires transfer learning and cross validation for time series or leave one session out"
    
    semilearn_config_dict = {
        'algorithm': args.unlabeled_algorithm,
        'net': args.model,
        'use_pretrain': True,  
        'pretrain_path': pretrain_path,
        'seed': args.seed,

        # optimization configs
        'epoch': args.epochs,  # set to 100
        'num_train_iter': args.epochs * (X_train.shape[0] // args.batch_size),  # set to 102400
        'num_eval_iter': X_train.shape[0] // args.batch_size,   # set to 1024
        'num_log_iter': X_train.shape[0] // args.batch_size,    # set to 256
        'optim': 'AdamW',   # AdamW optimizer
        'lr': args.learning_rate,  # Learning rate
        'layer_decay': 0.5,  # Layer-wise decay learning rate  
        'momentum': 0.9,  # Momentum
        'weight_decay': 0.0005,  # Weight decay
        'amp': True,  # Automatic mixed precision
        'train_sampler': 'RandomSampler',  # Random sampler
        'rank': 0,  # Rank
        'batch_size': args.batch_size,  # Batch size
        'eval_batch_size': args.batch_size, # Evaluation batch size
        'use_wandb': True,
        'ema_m': 0.999,
        'save_dir': './saved_models/unlabeled_domain_adaptation/',
        'save_name': f'{args.unlabeled_algorithm}_{args.model}_{args.dataset}_seed_{args.seed}_leave_{leaveOut}_unlabeled_domain_adaptation_{current_date_and_time}',
        'resume': True,
        'overwrite': True,
        'load_path': f'./saved_models/unlabeled_domain_adaptation/{args.unlabeled_algorithm}_{args.model}_{args.dataset}_seed_{args.seed}_leave_{leaveOut}_unlabeled_domain_adaptation_{current_date_and_time}/latest_model.pth',
        'scheduler': None,

        # dataset configs
        'dataset': 'none',
        'num_labels': X_train.shape[0],
        'num_classes': utils.numGestures,
        'input_size': 224,
        'data_dir': './data',

        # algorithm specific configs
        'hard_label': True,
        # 'uratio': 0.00232,
        'ulb_loss_ratio': 1.0,

        # device configs
        'gpu': 0,
        'world_size': 1,
        'distributed': False,
    }
    
    semilearn_config = get_config(semilearn_config_dict)
    semilearn_algorithm = get_algorithm(semilearn_config, get_net_builder(semilearn_config.net, from_name=False), tb_log=None, logger=None)
    semilearn_algorithm.model = send_model_cuda(semilearn_config, semilearn_algorithm.model)
    semilearn_algorithm.ema_model = send_model_cuda(semilearn_config, semilearn_algorithm.ema_model, clip_batch=False)
    
    class ToNumpy:
        """Custom transformation to convert PIL Images or Tensors to NumPy arrays."""
        def __call__(self, pic):
            if isinstance(pic, Image.Image):
                return np.array(pic)
            elif isinstance(pic, torch.Tensor):
                # Make sure the tensor is in CPU and convert it
                return np.float32(pic.cpu().detach().numpy())
            else:
                raise TypeError("Unsupported image type")

    
    semilearn_transform = transforms.Compose([transforms.Resize((224,224)), ToNumpy()])
    
    labeled_dataset = BasicDataset(semilearn_config, X_train, torch.argmax(Y_train, dim=1), semilearn_config.num_classes, semilearn_transform, is_ulb=False)
    if proportion_unlabeled_of_proportion_to_keep>0:
        unlabeled_dataset = BasicDataset(semilearn_config, X_train_unlabeled, torch.argmax(Y_train_unlabeled, dim=1), semilearn_config.num_classes, semilearn_transform, 
                                        is_ulb=True, strong_transform=semilearn_transform)
    if args.pretrain_and_finetune:
        finetune_dataset = BasicDataset(semilearn_config, X_train_finetuning, torch.argmax(Y_train_finetuning, dim=1), semilearn_config.num_classes, semilearn_transform, is_ulb=False)
    validation_dataset = BasicDataset(semilearn_config, X_validation, torch.argmax(Y_validation, dim=1), semilearn_config.num_classes, semilearn_transform, is_ulb=False)

    train_labeled_loader = get_data_loader(semilearn_config, labeled_dataset, semilearn_config.batch_size, num_workers=32)
    if proportion_unlabeled_of_proportion_to_keep>0:
        train_unlabeled_loader = get_data_loader(semilearn_config, unlabeled_dataset, semilearn_config.batch_size, num_workers=32)
    if args.pretrain_and_finetune:
        train_finetuning_loader = get_data_loader(semilearn_config, finetune_dataset, semilearn_config.batch_size, num_workers=32)
    validation_loader = get_data_loader(semilearn_config, validation_dataset, semilearn_config.eval_batch_size, num_workers=32)

else:
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
        model.add_module('fc3', nn.Linear(512, numGestures))
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
        n_outputs = numGestures

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
        model = timm.create_model(model_name, pretrained=True, num_classes=numGestures)
        # # Load the Vision Transformer model
        # model_name = 'vit_base_patch16_224'  # This is just one example, many variations exist
        # model = timm.create_model(model_name, pretrained=True, num_classes=utils.numGestures)

if not args.turn_on_unlabeled_domain_adaptation:
    num = 0
    for name, param in model.named_parameters():
        num += 1
        if (num > 0):
        #if (num > 72): # for -3
        #if (num > 33): # for -4
            param.requires_grad = True
        else:
            param.requires_grad = False

    batch_size = args.batch_size

    train_loader = DataLoader(list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=utils.seed_worker, pin_memory=True)
    val_loader = DataLoader(list(zip(X_validation, Y_validation)), batch_size=batch_size, num_workers=4, worker_init_fn=utils.seed_worker, pin_memory=True)
    if (args.held_out_test):
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
if (exercises):
    wandb_runname += '_exercises-' + ''.join(character for character in str(args.exercises) if character.isalnum())
if args.dataset == "OzdemirEMG":
    if args.full_dataset_ozdemir:
        wandb_runname += '_full-dataset'
    else:
        wandb_runname += '_partial-dataset'
if args.dataset == "ninapro-db2" or args.dataset == "ninapro-db5":
    if args.partial_dataset_ninapro:
        wandb_runname += '_partial-dataset'
if args.turn_on_spectrogram:
    wandb_runname += '_spectrogram'
if args.turn_on_cwt:
    wandb_runname += '_cwt'
if args.turn_on_hht:
    wandb_runname += '_hht'
if args.reduce_training_data_size:
    wandb_runname += '_reduced-training-data-size-' + str(args.reduced_training_data_size)
if args.leave_n_subjects_out_randomly != 0:
    wandb_runname += '_leave_n_subjects_out_randomly-'+str(args.leave_n_subjects_out_randomly)
if args.turn_off_scaler_normalization:
    wandb_runname += '_no-scaler-normalization'
if args.target_normalize:
    wandb_runname += '_target-normalize'
if args.load_few_images:
    wandb_runname += '_load-few-images'
if args.transfer_learning:
    wandb_runname += '_transfer-learning'
    wandb_runname += '-proportion-' + str(args.proportion_transfer_learning)
if args.cross_validation_for_time_series:   
    wandb_runname += '_cross-validation-for-time-series'
if args.reduce_data_for_transfer_learning != 1:
    wandb_runname += '_reduce-data-for-transfer-learning-' + str(args.reduce_data_for_transfer_learning)
if args.leave_one_session_out:
    wandb_runname += '_leave-one-session-out'
if args.leave_one_subject_out:
    wandb_runname += '_leave-one-subject-out'
if args.one_subject_for_training_set_for_session_test:
    wandb_runname += '_one-subject-for-training-set-for-session-test'
if args.held_out_test:
    wandb_runname += '_held-out-test'
if args.pretrain_and_finetune:
    wandb_runname += '_pretrain-and-finetune'
if args.turn_on_unlabeled_domain_adaptation:
    wandb_runname += '_unlabeled-adapt'
    wandb_runname += '-algo-' + args.unlabeled_algorithm
    wandb_runname += '-proportion-unlabeled-' + str(args.proportion_unlabeled_data)

if (args.held_out_test):
    if args.turn_on_kfold:
        project_name += '_k-fold-'+str(args.kfold)
    else:
        project_name += '_heldout'
elif args.leave_one_subject_out:
    project_name += '_LOSO'
elif args.leave_one_session_out:
    project_name += '_leave-one-session-out'
    

project_name += args.project_name_suffix

run = wandb.init(name=wandb_runname, project=project_name, entity='jehanyang')
wandb.config.lr = args.learning_rate
if args.leave_n_subjects_out_randomly != 0:
    wandb.config.left_out_subjects = leaveOutIndices

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
print("Device:", device)
if not args.turn_on_unlabeled_domain_adaptation:
    model.to(device)

    wandb.watch(model)

testrun_foldername = f'test/{project_name}/{wandb_runname}/{formatted_datetime}/'
# Make folder if it doesn't exist
if not os.path.exists(testrun_foldername):
    os.makedirs(testrun_foldername)
model_filename = f'{testrun_foldername}model_{formatted_datetime}.pth'

if (exercises):
    gesture_labels = utils.gesture_labels['Rest']
    for exercise_set in args.exercises:
        gesture_labels = gesture_labels + utils.gesture_labels[exercise_set]
else:
    gesture_labels = utils.gesture_labels
    
# if X_train, validation, test or Y_train validation, test are numpy arrays, convert them to tensors
X_train = torch.from_numpy(X_train).to(torch.float16) if isinstance(X_train, np.ndarray) else X_train
Y_train = torch.from_numpy(Y_train).to(torch.float16) if isinstance(Y_train, np.ndarray) else Y_train
X_validation = torch.from_numpy(X_validation).to(torch.float16) if isinstance(X_validation, np.ndarray) else X_validation
Y_validation = torch.from_numpy(Y_validation).to(torch.float16) if isinstance(Y_validation, np.ndarray) else Y_validation
if args.held_out_test:
    X_test = torch.from_numpy(X_test).to(torch.float16) if isinstance(X_test, np.ndarray) else X_test
    Y_test = torch.from_numpy(Y_test).to(torch.float16) if isinstance(Y_test, np.ndarray) else Y_test

# if args.held_out_test:
#     # Plot and log images
#     utils.plot_average_images(X_test, np.argmax(Y_test.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'test')
#     utils.plot_first_fifteen_images(X_test, np.argmax(Y_test.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'test')

# utils.plot_average_images(X_validation, np.argmax(Y_validation.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')
# utils.plot_first_fifteen_images(X_validation, np.argmax(Y_validation.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')

# utils.plot_average_images(X_train, np.argmax(Y_train.cpu().detach().numpy(), axis=1), utils.gesture_labels, testrun_foldername, args, formatted_datetime, 'train')
# utils.plot_first_fifteen_images(X_train, np.argmax(Y_train.cpu().detach().numpy(), axis=1), utils.gesture_labels, testrun_foldername, args, formatted_datetime, 'train')

if args.turn_on_unlabeled_domain_adaptation:
    semilearn_algorithm.loader_dict = {}
    semilearn_algorithm.loader_dict['train_lb'] = train_labeled_loader
    if proportion_unlabeled_of_proportion_to_keep>0:
        semilearn_algorithm.loader_dict['train_ulb'] = train_unlabeled_loader
    semilearn_algorithm.loader_dict['eval'] = validation_loader
    semilearn_algorithm.scheduler = None
    
    semilearn_algorithm.train()
    
    if args.pretrain_and_finetune:
        run = wandb.init(name=wandb_runname, project=project_name, entity='jehanyang')
        wandb.config.lr = args.learning_rate
        
        semilearn_config_dict['num_train_iter'] = semilearn_config_dict['num_train_iter'] + args.finetuning_epochs * (X_train_finetuning.shape[0] // args.batch_size)
        semilearn_config_dict['num_eval_iter'] = X_train_finetuning.shape[0] // args.batch_size
        semilearn_config_dict['num_log_iter'] = X_train_finetuning.shape[0] // args.batch_size
        semilearn_config_dict['algorithm'] = args.unlabeled_algorithm
        
        semilearn_config = get_config(semilearn_config_dict)
        semilearn_algorithm = get_algorithm(semilearn_config, get_net_builder(semilearn_config.net, from_name=False), tb_log=None, logger=None)
        semilearn_algorithm.epochs = args.epochs + args.finetuning_epochs # train for the same number of epochs as the previous training
        semilearn_algorithm.model = send_model_cuda(semilearn_config, semilearn_algorithm.model)
        semilearn_algorithm.load_model(semilearn_config.load_path)
        semilearn_algorithm.ema_model = send_model_cuda(semilearn_config, semilearn_algorithm.ema_model, clip_batch=False)
        semilearn_algorithm.loader_dict = {}
        semilearn_algorithm.loader_dict['train_lb'] = train_finetuning_loader
        semilearn_algorithm.scheduler = None
        
        if proportion_unlabeled_of_proportion_to_keep>0:
            semilearn_algorithm.loader_dict['train_ulb'] = train_unlabeled_loader
            
        semilearn_algorithm.loader_dict['eval'] = validation_loader
        semilearn_algorithm.train()
    # trainer = Trainer(semilearn_config, semilearn_algorithm)
    # trainer.fit(train_labeled_loader, train_unlabeled_loader, validation_loader)
    # trainer.evaluate(validation_loader)
    
else: 
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

    if args.pretrain_and_finetune:
        num_epochs = args.finetuning_epochs
        # train more on fine tuning dataset
        finetune_loader = DataLoader(list(zip(X_train_finetuning, Y_train_finetuning)), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=utils.seed_worker, pin_memory=True)
        for epoch in tqdm(range(num_epochs), desc="Finetuning Epoch"):
            model.train()
            train_acc = 0.0
            train_loss = 0.0
            with tqdm(finetune_loader, desc=f"Finetuning Epoch {epoch+1}/{num_epochs}", leave=False) as t:
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

            train_loss /= len(finetune_loader)
            train_acc /= len(finetune_loader)
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            print(f"Finetuning Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")
            wandb.log({
                "Finetuning Epoch": epoch,
                "Train Loss": train_loss,
                "Train Acc": train_acc,
                "Valid Loss": val_loss,
                "Valid Acc": val_acc, 
                "Learning Rate": optimizer.param_groups[0]['lr']})
            
    # Testing
    if (args.held_out_test):
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
        utils.plot_confusion_matrix(true, pred, gesture_labels, testrun_foldername, args, formatted_datetime, 'test')

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

    utils.plot_confusion_matrix(np.argmax(Y_validation.cpu().detach().numpy(), axis=1), np.array(validation_predictions), gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')   

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

    utils.plot_confusion_matrix(np.argmax(Y_train.cpu().detach().numpy(), axis=1), np.array(train_predictions), gesture_labels, testrun_foldername, args, formatted_datetime, 'train')
        
    run.finish()
