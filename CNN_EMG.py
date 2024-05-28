import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from sklearn import preprocessing, model_selection
import wandb
import multiprocessing
from tqdm import tqdm
import argparse
import random 
import utils_OzdemirEMG as utils
from sklearn.model_selection import StratifiedKFold
import os
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import timm
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import zarr
import cross_validation_utilities.train_test_split as tts # custom train test split to split stratified without shuffling
import gc
import datetime
from PIL import Image
from torch.utils.data import Dataset
import VisualTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer, split_ssl_data, BasicDataset
from semilearn.core.utils import send_model_cuda
import torchmetrics
import ml_metrics_utils as ml_utils


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
parser.add_argument('--target_normalize', type=float, help='use a poportion of leftout data for normalization. Set to 0 by default.', default=0)
# Test with transfer learning by using some data from the validation dataset
parser.add_argument('--transfer_learning', type=utils.str2bool, help='use some data from the validation dataset for transfer learning. Set to False by default.', default=False)
# Add argument for cross validation for time series
parser.add_argument('--cross_validation_for_time_series', type=utils.str2bool, help='whether or not to use cross validation for time series. Set to False by default.', default=False)
# Add argument for proportion of left-out-subject data to use for transfer learning
parser.add_argument('--proportion_transfer_learning_from_leftout_subject', type=float, help='proportion of left-out-subject data to use for transfer learning. Set to 0.25 by default.', default=0.25)
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
parser.add_argument('--unlabeled_algorithm', type=str, help='algorithm to use for unlabeled domain adaptation. Set to "fixmatch" by default.', default="fixmatch")
# Add argument to specify proportion from left-out-subject to keep as unlabeled data
parser.add_argument('--proportion_unlabeled_data_from_leftout_subject', type=float, help='proportion of data from left-out-subject to keep as unlabeled data. Set to 0.75 by default.', default=0.75)
# Add argument to specify batch size
parser.add_argument('--batch_size', type=int, help='batch size. Set to 64 by default.', default=64)
# Add argument for whether to use unlabeled data for subjects used for training as well
parser.add_argument('--proportion_unlabeled_data_from_training_subjects', type=float, help='proportion of data from training subjects to use as unlabeled data. Set to 0.0 by default.', default=0.0)
# Add argument for cutting down amount of total data for training subjects
parser.add_argument('--proportion_data_from_training_subjects', type=float, help='proportion of data from training subjects to use. Set to 1.0 by default.', default=1.0)
# Add argument for loading unlabeled data from jehan dataset
parser.add_argument('--load_unlabeled_data_jehan', type=utils.str2bool, help='whether or not to load unlabeled data from Jehan dataset. Set to False by default.', default=False)

# Parse the arguments
args = parser.parse_args()

exercises = False

if args.model == "MLP" or args.model == "SVC" or args.model == "RF":
    print("Warning: not using pytorch, many arguments will be ignored")
    if args.turn_on_unlabeled_domain_adaptation:
        NotImplementedError("Cannot use unlabeled domain adaptation with MLP, SVC, or RF")
    if args.pretrain_and_finetune:
        NotImplementedError("Cannot use pretrain and finetune with MLP, SVC, or RF")

if (args.dataset.lower() == "uciemg" or args.dataset.lower() == "uci"):
    import utils_UCI as utils
    print(f"The dataset being tested is uciEMG")
    project_name = 'emg_benchmarking_uci'
    args.dataset = "uciemg"

elif (args.dataset.lower() == "ninapro-db2" or args.dataset.lower() == "ninapro_db2"):
    import utils_NinaproDB2 as utils
    print(f"The dataset being tested is ninapro-db2")
    project_name = 'emg_benchmarking_ninapro-db2'
    exercises = True
    if args.leave_one_session_out:
        ValueError("leave-one-session-out not implemented for ninapro-db2; only one session exists")
    args.dataset = 'ninapro-db2'

elif (args.dataset.lower() == "ninapro-db5" or args.dataset.lower() == "ninapro_db5"):
    import utils_NinaproDB5 as utils
    print(f"The dataset being tested is ninapro-db5")
    project_name = 'emg_benchmarking_ninapro-db5'
    exercises = True
    if args.leave_one_session_out:
        ValueError("leave-one-session-out not implemented for ninapro-db5; only one session exists")
    args.dataset = 'ninapro-db5'

elif (args.dataset.lower() == "ninapro-db3" or args.dataset.lower() == "ninapro_db3"):
    import utils_NinaproDB3 as utils
    assert args.exercises == [1], "Exercises C and D are not implemented due to missing data."
    print(f"The dataset being tested is ninapro-db3")
    project_name = 'emg_benchmarking_ninapro-db3'
    exercises = True
    if args.leave_one_session_out:
        ValueError("leave-one-session-out not implemented for ninapro-db3; only one session exists")

elif (args.dataset.lower() == "m-dataset" or args.dataset.lower() == "m_dataset"):
    import utils_M_dataset as utils
    print(f"The dataset being tested is M_dataset")
    project_name = 'emg_benchmarking_M_dataset'
    if args.leave_one_session_out:
        ValueError("leave-one-session-out not implemented for M_dataset; only one session exists")
    args.dataset = 'm-dataset'

elif (args.dataset.lower() == "hyser"):
    import utils_Hyser as utils
    print(f"The dataset being tested is hyser")
    project_name = 'emg_benchmarking_hyser'
    args.dataset = 'hyser'

elif (args.dataset.lower() == "capgmyo"):
    import utils_CapgMyo as utils
    print(f"The dataset being tested is CapgMyo")
    project_name = 'emg_benchmarking_capgmyo'
    if args.leave_one_session_out:
        utils.num_subjects = 10
    args.dataset = 'capgmyo'

elif (args.dataset.lower() == "jehan"):
    import utils_JehanData as utils
    print(f"The dataset being tested is JehanDataset")
    project_name = 'emg_benchmarking_jehandataset'
    if args.leave_one_session_out:
        ValueError("leave-one-session-out not implemented for JehanDataset; only one session exists")
    args.dataset = 'jehan'

elif (args.dataset.lower() == "sci"):
    import utils_SCI as utils
    print(f"The dataset being tested is SCI")
    project_name = 'emg_benchmarking_sci'
    args.dataset = 'sci'
    

elif (args.dataset.lower() == "ozdemiremg" or args.dataset.lower() == "ozdemir_emg"):
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
    args.dataset = 'ozdemiremg'
    
else: 
    raise ValueError("Dataset not recognized. Please choose from 'uciemg', 'ninapro-db2', 'ninapro-db5', 'm-dataset', 'hyser'," +
                    "'capgmyo', 'jehan', 'sci', or 'ozdemiremg'")

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
print(f"The value of --proportion_transfer_learning_from_leftout_subject is {args.proportion_transfer_learning_from_leftout_subject}")
print(f"The value of --reduce_data_for_transfer_learning is {args.reduce_data_for_transfer_learning}")
print(f"The value of --leave_one_session_out is {args.leave_one_session_out}")
print(f"The value of --held_out_test is {args.held_out_test}")
print(f"The value of --one_subject_for_training_set_for_session_test is {args.one_subject_for_training_set_for_session_test}")
print(f"The value of --pretrain_and_finetune is {args.pretrain_and_finetune}")
print(f"The value of --finetuning_epochs is {args.finetuning_epochs}")

print(f"The value of --turn_on_unlabeled_domain_adaptation is {args.turn_on_unlabeled_domain_adaptation}")
print(f"The value of --unlabeled_algorithm is {args.unlabeled_algorithm}")
print(f"The value of --proportion_unlabeled_data_from_leftout_subject is {args.proportion_unlabeled_data_from_leftout_subject}")

print(f"The value of --batch_size is {args.batch_size}")

print(f"The value of --proportion_unlabeled_data_from_training_subjects is {args.proportion_unlabeled_data_from_training_subjects}")
print(f"The value of --proportion_data_from_training_subjects is {args.proportion_data_from_training_subjects}")
print(f"The value of --load_unlabeled_data_jehan is {args.load_unlabeled_data_jehan}")

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
        elif args.dataset == "ninapro-db3":
            args.exercises = [1]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
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
    if (args.target_normalize > 0):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
            if args.leave_one_session_out:
                NotImplementedError("leave-one-session-out not implemented with target_normalize yet")

            mins, maxes = utils.getExtrema(args.leftout_subject, args.target_normalize)
            
            emg_async = pool.map_async(utils.getEMG, [(i+1, mins, maxes, args.leftout_subject + 1) for i in range(utils.num_subjects)])
            emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
            
            labels_async = pool.map_async(utils.getLabels, [(i+1) for i in range(utils.num_subjects)])
            labels = labels_async.get()
    else: # Not target_normalize
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
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
                if args.dataset == "capgmyo":
                    dataset_identifiers = 20 # 20 identifiers for capgmyo dbb (10 subjects, 2 sessions each)
                else:
                    dataset_identifiers = utils.num_subjects
                    
                emg_async = pool.map_async(utils.getEMG, [(i+1) for i in range(dataset_identifiers)])
                emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
                
                labels_async = pool.map_async(utils.getLabels, [(i+1) for i in range(dataset_identifiers)])
                labels = labels_async.get()

    print("subject 1 mean", torch.mean(emg[0]))
    numGestures = utils.numGestures

if args.dataset == "capgmyo" and not args.leave_one_session_out:
    # Condense lists of 20 into list of 10
    emg = [torch.cat((emg[i], emg[i+1]), dim=0) for i in range(0, len(emg), 2)]
    labels = [torch.cat((labels[i], labels[i+1]), dim=0) for i in range(0, len(labels), 2)]
    
if args.load_unlabeled_data_jehan:
    assert args.dataset == "jehan", "Can only load unlabeled online data from Jehan dataset"
    print("Loading unlabeled online data from Jehan dataset")
    unlabeled_online_data = utils.getOnlineUnlabeledData(args.leftout_subject)

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
if args.leave_n_subjects_out_randomly != 0 and (not args.turn_off_scaler_normalization and not (args.target_normalize > 0)): # will have to run and test this again, or just remove
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
    if (args.held_out_test): # should be deprecated and deleted
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

    elif (not args.turn_off_scaler_normalization and not (args.target_normalize > 0)): # Running LOSO standardization
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

# add tqdm to show progress bar
print("Width of EMG data: ", width)
print("Length of EMG data: ", length)

base_foldername_zarr = ""

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
else:
    base_foldername_zarr += 'raw/'

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
        if (args.target_normalize > 0):
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
        
if args.load_unlabeled_data_jehan:
    unlabeled_images = utils.getImages(unlabeled_online_data, scaler, length, width,
                                                turn_on_rms=args.turn_on_rms, rms_windows=args.rms_input_windowsize,
                                                turn_on_magnitude=args.turn_on_magnitude, global_min=global_low_value, global_max=global_high_value,
                                                turn_on_spectrogram=args.turn_on_spectrogram, turn_on_cwt=args.turn_on_cwt,
                                                turn_on_hht=args.turn_on_hht)
    unlabeled_images = np.array(unlabeled_images, dtype=np.float16)
    unlabeled_data = unlabeled_images
    del unlabeled_images, unlabeled_online_data

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
        total_number_of_sessions = 2 # all datasets used in our benchmark have at most 2 sessions but this can be changed using a variable from dataset-specific utils instead
        left_out_subject_last_session_index = (total_number_of_sessions - 1) * utils.num_subjects + leaveOut-1
        left_out_subject_first_n_sessions_indices = [i for i in range(total_number_of_sessions * utils.num_subjects) if i % utils.num_subjects == (leaveOut-1) and i != left_out_subject_last_session_index]
        print("left_out_subject_last_session_index:", left_out_subject_last_session_index)
        print("left_out_subject_first_n_sessions_indices:", left_out_subject_first_n_sessions_indices)

        X_pretrain = []
        Y_pretrain = []
        if args.proportion_unlabeled_data_from_training_subjects>0:
            X_pretrain_unlabeled_list = []
            Y_pretrain_unlabeled_list = []

        X_finetune = []
        Y_finetune = []
        if args.proportion_unlabeled_data_from_leftout_subject>0:
            X_finetune_unlabeled_list = []
            Y_finetune_unlabeled_list = []

        for i in range(total_number_of_sessions * utils.num_subjects):
            X_train_temp = data[i]
            Y_train_temp = labels[i]
            if i != left_out_subject_last_session_index and i not in left_out_subject_first_n_sessions_indices:
                if args.proportion_data_from_training_subjects<1.0:
                    X_train_temp, _, Y_train_temp, _ = tts.train_test_split(
                        X_train_temp, Y_train_temp, train_size=args.proportion_data_from_training_subjects, stratify=Y_train_temp, random_state=args.seed, shuffle=(not args.cross_validation_for_time_series))
                if args.proportion_unlabeled_data_from_training_subjects>0:
                    X_pretrain_labeled, X_pretrain_unlabeled, Y_pretrain_labeled, Y_pretrain_unlabeled = tts.train_test_split(
                        X_train_temp, Y_train_temp, train_size=1-args.proportion_unlabeled_data_from_training_subjects, stratify=labels[i], random_state=args.seed, shuffle=(not args.cross_validation_for_time_series))
                    X_pretrain.append(np.array(X_pretrain_labeled))
                    Y_pretrain.append(np.array(Y_pretrain_labeled))
                    X_pretrain_unlabeled_list.append(np.array(X_pretrain_unlabeled))
                    Y_pretrain_unlabeled_list.append(np.array(Y_pretrain_unlabeled))
                else:
                    X_pretrain.append(np.array(X_train_temp))
                    Y_pretrain.append(np.array(Y_train_temp))
            elif i in left_out_subject_first_n_sessions_indices:
                if args.proportion_unlabeled_data_from_leftout_subject>0:
                    X_finetune_labeled, X_finetune_unlabeled, Y_finetune_labeled, Y_finetune_unlabeled = tts.train_test_split(
                        X_train_temp, Y_train_temp, train_size=1-args.proportion_unlabeled_data_from_leftout_subject, stratify=labels[i], random_state=args.seed, shuffle=(not args.cross_validation_for_time_series))
                    X_finetune.append(np.array(X_finetune_labeled))
                    Y_finetune.append(np.array(Y_finetune_labeled))
                    X_finetune_unlabeled_list.append(np.array(X_finetune_unlabeled))
                    Y_finetune_unlabeled_list.append(np.array(Y_finetune_unlabeled))
                else:
                    X_finetune.append(np.array(X_train_temp))
                    Y_finetune.append(np.array(Y_train_temp))
        if args.load_unlabeled_data_jehan:
            X_finetune_unlabeled_list.append(unlabeled_data)
            Y_finetune_unlabeled_list.append(np.zeros(unlabeled_data.shape[0]))

        X_pretrain = np.concatenate(X_pretrain, axis=0, dtype=np.float16)
        Y_pretrain = np.concatenate(Y_pretrain, axis=0, dtype=np.float16)
        X_finetune = np.concatenate(X_finetune, axis=0, dtype=np.float16)
        Y_finetune = np.concatenate(Y_finetune, axis=0, dtype=np.float16)
        X_validation = np.array(data[left_out_subject_last_session_index])
        Y_validation = np.array(labels[left_out_subject_last_session_index])
        if args.proportion_unlabeled_data_from_training_subjects>0:
            X_pretrain_unlabeled = np.concatenate(X_pretrain_unlabeled_list, axis=0, dtype=np.float16)
            Y_pretrain_unlabeled = np.concatenate(Y_pretrain_unlabeled_list, axis=0, dtype=np.float16)
        if args.proportion_unlabeled_data_from_leftout_subject>0 or args.load_unlabeled_data_jehan:
            X_finetune_unlabeled = np.concatenate(X_finetune_unlabeled_list, axis=0, dtype=np.float16)
            Y_finetune_unlabeled = np.concatenate(Y_finetune_unlabeled_list, axis=0, dtype=np.float16)
        
        X_train = torch.from_numpy(X_pretrain).to(torch.float16)
        Y_train = torch.from_numpy(Y_pretrain).to(torch.float16)
        X_train_finetuning = torch.from_numpy(X_finetune).to(torch.float16)
        Y_train_finetuning = torch.from_numpy(Y_finetune).to(torch.float16)
        X_validation = torch.from_numpy(X_validation).to(torch.float16)
        Y_validation = torch.from_numpy(Y_validation).to(torch.float16)
        if args.proportion_unlabeled_data_from_training_subjects>0:
            X_train_unlabeled = torch.from_numpy(X_pretrain_unlabeled).to(torch.float16)
            Y_train_unlabeled = torch.from_numpy(Y_pretrain_unlabeled).to(torch.float16)
        if args.proportion_unlabeled_data_from_leftout_subject>0 or args.load_unlabeled_data_jehan:
            X_train_finetuning_unlabeled = torch.from_numpy(X_finetune_unlabeled).to(torch.float16)
            Y_train_finetuning_unlabeled = torch.from_numpy(Y_finetune_unlabeled).to(torch.float16)

        del X_finetune
        del Y_finetune
        
        if args.turn_on_unlabeled_domain_adaptation: # while in leave one session out
            proportion_to_keep_of_leftout_subject_for_training = args.proportion_transfer_learning_from_leftout_subject
            proportion_unlabeled_of_proportion_to_keep_of_leftout = args.proportion_unlabeled_data_from_leftout_subject
            proportion_unlabeled_of_training_subjects = args.proportion_unlabeled_data_from_training_subjects

            if args.proportion_unlabeled_data_from_training_subjects>0:
                # X_train_labeled, X_train_unlabeled, Y_train_labeled, Y_train_unlabeled = tts.train_test_split( # DELETE
                #     X_train, Y_train, train_size=1-proportion_unlabeled_of_training_subjects, stratify=Y_train, random_state=args.seed, shuffle=True)
                # X_train = torch.tensor(X_train_labeled)
                # Y_train = torch.tensor(Y_train_labeled)
                # X_train_unlabeled = torch.tensor(X_train_unlabeled)
                # Y_train_unlabeled = torch.tensor(Y_train_unlabeled)
                
                # print("Size of X_train:     ", X_train.size())
                # print("Size of Y_train:     ", Y_train.size())
                print("Size of X_train_unlabeled:     ", X_train_unlabeled.size())
                print("Size of Y_train_unlabeled:     ", Y_train_unlabeled.size())

            if args.proportion_unlabeled_data_from_leftout_subject>0:
                # proportion_unlabeled_of_proportion_to_keep = args.proportion_unlabeled_data_from_leftout_subject # DELETE
                # X_train_labeled_leftout_subject, X_train_unlabeled_leftout_subject, Y_train_labeled_leftout_subject, Y_train_unlabeled_leftout_subject = tts.train_test_split(
                #     X_train_finetuning, Y_train_finetuning, train_size=1-proportion_unlabeled_of_proportion_to_keep, stratify=Y_finetune, random_state=args.seed, shuffle=False)
                # X_train_finetuning = torch.tensor(X_train_labeled_leftout_subject)
                # Y_train_finetuning = torch.tensor(Y_train_labeled_leftout_subject)
                # X_train_finetuning_unlabeled = torch.tensor(X_train_unlabeled_leftout_subject)
                # Y_train_finetuning_unlabeled = torch.tensor(Y_train_unlabeled_leftout_subject)
                
                print("Size of X_train_finetuning:     ", X_train_finetuning.shape)
                print("Size of Y_train_finetuning:     ", Y_train_finetuning.shape)
                print("Size of X_train_finetuning_unlabeled:     ", X_train_finetuning_unlabeled.shape)
                print("Size of Y_train_finetuning_unlabeled:     ", Y_train_finetuning_unlabeled.shape)
            else: 
                X_train_finetuning = torch.tensor(X_train_finetuning)
                Y_train_finetuning = torch.tensor(Y_train_finetuning)
                X_train_finetuning_unlabeled = None
                Y_train_finetuning_unlabeled = None
                
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
        
    elif args.leave_one_subject_out: # Running LOSO rather than leave one session out
        if args.reduce_training_data_size:
            reduced_size_per_subject = args.reduced_training_data_size // (utils.num_subjects - 1)

        X_validation = np.array(data[leaveOut-1])
        Y_validation = np.array(labels[leaveOut-1])

        X_train_list = []
        Y_train_list = []

        if args.proportion_unlabeled_data_from_training_subjects>0:
            X_train_unlabeled_list = []
            Y_train_unlabeled_list = []
        
        for i in range(len(data)):
            if i == leaveOut-1:
                continue
            current_data = np.array(data[i])
            current_labels = np.array(labels[i])

            if args.reduce_training_data_size:
                proportion_to_keep = reduced_size_per_subject / current_data.shape[0]
                current_data, _, current_labels, _ = model_selection.train_test_split(current_data, current_labels, 
                                                                                        train_size=proportion_to_keep, stratify=current_labels, 
                                                                                        random_state=args.seed, shuffle=(not args.cross_validation_for_time_series))
                
            if args.proportion_data_from_training_subjects<1.0:
                current_data, _, current_labels, _ = tts.train_test_split(
                    current_data, current_labels, train_size=args.proportion_data_from_training_subjects, stratify=current_labels, random_state=args.seed, shuffle=(not args.cross_validation_for_time_series))
                
            if args.proportion_unlabeled_data_from_training_subjects>0:
                X_train_labeled, X_train_unlabeled, Y_train_labeled, Y_train_unlabeled = tts.train_test_split(
                    current_data, current_labels, train_size=1-args.proportion_unlabeled_data_from_training_subjects, stratify=current_labels, random_state=args.seed, shuffle=(not args.cross_validation_for_time_series))
                current_data = X_train_labeled
                current_labels = Y_train_labeled

                X_train_unlabeled_list.append(X_train_unlabeled)
                Y_train_unlabeled_list.append(Y_train_unlabeled)

            X_train_list.append(current_data)
            Y_train_list.append(current_labels)
            
        X_train = torch.from_numpy(np.concatenate(X_train_list, axis=0)).to(torch.float16)
        Y_train = torch.from_numpy(np.concatenate(Y_train_list, axis=0)).to(torch.float16)
        if args.proportion_unlabeled_data_from_training_subjects>0:
            X_train_unlabeled = torch.from_numpy(np.concatenate(X_train_unlabeled_list, axis=0)).to(torch.float16)
            Y_train_unlabeled = torch.from_numpy(np.concatenate(Y_train_unlabeled_list, axis=0)).to(torch.float16)
        X_validation = torch.from_numpy(X_validation).to(torch.float16)
        Y_validation = torch.from_numpy(Y_validation).to(torch.float16)

        if args.transfer_learning: # while in leave one subject out
            proportion_to_keep_of_leftout_subject_for_training = args.proportion_transfer_learning_from_leftout_subject
            proportion_unlabeled_of_proportion_to_keep_of_leftout = args.proportion_unlabeled_data_from_leftout_subject
            proportion_unlabeled_of_training_subjects = args.proportion_unlabeled_data_from_training_subjects
            
            if proportion_to_keep_of_leftout_subject_for_training>0.0:
                if args.cross_validation_for_time_series:
                    X_train_partial_leftout_subject, X_validation_partial_leftout_subject, Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject = tts.train_test_split(
                        X_validation, Y_validation, train_size=proportion_to_keep_of_leftout_subject_for_training, stratify=Y_validation, random_state=args.seed, shuffle=False)
                else:
                    # Split the validation data into train and validation subsets
                    X_train_partial_leftout_subject, X_validation_partial_leftout_subject, Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject = tts.train_test_split(
                        X_validation, Y_validation, train_size=proportion_to_keep_of_leftout_subject_for_training, stratify=Y_validation, random_state=args.seed, shuffle=True)
            else:
                X_validation_partial_leftout_subject = X_validation
                Y_validation_partial_leftout_subject = Y_validation
                X_train_partial_leftout_subject = torch.tensor([])
                Y_train_partial_leftout_subject = torch.tensor([])
                
            if args.turn_on_unlabeled_domain_adaptation and proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
                if args.cross_validation_for_time_series:
                    X_train_labeled_partial_leftout_subject, X_train_unlabeled_partial_leftout_subject, \
                    Y_train_labeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject = tts.train_test_split(
                        X_train_partial_leftout_subject, Y_train_partial_leftout_subject, train_size=1-proportion_unlabeled_of_proportion_to_keep_of_leftout, stratify=Y_train_partial_leftout_subject, random_state=args.seed, shuffle=False)
                else:
                    X_train_labeled_partial_leftout_subject, X_train_unlabeled_partial_leftout_subject, \
                    Y_train_labeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject = tts.train_test_split(
                        X_train_partial_leftout_subject, Y_train_partial_leftout_subject, train_size=1-proportion_unlabeled_of_proportion_to_keep_of_leftout, stratify=Y_train_partial_leftout_subject, random_state=args.seed, shuffle=True)
            
            if args.load_unlabeled_data_jehan:
                if proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
                    X_train_unlabeled_partial_leftout_subject = np.concatenate([X_train_unlabeled_partial_leftout_subject, unlabeled_data], axis=0)
                    Y_train_unlabeled_partial_leftout_subject = np.concatenate([Y_train_unlabeled_partial_leftout_subject, np.zeros((unlabeled_data.shape[0], utils.numGestures))], axis=0)
                else:
                    X_train_unlabeled_partial_leftout_subject = unlabeled_data
                    Y_train_unlabeled_partial_leftout_subject = np.zeros((unlabeled_data.shape[0], utils.numGestures))

            print("Size of X_train_partial_leftout_subject:     ", X_train_partial_leftout_subject.shape) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
            print("Size of Y_train_partial_leftout_subject:     ", Y_train_partial_leftout_subject.shape) # (SAMPLE, GESTURE)

            if not args.turn_on_unlabeled_domain_adaptation:
                # Append the partial validation data to the training data
                if proportion_to_keep_of_leftout_subject_for_training>0:
                    if not args.pretrain_and_finetune:
                        X_train = np.concatenate((X_train, X_train_partial_leftout_subject), axis=0)
                        Y_train = np.concatenate((Y_train, Y_train_partial_leftout_subject), axis=0)
                    else:
                        X_train_finetuning = torch.tensor(X_train_partial_leftout_subject)
                        Y_train_finetuning = torch.tensor(Y_train_partial_leftout_subject)

            else: # unlabeled domain adaptation
                if proportion_unlabeled_of_training_subjects>0:
                    X_train = torch.tensor(X_train)
                    Y_train = torch.tensor(Y_train)
                    X_train_unlabeled = torch.tensor(X_train_unlabeled)
                    Y_train_unlabeled = torch.tensor(Y_train_unlabeled)

                if proportion_unlabeled_of_proportion_to_keep_of_leftout>0 or args.load_unlabeled_data_jehan:
                    if proportion_unlabeled_of_proportion_to_keep_of_leftout==0:
                        X_train_labeled_partial_leftout_subject = X_train_partial_leftout_subject
                        Y_train_labeled_partial_leftout_subject = Y_train_partial_leftout_subject
                    if not args.pretrain_and_finetune:
                        X_train = torch.tensor(np.concatenate((X_train, X_train_labeled_partial_leftout_subject), axis=0))
                        Y_train = torch.tensor(np.concatenate((Y_train, Y_train_labeled_partial_leftout_subject), axis=0))
                        X_train_unlabeled = torch.tensor(np.concatenate((X_train_unlabeled, X_train_unlabeled_partial_leftout_subject), axis=0))
                        Y_train_unlabeled = torch.tensor(np.concatenate((Y_train_unlabeled, Y_train_unlabeled_partial_leftout_subject), axis=0))
                    else:
                        X_train_finetuning = torch.tensor(X_train_labeled_partial_leftout_subject)
                        Y_train_finetuning = torch.tensor(Y_train_labeled_partial_leftout_subject)
                        X_train_finetuning_unlabeled = torch.tensor(X_train_unlabeled_partial_leftout_subject)
                        Y_train_finetuning_unlabeled = torch.tensor(Y_train_unlabeled_partial_leftout_subject)
                else:
                    if proportion_to_keep_of_leftout_subject_for_training>0:
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
        
        if args.turn_on_unlabeled_domain_adaptation and proportion_unlabeled_of_training_subjects>0:
            print("Size of X_train_unlabeled:     ", X_train_unlabeled.shape)
            print("Size of Y_train_unlabeled:     ", Y_train_unlabeled.shape)
        if args.turn_on_unlabeled_domain_adaptation and proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
            print("Size of X_train_finetuning_unlabeled:     ", X_train_finetuning_unlabeled.shape)
            print("Size of Y_train_finetuning_unlabeled:     ", Y_train_finetuning_unlabeled.shape)
            
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

def ceildiv(a, b):
        return -(a // -b)
    
if args.turn_on_unlabeled_domain_adaptation:
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
        # 'num_train_iter': args.epochs * ceildiv(X_train.shape[0], args.batch_size),
        # 'num_eval_iter': ceildiv(X_train.shape[0], args.batch_size),
        # 'num_log_iter': ceildiv(X_train.shape[0], args.batch_size),
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
        'num_classes': numGestures,
        'input_size': 224,
        'data_dir': './data',

        # algorithm specific configs
        'hard_label': True,
        'uratio': 1.5,
        'ulb_loss_ratio': 1.0,

        # device configs
        'gpu': args.gpu,
        'world_size': 1,
        'distributed': False,
    } 
    
    semilearn_config = get_config(semilearn_config_dict)

    if args.model == 'vit_tiny_patch2_32':
        semilearn_transform = transforms.Compose([transforms.Resize((32,32)), ToNumpy()])
    else: 
        semilearn_transform = transforms.Compose([transforms.Resize((224,224)), ToNumpy()])
    
    labeled_dataset = BasicDataset(semilearn_config, X_train, torch.argmax(Y_train, dim=1), semilearn_config.num_classes, semilearn_transform, is_ulb=False)
    if proportion_unlabeled_of_training_subjects>0:
        unlabeled_dataset = BasicDataset(semilearn_config, X_train_unlabeled, torch.argmax(Y_train_unlabeled, dim=1), semilearn_config.num_classes, semilearn_transform, 
                                        is_ulb=True, strong_transform=semilearn_transform)
        # proportion_unlabeled_to_use = args.proportion_unlabeled_data_from_training_subjects
    elif proportion_unlabeled_of_proportion_to_keep_of_leftout>0 or args.load_unlabeled_data_jehan:
        unlabeled_dataset = BasicDataset(semilearn_config, X_train_finetuning_unlabeled, torch.argmax(Y_train_finetuning_unlabeled, dim=1), semilearn_config.num_classes, semilearn_transform, 
                                        is_ulb=True, strong_transform=semilearn_transform)
        # proportion_unlabeled_to_use = args.proportion_unlabeled_data_from_leftout_subject
    if args.pretrain_and_finetune:
        finetune_dataset = BasicDataset(semilearn_config, X_train_finetuning, torch.argmax(Y_train_finetuning, dim=1), semilearn_config.num_classes, semilearn_transform, is_ulb=False)
        finetune_unlabeled_dataset = BasicDataset(semilearn_config, X_train_finetuning_unlabeled, torch.argmax(Y_train_finetuning_unlabeled, dim=1), 
                                                  semilearn_config.num_classes, semilearn_transform, is_ulb=True, strong_transform=semilearn_transform)
        
    validation_dataset = BasicDataset(semilearn_config, X_validation, torch.argmax(Y_validation, dim=1), semilearn_config.num_classes, semilearn_transform, is_ulb=False)

    proportion_unlabeled_to_use = len(unlabeled_dataset) / (len(labeled_dataset) + len(unlabeled_dataset)) 
    labeled_batch_size = int(semilearn_config.batch_size * (1-proportion_unlabeled_to_use))
    unlabeled_batch_size = int(semilearn_config.batch_size * proportion_unlabeled_to_use)
    if labeled_batch_size + unlabeled_batch_size < semilearn_config.batch_size:
        if labeled_batch_size < unlabeled_batch_size:
            labeled_batch_size += 1
        else:
            unlabeled_batch_size += 1
        
    labeled_iters = args.epochs * ceildiv(len(labeled_dataset), labeled_batch_size)
    unlabeled_iters = args.epochs * ceildiv(len(unlabeled_dataset), unlabeled_batch_size)
    iters_for_loader = max(labeled_iters, unlabeled_iters)
    train_labeled_loader = get_data_loader(semilearn_config, labeled_dataset, labeled_batch_size, num_workers=multiprocessing.cpu_count()//8, 
                                           num_epochs=args.epochs, num_iters=iters_for_loader)
    if proportion_unlabeled_of_training_subjects>0 or proportion_unlabeled_of_proportion_to_keep_of_leftout>0 or args.load_unlabeled_data_jehan:
        train_unlabeled_loader = get_data_loader(semilearn_config, unlabeled_dataset, unlabeled_batch_size, num_workers=multiprocessing.cpu_count()//8,
                                                 num_epochs=args.epochs, num_iters=iters_for_loader)
        
    semilearn_config.num_train_iter = iters_for_loader
    semilearn_config.num_eval_iter = ceildiv(iters_for_loader, args.epochs)
    semilearn_config.num_log_iter = ceildiv(iters_for_loader, args.epochs)
    
    semilearn_algorithm = get_algorithm(semilearn_config, get_net_builder(semilearn_config.net, from_name=False), tb_log=None, logger=None)
    semilearn_algorithm.model = send_model_cuda(semilearn_config, semilearn_algorithm.model)
    semilearn_algorithm.ema_model = send_model_cuda(semilearn_config, semilearn_algorithm.ema_model, clip_batch=False)
    
    print("Batches per epoch:", semilearn_config.num_eval_iter)
        
    if args.pretrain_and_finetune:
        proportion_unlabeled_to_use = len(finetune_unlabeled_dataset) / (len(finetune_dataset) + len(finetune_unlabeled_dataset))
        labeled_batch_size = int(semilearn_config.batch_size * (1-proportion_unlabeled_to_use))
        unlabeled_batch_size = int(semilearn_config.batch_size * proportion_unlabeled_to_use)
        if labeled_batch_size + unlabeled_batch_size < semilearn_config.batch_size:
            if labeled_batch_size < unlabeled_batch_size:
                labeled_batch_size += 1
            else:
                unlabeled_batch_size += 1
        labeled_iters = args.epochs * ceildiv(len(finetune_dataset), labeled_batch_size)
        unlabeled_iters = args.epochs * ceildiv(len(finetune_unlabeled_dataset), unlabeled_batch_size)
        iters_for_loader = max(labeled_iters, unlabeled_iters)
        train_finetuning_loader = get_data_loader(semilearn_config, finetune_dataset, labeled_batch_size, num_workers=multiprocessing.cpu_count()//8,
                                                  num_epochs=args.epochs, num_iters=iters_for_loader)
        train_finetuning_unlabeled_loader = get_data_loader(semilearn_config, finetune_unlabeled_dataset, unlabeled_batch_size, num_workers=multiprocessing.cpu_count()//8,
                                                            num_epochs=args.epochs, num_iters=iters_for_loader)
    validation_loader = get_data_loader(semilearn_config, validation_dataset, semilearn_config.eval_batch_size, num_workers=multiprocessing.cpu_count()//8)

else:
    if args.model == 'resnet50_custom':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model = nn.Sequential(*list(model.children())[:-4])
        # #model = nn.Sequential(*list(model.children())[:-4])
        num_features = model[-1][-1].conv3.out_channels
        # #num_features = model.fc.in_features
        dropout = 0.5
        model.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
        model.add_module('fc1', nn.Linear(num_features, 512))
        model.add_module('relu', nn.ReLU())
        model.add_module('dropout1', nn.Dropout(dropout))
        model.add_module('fc3', nn.Linear(512, numGestures))
        model.add_module('softmax', nn.Softmax(dim=1))
    elif args.model == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Replace the last fully connected layer
        num_ftrs = model.fc.in_features  # Get the number of input features of the original fc layer
        model.fc = nn.Linear(num_ftrs, numGestures)  # Replace with a new linear layer
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
    elif args.model == 'vit_tiny_patch2_32':
        pretrain_path = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth"
        model = VisualTransformer.vit_tiny_patch2_32(pretrained=True, pretrained_path=pretrain_path, num_classes=numGestures)
    elif args.model == 'MLP' or args.model == 'SVC' or args.model == 'RF':
        model = None # Will be initialized later
    else: 
        # model_name = 'efficientnet_b0'  # or 'efficientnet_b1', ..., 'efficientnet_b7'
        # model_name = 'tf_efficientnet_b3.ns_jft_in1k'
        model = timm.create_model(model_name, pretrained=True, num_classes=numGestures)
        # # Load the Vision Transformer model
        # model_name = 'vit_base_patch16_224'  # This is just one example, many variations exist
        # model = timm.create_model(model_name, pretrained=True, num_classes=utils.numGestures)

class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        if self.transform:
            x = self.transform(x)
        return x, y

if not args.turn_on_unlabeled_domain_adaptation:
    if args.model not in ['MLP', 'SVC', 'RF']:
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

    if args.model == 'vit_tiny_patch2_32':
        resize_transform = transforms.Compose([transforms.Resize((32,32)), ToNumpy()])
    else:
        resize_transform = transforms.Compose([transforms.Resize((224,224)), ToNumpy()])

    train_dataset = CustomDataset(X_train, Y_train, transform=resize_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)
    val_dataset = CustomDataset(X_validation, Y_validation, transform=resize_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)
    if (args.held_out_test):
        test_dataset = CustomDataset(X_test, Y_test, transform=resize_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    learn = args.learning_rate
    if args.model not in ['MLP', 'SVC', 'RF']:
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
    wandb_runname += '_rms-'+str(args.rms_input_windowsize)
if args.turn_on_magnitude:  
    wandb_runname += '_mag-'
if args.leftout_subject != 0:
    wandb_runname += '_LOSO-'+str(args.leftout_subject)
wandb_runname += '_' + model_name
if (exercises and not args.partial_dataset_ninapro):
    wandb_runname += '_exer-' + ''.join(character for character in str(args.exercises) if character.isalnum())
if args.dataset == "OzdemirEMG":
    if args.full_dataset_ozdemir:
        wandb_runname += '_full'
    else:
        wandb_runname += '_partial'
if args.dataset == "ninapro-db2" or args.dataset == "ninapro-db5":
    if args.partial_dataset_ninapro:
        wandb_runname += '_partial'
if args.turn_on_spectrogram:
    wandb_runname += '_spect'
if args.turn_on_cwt:
    wandb_runname += '_cwt'
if args.turn_on_hht:
    wandb_runname += '_hht'
if args.reduce_training_data_size:
    wandb_runname += '_reduced-training-data-size-' + str(args.reduced_training_data_size)
if args.leave_n_subjects_out_randomly != 0:
    wandb_runname += '_leave_n_subjects_out-'+str(args.leave_n_subjects_out_randomly)
if args.turn_off_scaler_normalization:
    wandb_runname += '_no-scal-norm'
if args.target_normalize > 0:
    wandb_runname += '_targ-norm'
if args.load_few_images:
    wandb_runname += '_load-few'
if args.transfer_learning:
    wandb_runname += '_tran-learn'
    wandb_runname += '-prop-' + str(args.proportion_transfer_learning_from_leftout_subject)
if args.cross_validation_for_time_series:   
    wandb_runname += '_cv-for-ts'
if args.reduce_data_for_transfer_learning != 1:
    wandb_runname += '_red-data-for-tran-learn-' + str(args.reduce_data_for_transfer_learning)
if args.leave_one_session_out:
    wandb_runname += '_leave-one-sess-out'
if args.leave_one_subject_out:
    wandb_runname += '_loso'
if args.one_subject_for_training_set_for_session_test:
    wandb_runname += '_one-subj-for-training-set'
if args.held_out_test:
    wandb_runname += '_held-out'
if args.pretrain_and_finetune:
    wandb_runname += '_pretrain-finetune'
if args.turn_on_unlabeled_domain_adaptation:
    wandb_runname += '_unlabeled-adapt'
    wandb_runname += '-algo-' + args.unlabeled_algorithm
    wandb_runname += '-prop-unlabel-leftout' + str(args.proportion_unlabeled_data_from_leftout_subject)
if args.proportion_data_from_training_subjects<1.0:
    wandb_runname += '_train-subj-prop-' + str(args.proportion_data_from_training_subjects)
if args.proportion_unlabeled_data_from_training_subjects>0:
    wandb_runname += '_unlabel-subj-prop-' + str(args.proportion_unlabeled_data_from_training_subjects)
if args.load_unlabeled_data_jehan:
    wandb_runname += '_load-unlabel-data-jehan'

if (args.held_out_test):
    if args.turn_on_kfold:
        project_name += '_k-fold-'+str(args.kfold)
    else:
        project_name += '_heldout'
elif args.leave_one_subject_out:
    project_name += '_LOSO'
elif args.leave_one_session_out:
    project_name += '_leave-one-session-out'
    
def freezeAllLayersButLastLayer(model):
    # Convert model children to a list
    children = list(model.children())

    # Freeze parameters in all children except the last one
    for child in children[:-1]:
        for param in child.parameters():
            param.requires_grad = False

    # Unfreeze the last layer
    for param in children[-1].parameters():
        param.requires_grad = True

def unfreezeAllLayers(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model

project_name += args.project_name_suffix

run = wandb.init(name=wandb_runname, project=project_name)
wandb.config.lr = args.learning_rate

if args.leave_n_subjects_out_randomly != 0:
    wandb.config.left_out_subjects = leaveOutIndices

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
print("Device:", device)
if not args.turn_on_unlabeled_domain_adaptation and args.model not in ['MLP', 'SVC', 'RF']:
    model.to(device)

    wandb.watch(model)

testrun_foldername = f'test/{project_name}/{wandb_runname}/{formatted_datetime}/'
# Make folder if it doesn't exist
if not os.path.exists(testrun_foldername):
    os.makedirs(testrun_foldername)
model_filename = f'{testrun_foldername}model_{formatted_datetime}.pth'

if (exercises):
    if not args.partial_dataset_ninapro:
        gesture_labels = utils.gesture_labels['Rest']
        for exercise_set in args.exercises:
            gesture_labels = gesture_labels + utils.gesture_labels[exercise_set]
    else:
        gesture_labels = utils.partial_gesture_labels
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

if args.held_out_test:
    # Plot and log images
    utils.plot_average_images(X_test, np.argmax(Y_test.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'test')
    utils.plot_first_fifteen_images(X_test, np.argmax(Y_test.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'test')

utils.plot_average_images(X_validation, np.argmax(Y_validation.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')
utils.plot_first_fifteen_images(X_validation, np.argmax(Y_validation.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')

utils.plot_average_images(X_train, np.argmax(Y_train.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'train')
utils.plot_first_fifteen_images(X_train, np.argmax(Y_train.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'train')

if args.pretrain_and_finetune:
    utils.plot_average_images(X_train_finetuning, np.argmax(Y_train_finetuning.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'train_finetuning')
    utils.plot_first_fifteen_images(X_train_finetuning, np.argmax(Y_train_finetuning.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'train_finetuning')

if args.turn_on_unlabeled_domain_adaptation:
    print("Pretraining the model...")
    semilearn_algorithm.loader_dict = {}
    semilearn_algorithm.loader_dict['train_lb'] = train_labeled_loader
    if proportion_unlabeled_of_training_subjects>0 or proportion_unlabeled_of_proportion_to_keep_of_leftout>0 or args.load_unlabeled_data_jehan:
        semilearn_algorithm.loader_dict['train_ulb'] = train_unlabeled_loader
    semilearn_algorithm.loader_dict['eval'] = validation_loader
    semilearn_algorithm.scheduler = None
    
    semilearn_algorithm.train()
    
    if args.pretrain_and_finetune:
        print("Finetuning the model...")
        run = wandb.init(name=wandb_runname+"_unlab_finetune", project=project_name)
        wandb.config.lr = args.learning_rate
        
        semilearn_config_dict['num_train_iter'] = semilearn_config.num_train_iter + iters_for_loader
        semilearn_config_dict['num_eval_iter'] = ceildiv(iters_for_loader, args.finetuning_epochs)
        semilearn_config_dict['num_log_iter'] = ceildiv(iters_for_loader, args.finetuning_epochs)
        semilearn_config_dict['epoch'] = args.finetuning_epochs + args.epochs
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
        
        if proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
            semilearn_algorithm.loader_dict['train_ulb'] = train_finetuning_unlabeled_loader
        elif proportion_unlabeled_of_training_subjects>0 or args.load_unlabeled_data_jehan:
            semilearn_algorithm.loader_dict['train_ulb'] = train_unlabeled_loader

        semilearn_algorithm.loader_dict['eval'] = validation_loader
        semilearn_algorithm.train()

else: 
    if args.model in ['MLP', 'SVC', 'RF']:
        class MLP(nn.Module):
            def __init__(self, input_size, hidden_sizes, output_size):
                super(MLP, self).__init__()
                self.hidden_layers = nn.ModuleList()
                self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
                for i in range(1, len(hidden_sizes)):
                    self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

            def forward(self, x):
                for hidden in self.hidden_layers:
                    x = F.relu(hidden(x))
                x = self.output_layer(x)
                return x
            
        def get_data_from_loader(loader):
            X = []
            Y = []
            for X_batch, Y_batch in tqdm(loader, desc="Batches convert to Numpy"):
                # Flatten each image from [batch_size, 3, 224, 224] to [batch_size, 3*224*224]
                X_batch_flat = X_batch.view(X_batch.size(0), -1).cpu().numpy().astype(np.float64)
                Y_batch_indices = torch.argmax(Y_batch, dim=1)  # Convert one-hot to class indices
                X.append(X_batch_flat)
                Y.append(Y_batch_indices.cpu().numpy().astype(np.int64))
            return np.vstack(X), np.hstack(Y)
        
        if args.model == 'MLP':
            # PyTorch MLP model
            input_size = 3 * 224 * 224  # Change according to your input size
            hidden_sizes = [512, 256]  # Example hidden layer sizes
            output_size = 10  # Number of classes
            model = MLP(input_size, hidden_sizes, output_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
        
        elif args.model == 'SVC':
            model = SVC(probability=True)
        
        elif args.model == 'RF':
            model = RandomForestClassifier()
            
        if args.model == 'MLP':
            # PyTorch training loop for MLP

            for epoch in tqdm(range(num_epochs), desc="Epoch"):
                model.train()

                # Metrics
                train_acc = torchmetrics.Accuracy().to(device)
                precision = torchmetrics.Precision().to(device)
                recall = torchmetrics.Recall().to(device)
                f1 = torchmetrics.F1().to(device)
                top5_acc = torchmetrics.Accuracy(top_k=5).to(device)

                train_loss = 0.0
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as t:
                    for X_batch, Y_batch in t:
                        X_batch = X_batch.view(X_batch.size(0), -1).to(device).to(torch.float32)
                        Y_batch = torch.argmax(Y_batch, dim=1).to(device).to(torch.int64)

                        optimizer.zero_grad()
                        output = model(X_batch)
                        loss = criterion(output, Y_batch)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        train_acc(output, Y_batch)
                        precision(output, Y_batch)
                        recall(output, Y_batch)
                        f1(output, Y_batch)
                        top5_acc(output, Y_batch)

                        if t.n % 10 == 0:
                            t.set_postfix({"Batch Loss": loss.item(), "Batch Acc": train_acc.compute().item()})

                        del X_batch, Y_batch, output
                        torch.cuda.empty_cache()

                # Validation
                model.eval()
                val_acc = torchmetrics.Accuracy().to(device)
                val_precision = torchmetrics.Precision().to(device)
                val_recall = torchmetrics.Recall().to(device)
                val_f1 = torchmetrics.F1().to(device)
                val_top5_acc = torchmetrics.Accuracy(top_k=5).to(device)
                
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, Y_batch in val_loader:
                        X_batch = X_batch.view(X_batch.size(0), -1).to(device).to(torch.float32)
                        Y_batch = torch.argmax(Y_batch, dim=1).to(device).to(torch.int64)

                        output = model(X_batch)
                        val_loss += criterion(output, Y_batch).item()
                        val_acc(output, Y_batch)
                        val_precision(output, Y_batch)
                        val_recall(output, Y_batch)
                        val_f1(output, Y_batch)
                        val_top5_acc(output, Y_batch)

                        del X_batch, Y_batch
                        torch.cuda.empty_cache()

                # Average the losses and print the metrics
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)

                tpr_results = ml_utils.evaluate_model_tpr_at_fpr(model, val_loader, device, numGestures)
                confidence_levels = ml_utils.evaluate_confidence_thresholding(model, val_loader, device, numGestures)

                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"Train Metrics: Acc: {train_acc.compute().item():.4f}, Precision: {precision.compute().item():.4f}, Recall: {recall.compute().item():.4f}, F1: {f1.compute().item():.4f}, Top-5 Acc: {top5_acc.compute().item():.4f}")
                print(f"Val Metrics: Acc: {val_acc.compute().item():.4f}, Precision: {val_precision.compute().item():.4f}, Recall: {val_recall.compute().item():.4f}, F1: {val_f1.compute().item():.4f}, Top-5 Acc: {val_top5_acc.compute().item():.4f}")
                for fpr, tprs in tpr_results.items():
                    print(f"TPR at {fpr}: {', '.join(f'{tpr:.4f}' for tpr in tprs)}")
                for confidence_level, accuracy in confidence_levels.items():
                    print(f"Accuracy at confidence level >{confidence_level}: {accuracy:.4f}")

                # Log metrics to wandb or any other tracking tool
                wandb.log({
                    "Epoch": epoch,
                    "Train Loss": train_loss,
                    "Train Acc": train_acc.compute().item(),
                    "Train Precision": precision.compute().item(),
                    "Train Recall": recall.compute().item(),
                    "Train F1": f1.compute().item(),
                    "Train Top-5 Acc": top5_acc.compute().item(),
                    "Valid Loss": val_loss,
                    "Valid Acc": val_acc.compute().item(),
                    "Valid Precision": val_precision.compute().item(),
                    "Valid Recall": val_recall.compute().item(),
                    "Valid F1": val_f1.compute().item(),
                    "Valid Top-5 Acc": val_top5_acc.compute().item(),
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                    **{f"TPR at {fpr}": tprs for fpr, tprs in tpr_results.items()}, 
                    **{f"Accuracy at confidence level >{confidence_level}": accuracy for confidence_level, accuracy in confidence_levels.items()}
                })


            torch.save(model.state_dict(), model_filename)
            wandb.save(f'model/modelParameters_{formatted_datetime}.pth')

        else:
            X_train, Y_train = get_data_from_loader(train_loader)
            X_val, Y_val = get_data_from_loader(val_loader)
            # X_test, Y_test = get_data_from_loader(test_loader)

            print("Data loaded")
            model.fit(X_train, Y_train)
            print("Model trained")
            train_preds = model.predict(X_train)
            print("Train predictions made")
            val_preds = model.predict(X_val)
            print("Validation predictions made")
            # test_preds = model.predict(X_test)

            train_acc = accuracy_score(Y_train, train_preds)
            val_acc = accuracy_score(Y_val, val_preds)
            # test_acc = accuracy_score(Y_test, test_preds)

            train_loss = log_loss(Y_train, model.predict_proba(X_train))
            val_loss = log_loss(Y_val, model.predict_proba(X_val))
            # test_loss = log_loss(Y_test, model.predict_proba(X_test))

            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
            # print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

            wandb.log({
                "Train Loss": train_loss,
                "Train Acc": train_acc,
                "Valid Loss": val_loss,
                "Valid Acc": val_acc,
                # "Test Loss": test_loss,
                # "Test Acc": test_acc
            })

    else: # CNN training
        # Metrics for training
        train_acc_metric = torchmetrics.Accuracy().to(device)
        train_precision_metric = torchmetrics.Precision().to(device)
        train_recall_metric = torchmetrics.Recall().to(device)
        train_f1_score_metric = torchmetrics.F1().to(device)
        train_top5_acc_metric = torchmetrics.Accuracy(top_k=5).to(device)

        # Metrics for validation
        val_acc_metric = torchmetrics.Accuracy().to(device)
        val_precision_metric = torchmetrics.Precision().to(device)
        val_recall_metric = torchmetrics.Recall().to(device)
        val_f1_score_metric = torchmetrics.F1().to(device)
        val_top5_acc_metric = torchmetrics.Accuracy(top_k=5).to(device)

        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            model.train()
            train_loss = 0.0

            # Reset training metrics at the start of each epoch
            train_acc_metric.reset()
            train_precision_metric.reset()
            train_recall_metric.reset()
            train_f1_score_metric.reset()
            train_top5_acc_metric.reset()

            for X_batch, Y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                X_batch = X_batch.to(device).to(torch.float32)
                Y_batch = Y_batch.to(device).to(torch.float32)
                Y_batch_long = torch.argmax(Y_batch, dim=1)

                optimizer.zero_grad()
                output = model(X_batch)
                if isinstance(output, dict):
                    output = output['logits']
                loss = criterion(output, Y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc_metric(output, Y_batch_long)
                train_precision_metric(output, Y_batch_long)
                train_recall_metric(output, Y_batch_long)
                train_f1_score_metric(output, Y_batch_long)
                train_top5_acc_metric(output, Y_batch_long)

                if t.n % 10 == 0:
                    t.set_postfix({
                        "Batch Loss": loss.item(), 
                        "Batch Acc": train_acc_metric.compute().item()
                    })

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_acc_metric.reset()
            val_precision_metric.reset()
            val_recall_metric.reset()
            val_f1_score_metric.reset()
            val_top5_acc_metric.reset()
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch = X_batch.to(device).to(torch.float32)
                    Y_batch = Y_batch.to(device).to(torch.float32)
                    Y_batch_long = torch.argmax(Y_batch, dim=1)

                    output = model(X_batch)
                    if isinstance(output, dict):
                        output = output['logits']
                    val_loss += criterion(output, Y_batch).item()
                    val_acc_metric(output, Y_batch_long)
                    val_precision_metric(output, Y_batch_long)
                    val_recall_metric(output, Y_batch_long)
                    val_f1_score_metric(output, Y_batch_long)
                    val_top5_acc_metric(output, Y_batch_long)

            # Calculate average loss and metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_acc_metric.compute()
            train_precision = train_precision_metric.compute()
            train_recall = train_recall_metric.compute()
            train_f1_score = train_f1_score_metric.compute()
            train_top5_acc = train_top5_acc_metric.compute()
            val_acc = val_acc_metric.compute()
            val_precision = val_precision_metric.compute()
            val_recall = val_recall_metric.compute()
            val_f1_score = val_f1_score_metric.compute()
            val_top5_acc = val_top5_acc_metric.compute()

            tpr_results = ml_utils.evaluate_model_tpr_at_fpr(model, val_loader, device, numGestures)
            confidence_levels = ml_utils.evaluate_confidence_thresholding(model, val_loader, device)

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.4f} | Train Precision: {train_precision:.4f} | Train Recall: {train_recall:.4f} | Train F1: {train_f1_score:.4f} | Train Top-5 Acc: {train_top5_acc:.4f}")
            print(f"Val Accuracy: {val_acc:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1_score:.4f} | Val Top-5 Acc: {val_top5_acc:.4f}")
            for fpr, tprs in tpr_results.items():
                print(f"TPR at {fpr}: {', '.join(f'{tpr:.4f}' for tpr in tprs)}")
            for confidence_level, acc in confidence_levels.items():
                print(f"Accuracy at {confidence_level} confidence level: {acc:.4f}")

            wandb.log({
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Train Acc": train_acc,
                "Train Precision": train_precision,
                "Train Recall": train_recall,
                "Train F1": train_f1_score,
                "Train Top-5 Acc": train_top5_acc,
                "Val Loss": val_loss,
                "Val Acc": val_acc,
                "Val Precision": val_precision,
                "Val Recall": val_recall,
                "Val F1": val_f1_score,
                "Val Top-5 Acc": val_top5_acc,
                "Learning Rate": optimizer.param_groups[0]['lr'], 
                **{f"TPR at {fpr}": tprs for fpr, tprs in tpr_results.items()}, 
                **{f"Accuracy at {confidence_level} confidence level": acc for confidence_level, acc in confidence_levels.items()}
            })

        torch.save(model.state_dict(), model_filename)
        wandb.save(f'model/modelParameters_{formatted_datetime}.pth')

        if args.pretrain_and_finetune:
            run.finish()
            run = wandb.init(name=wandb_runname+"_finetune", project=project_name)
            num_epochs = args.finetuning_epochs
            # train more on fine tuning dataset
            finetune_dataset = CustomDataset(X_train_finetuning, Y_train_finetuning, transform=resize_transform)
            finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)
            # Initialize metrics for finetuning training
            finetune_train_acc_metric = torchmetrics.Accuracy().to(device)
            finetune_precision_metric = torchmetrics.Precision().to(device)
            finetune_recall_metric = torchmetrics.Recall().to(device)
            finetune_f1_score_metric = torchmetrics.F1().to(device)
            finetune_top5_acc_metric = torchmetrics.Accuracy(top_k=5).to(device)

            # Initialize metrics for finetuning validation
            finetune_val_acc_metric = torchmetrics.Accuracy().to(device)
            finetune_val_precision_metric = torchmetrics.Precision().to(device)
            finetune_val_recall_metric = torchmetrics.Recall().to(device)
            finetune_val_f1_score_metric = torchmetrics.F1().to(device)
            finetune_val_top5_acc_metric = torchmetrics.Accuracy(top_k=5).to(device)

            for epoch in tqdm(range(num_epochs), desc="Finetuning Epoch"):
                model.train()
                train_loss = 0.0

                # Reset finetuning training metrics at the start of each epoch
                finetune_train_acc_metric.reset()
                finetune_precision_metric.reset()
                finetune_recall_metric.reset()
                finetune_f1_score_metric.reset()
                finetune_top5_acc_metric.reset()

                for X_batch, Y_batch in tqdm(finetune_loader, desc=f"Finetuning Epoch {epoch+1}/{num_epochs}", leave=False):
                    X_batch = X_batch.to(device).to(torch.float32)
                    Y_batch = Y_batch.to(device).to(torch.float32)
                    Y_batch_long = torch.argmax(Y_batch, dim=1)

                    optimizer.zero_grad()
                    output = model(X_batch)
                    if isinstance(output, dict):
                        output = output['logits']
                    loss = criterion(output, Y_batch)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    finetune_train_acc_metric(output, Y_batch_long)
                    finetune_precision_metric(output, Y_batch_long)
                    finetune_recall_metric(output, Y_batch_long)
                    finetune_f1_score_metric(output, Y_batch_long)
                    finetune_top5_acc_metric(output, Y_batch_long)

                    if t.n % 10 == 0:
                        t.set_postfix({
                            "Batch Loss": loss.item(), 
                            "Batch Acc": finetune_train_acc_metric.compute().item()
                        })

                # Finetuning Validation
                model.eval()
                val_loss = 0.0
                finetune_val_acc_metric.reset()
                finetune_val_precision_metric.reset()
                finetune_val_recall_metric.reset()
                finetune_val_f1_score_metric.reset()
                finetune_val_top5_acc_metric.reset()

                with torch.no_grad():
                    for X_batch, Y_batch in val_loader:
                        X_batch = X_batch.to(device).to(torch.float32)
                        Y_batch = Y_batch.to(device).to(torch.float32)
                        Y_batch_long = torch.argmax(Y_batch, dim=1)

                        output = model(X_batch)
                        if isinstance(output, dict):
                            output = output['logits']
                        val_loss += criterion(output, Y_batch).item()
                        finetune_val_acc_metric(output, Y_batch_long)
                        finetune_val_precision_metric(output, Y_batch_long)
                        finetune_val_recall_metric(output, Y_batch_long)
                        finetune_val_f1_score_metric(output, Y_batch_long)
                        finetune_val_top5_acc_metric(output, Y_batch_long)

                # Calculate average metrics
                train_loss /= len(finetune_loader)
                finetune_train_acc = finetune_train_acc_metric.compute()
                finetune_train_precision = finetune_precision_metric.compute()
                finetune_train_recall = finetune_recall_metric.compute()
                finetune_train_f1_score = finetune_f1_score_metric.compute()
                finetune_train_top5_acc = finetune_top5_acc_metric.compute()
                val_loss /= len(val_loader)
                finetune_val_acc = finetune_val_acc_metric.compute()
                finetune_val_precision = finetune_val_precision_metric.compute()
                finetune_val_recall = finetune_val_recall_metric.compute()
                finetune_val_f1_score = finetune_val_f1_score_metric.compute()
                finetune_val_top5_acc = finetune_val_top5_acc_metric.compute()

                tpr_results = ml_utils.evaluate_model_tpr_at_fpr(model, val_loader, device, numGestures)
                confidence_levels = ml_utils.evaluate_confidence_thresholding(model, val_loader, device)

                print(f"Finetuning Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"Train Accuracy: {finetune_train_acc:.4f} | Train Precision: {finetune_train_precision:.4f} | Train Recall: {finetune_train_recall:.4f} | Train F1: {finetune_train_f1_score:.4f} | Train Top-5 Acc: {finetune_train_top5_acc:.4f}")
                print(f"Val Accuracy: {finetune_val_acc:.4f} | Val Precision: {finetune_val_precision:.4f} | Val Recall: {finetune_val_recall:.4f} | Val F1: {finetune_val_f1_score:.4f} | Val Top-5 Acc: {finetune_val_top5_acc:.4f}")
                for fpr, tprs in tpr_results.items():
                    print(f"TPR at {fpr}: {', '.join(f'{tpr:.4f}' for tpr in tprs)}")
                for confidence_level, acc in confidence_levels.items():
                    print(f"Accuracy at {confidence_level} confidence level: {acc:.4f}")

                wandb.log({
                    "Finetuning Epoch": epoch,
                    "Train Loss": train_loss,
                    "Train Acc": finetune_train_acc,
                    "Train Precision": finetune_train_precision,
                    "Train Recall": finetune_train_recall,
                    "Train F1": finetune_train_f1_score,
                    "Train Top-5 Acc": finetune_train_top5_acc,
                    "Valid Loss": val_loss,
                    "Valid Acc": finetune_val_acc,
                    "Valid Precision": finetune_val_precision,
                    "Valid Recall": finetune_val_recall,
                    "Valid F1": finetune_val_f1_score,
                    "Valid Top-5 Acc": finetune_val_top5_acc,
                    "Learning Rate": optimizer.param_groups[0]['lr'], 
                    **{f"TPR at {fpr}": tprs for fpr, tprs in tpr_results.items()}, 
                    **{f"Accuracy at {confidence_level} confidence level": acc for confidence_level, acc in confidence_levels.items()}
                })
                
        # Testing
        # Initialize metrics for testing
        test_acc_metric = torchmetrics.Accuracy().to(device)
        test_precision_metric = torchmetrics.Precision().to(device)
        test_recall_metric = torchmetrics.Recall().to(device)
        test_f1_score_metric = torchmetrics.F1().to(device)
        test_top5_acc_metric = torchmetrics.Accuracy(top_k=5).to(device)

        # Assuming model, criterion, device, and test_loader are defined
        if args.held_out_test:
            model.eval()
            test_loss = 0.0

            # Reset test metrics
            test_acc_metric.reset()
            test_precision_metric.reset()
            test_recall_metric.reset()
            test_f1_score_metric.reset()
            test_top5_acc_metric.reset()

            pred = []
            true = []

            with torch.no_grad():
                for X_batch, Y_batch in test_loader:
                    X_batch = X_batch.to(device).to(torch.float32)
                    Y_batch = Y_batch.to(device).to(torch.float32)
                    Y_batch_long = torch.argmax(Y_batch, dim=1)

                    output = model(X_batch)
                    if isinstance(output, dict):
                        output = output['logits']
                    pred.extend(torch.argmax(output, dim=1).cpu().detach().numpy())
                    true.extend(Y_batch_long.cpu().detach().numpy())

                    test_loss += criterion(output, Y_batch).item()
                    test_acc_metric(output, Y_batch_long)
                    test_precision_metric(output, Y_batch_long)
                    test_recall_metric(output, Y_batch_long)
                    test_f1_score_metric(output, Y_batch_long)
                    test_top5_acc_metric(output, Y_batch_long)

            # Calculate average loss and metrics
            test_loss /= len(test_loader)
            test_acc = test_acc_metric.compute()
            test_precision = test_precision_metric.compute()
            test_recall = test_recall_metric.compute()
            test_f1_score = test_f1_score_metric.compute()
            test_top5_acc = test_top5_acc_metric.compute()
            tpr_results = ml_utils.evaluate_model_tpr_at_fpr(model, test_loader, device, numGestures)
            confidence_levels = ml_utils.evaluate_confidence_thresholding(model, test_loader, device)

            print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1 Score: {test_f1_score:.4f} | Test Top-5 Accuracy: {test_top5_acc:.4f}")
            for fpr, tprs in tpr_results.items():
                print(f"TPR at {fpr}: {', '.join(f'{tpr:.4f}' for tpr in tprs)}")
            for confidence_level, acc in confidence_levels.items():
                print(f"Accuracy at {confidence_level} confidence level: {acc:.4f}")

            wandb.log({
                "Test Loss": test_loss,
                "Test Acc": test_acc,
                "Test Precision": test_precision,
                "Test Recall": test_recall,
                "Test F1": test_f1_score,
                "Test Top-5 Acc": test_top5_acc, 
                **{f"TPR at {fpr}": tprs for fpr, tprs in tpr_results.items()}, 
                **{f"Accuracy at {confidence_level} confidence level": acc for confidence_level, acc in confidence_levels.items()}
            })
            
            # %% Confusion Matrix
            # Plot and log confusion matrix in wandb
            utils.plot_confusion_matrix(true, pred, gesture_labels, testrun_foldername, args, formatted_datetime, 'test')

        # Load validation in smaller batches for memory purposes
        torch.cuda.empty_cache()  # Clear cache if needed

        model.eval()
        with torch.no_grad():
            validation_predictions = []
            for X_batch, Y_batch in tqdm(val_loader, desc="Validation Batch Loading"):
                X_batch = X_batch.to(device).to(torch.float32)
                outputs = model(X_batch)
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                validation_predictions.extend(preds)

        utils.plot_confusion_matrix(np.argmax(Y_validation.cpu().detach().numpy(), axis=1), np.array(validation_predictions), gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')   

        # Load training in smaller batches for memory purposes
        torch.cuda.empty_cache()  # Clear cache if needed

        model.eval()
        train_loader_unshuffled = DataLoader(train_dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)
        with torch.no_grad():
            train_predictions = []
            for X_batch, Y_batch in tqdm(train_loader_unshuffled, desc="Training Batch Loading"):
                X_batch = X_batch.to(device).to(torch.float32)
                outputs = model(X_batch)
                if isinstance(outputs, dict):
                        outputs = outputs['logits']
                preds = torch.argmax(outputs, dim=1)
                train_predictions.extend(preds.cpu().detach().numpy())

        utils.plot_confusion_matrix(np.argmax(Y_train.cpu().detach().numpy(), axis=1), np.array(train_predictions), gesture_labels, testrun_foldername, args, formatted_datetime, 'train')
            
    run.finish()