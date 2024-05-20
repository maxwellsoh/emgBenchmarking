import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import wandb
from sklearn.metrics import confusion_matrix
import pandas as pd
import multiprocessing
from tqdm import tqdm
import argparse
import random 
# import utils_OzdemirEMG as utils
from sklearn.model_selection import StratifiedKFold
import os
import datetime
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
from torch.utils.data import Dataset
import VisualTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F
import subprocess
import get_datasets
import parse_config as parse_config


# way to get round importing multiple utils


# TODO: rename this class 
class my_class(object):
    
    def __init__(self):
        self.args = parse_config.get()    # Parse arguments
        self.exercises = False 
        self.utils = None  
        self.current_datetime = datetime.datetime.now()
        self.formatted_datetime = self.current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        self.width = None
        self.length = None
        self.emg = None
        self.lables = None
        self.leaveOut = None
        self.numGestures = None
        self.project_name = None
    
    def run(self):
        self.initialize()
        self.setup_datasets()
        self.add_lables()
        self.reshape_data()
        self.rest()

    def initialize(self):
        if self.args.model == "MLP" or self.args.model == "SVC" or self.args.model == "RF":
            print("Warning: not using pytorch, many arguments will be ignored")
            if self.args.turn_on_unlabeled_domain_adaptation:
                NotImplementedError("Cannot use unlabeled domain adaptation with MLP, SVC, or RF")
            if self.args.pretrain_and_finetune:
                NotImplementedError("Cannot use pretrain and finetune with MLP, SVC, or RF")

        self.setup_datasets()
        self.print_arguments()

        # 0 for no LOSO; participants here are 1-13
        self.leaveOut = int(self.args.leftout_subject)

        # Set seeds for reproducibility
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_datasets(self):
        """
        Checks for needed dataset and downlaods if not present.
        Imports the relevant utils and sets up project_name and exercises. 
        If no dataset is provided, defaults to OzdemirEMG. 
        """
        #TODO: get rid of the chains and have it just do one chain
        if (self.args.dataset == "uciEMG"):
            if (not os.path.exists("./uciEMG")):
                print("uciEMG dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--UCI'])
            import utils_UCI as utils
            print(f"The dataset being tested is uciEMG")
            self.project_name = 'emg_benchmarking_uci'

        elif (self.args.dataset == "ninapro-db2"):
            if (not os.path.exists("./NinaproDB2")):
                print("NinaproDB2 dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--NinaproDB2'])
            import utils_NinaproDB2 as utils
            print(f"The dataset being tested is ninapro-db2")
            self.project_name = 'emg_benchmarking_ninapro-db2'
            exercises = True
            if self.args.leave_one_session_out:
                ValueError("leave-one-session-out not implemented for ninapro-db2; only one session exists")

        elif (self.args.dataset == "ninapro-db5"):
            if (not os.path.exists("./NinaproDB5")):
                print("NinaproDB5 dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--NinaproDB5'])

            import utils_NinaproDB5 as utils
            print(f"The dataset being tested is ninapro-db5")
            self.project_name = 'emg_benchmarking_ninapro-db5'
            exercises = True
            if self.args.leave_one_session_out:
                ValueError("leave-one-session-out not implemented for ninapro-db5; only one session exists")

        elif (self.args.dataset == "M_dataset"):
            if (not os.path.exists("./M_dataset")):
                print("M_dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--M_Dataset'])
            import utils_M_dataset as utils
            print(f"The dataset being tested is M_dataset")
            self.project_name = 'emg_benchmarking_M_dataset'
            if self.args.leave_one_session_out:
                ValueError("leave-one-session-out not implemented for M_dataset; only one session exists")

        elif (self.args.dataset == "hyser"):
            if (not os.path.exists("./hyser")):
                print("Hyser dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--hyser'])
            import utils_Hyser as utils
            print(f"The dataset being tested is hyser")
            self.project_name = 'emg_benchmarking_hyser'

        elif (self.args.dataset == "capgmyo"):
            if (not os.path.exists("./CapgMyo_B")):
                print("CapgMyo_B dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--CapgMyo_B'])
            import utils_CapgMyo as utils
            print(f"The dataset being tested is CapgMyo")
            self.project_name = 'emg_benchmarking_capgmyo'
            if self.args.leave_one_session_out:
                ValueError("leave-one-session-out not implemented for CapgMyo; only one session exists")

        elif (self.args.dataset == "jehan"):
            if (not os.path.exists("./Jehan_Dataset")):
                print("Jehan dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--Jehan_Dataset'])
            import utils_JehanData as utils
            print(f"The dataset being tested is JehanDataset")
            self.project_name = 'emg_benchmarking_jehandataset'
            if self.args.leave_one_session_out:
                ValueError("leave-one-session-out not implemented for JehanDataset; only one session exists")
            
        else:
            import utils_OzdemirEMG as utils
            if (not os.path.exists("./OzdemirEMG")):
                print("Ozdemir dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--OzdemirEMG'])
            print(f"The dataset being tested is OzdemirEMG")
            self.project_name = 'emg_benchmarking_ozdemir'
            if self.args.full_dataset_ozdemir:
                print(f"Using the full dataset for Ozdemir EMG")
                utils.gesture_labels = utils.gesture_labels_full
                utils.numGestures = len(utils.gesture_labels)
            else: 
                print(f"Using the partial dataset for Ozdemir EMG")
                utils.gesture_labels = utils.gesture_labels_partial
                utils.numGestures = len(utils.gesture_labels)
            if self.args.leave_one_session_out:
                ValueError("leave-one-session-out not implemented for OzdemirEMG; only one session exists")

        self.utils = utils

    def print_arguments(self):
        """
        Lists out configuration values. 
        """
        
        # TODO: stop chaining everything 

        # Use the arguments
        print(f"The value of --leftout_subject is {self.args.leftout_subject}")
        print(f"The value of --seed is {self.args.seed}")
        print(f"The value of --epochs is {self.args.epochs}")
        print(f"The model to use is {self.args.model}")
        if self.args.turn_on_kfold:
            print(f"The value of --turn_on_kfold is {self.args.turn_on_kfold}")
            print(f"The value of --kfold is {self.args.kfold}")
            print(f"The value of --fold_index is {self.args.fold_index}")
            
        if self.args.turn_on_cyclical_lr:
            print(f"The value of --turn_on_cyclical_lr is {self.args.turn_on_cyclical_lr}")
        if self.args.turn_on_cosine_annealing:
            print(f"The value of --turn_on_cosine_annealing is {self.args.turn_on_cosine_annealing}")
        if self.args.turn_on_cyclical_lr and self.args.turn_on_cosine_annealing:
            print("Cannot turn on both cyclical learning rate and cosine annealing")
            exit()
        if self.args.turn_on_rms:
            print(f"The value of --turn_on_rms is {self.args.turn_on_rms}")
            print(f"The value of --rms_input_windowsize is {self.args.rms_input_windowsize}")
        if self.args.turn_on_magnitude:
            print(f"The value of --turn_on_magnitude is {self.args.turn_on_magnitude}")
        if self.exercises:
            print(f"The value of --exercises is {self.args.exercises}")
        print(f"The value of --project_name_suffix is {self.args.project_name_suffix}")
        print(f"The value of --turn_on_spectrogram is {self.args.turn_on_spectrogram}")
        print(f"The value of --turn_on_cwt is {self.args.turn_on_cwt}")
        print(f"The value of --turn_on_hht is {self.args.turn_on_hht}")

        print(f"The value of --save_images is {self.args.save_images}")
        print(f"The value of --turn_off_scaler_normalization is {self.args.turn_off_scaler_normalization}")
        print(f"The value of --learning_rate is {self.args.learning_rate}")
        print(f"The value of --gpu is {self.args.gpu}")

        print(f"The value of --load_few_images is {self.args.load_few_images}")
        print(f"The value of --reduce_training_data_size is {self.args.reduce_training_data_size}")
        print(f"The value of --reduced_training_data_size is {self.args.reduced_training_data_size}")

        print(f"The value of --leave_n_subjects_out_randomly is {self.args.leave_n_subjects_out_randomly}")
        print(f"The value of --target_normalize is {self.args.target_normalize}")
        print(f"The value of --transfer_learning is {self.args.transfer_learning}")
        print(f"The value of --cross_validation_for_time_series is {self.args.cross_validation_for_time_series}")
        print(f"The value of --proportion_transfer_learning_from_leftout_subject is {self.args.proportion_transfer_learning_from_leftout_subject}")
        print(f"The value of --reduce_data_for_transfer_learning is {self.args.reduce_data_for_transfer_learning}")
        print(f"The value of --leave_one_session_out is {self.args.leave_one_session_out}")
        print(f"The value of --held_out_test is {self.args.held_out_test}")
        print(f"The value of --one_subject_for_training_set_for_session_test is {self.args.one_subject_for_training_set_for_session_test}")
        print(f"The value of --pretrain_and_finetune is {self.args.pretrain_and_finetune}")
        print(f"The value of --finetuning_epochs is {self.args.finetuning_epochs}")

        print(f"The value of --turn_on_unlabeled_domain_adaptation is {self.args.turn_on_unlabeled_domain_adaptation}")
        print(f"The value of --unlabeled_algorithm is {self.args.unlabeled_algorithm}")
        print(f"The value of --proportion_unlabeled_data_from_leftout_subject is {self.args.proportion_unlabeled_data_from_leftout_subject}")

        print(f"The value of --batch_size is {self.args.batch_size}")

        print(f"The value of --proportion_unlabeled_data_from_training_subjects is {self.args.proportion_unlabeled_data_from_training_subjects}")
        print(f"The value of --proportion_data_from_training_subjects is {self.args.proportion_data_from_training_subjects}")

        # Add date and time to filename

        print("------------------------------------------------------------------------------------------------------------------------")
        print("Starting run at", self.formatted_datetime)
        print("------------------------------------------------------------------------------------------------------------------------")
    
    def add_lables(self):
        """
        Adds labels to the data.
        """
        if self.exercises:
            self.emg = []
            self.labels = []

            if self.args.partial_dataset_ninapro:
                if self.args.dataset == "ninapro-db2":
                    self.args.exercises = [1]
                elif self.args.dataset == "ninapro-db5":
                    self.args.exercises = [2]

            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                for exercise in self.args.exercises:
                    emg_async = pool.map_async(self.utils.getEMG, list(zip([(i+1) for i in range(self.utils.num_subjects)], exercise*np.ones(self.utils.num_subjects).astype(int))))
                    self.emg.append(emg_async.get()) # (EXERCISE SET, SUBJECT, TRIAL, CHANNEL, TIME)
                    
                    labels_async = pool.map_async(self.utils.getLabels, list(zip([(i+1) for i in range(self.utils.num_subjects)], exercise*np.ones(self.utils.num_subjects).astype(int))))
                    self.labels.append(labels_async.get())
                    
                    assert len(self.emg[-1]) == len(self.labels[-1]), "Number of trials for EMG and labels do not match"
                    
            # Append exercise sets together and add dimensions to labels if necessary

            new_emg = []  # This will store the concatenated data for each subject
            new_labels = []  # This will store the concatenated labels for each subject
            self.numGestures = 0 # This will store the number of gestures for each subject

            for subject in range(self.utils.num_subjects): 
                subject_trials = []  # List to store trials for this subject across all exercise sets
                subject_labels = []  # List to store labels for this subject across all exercise sets
                
                for exercise_set in range(len(self.emg)):  
                    # Append the trials of this subject in this exercise set
                    subject_trials.append(self.emg[exercise_set][subject])
                    subject_labels.append(self.labels[exercise_set][subject])

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
                    if self.args.dataset == "ninapro-db5":
                        index_to_start_at = max(subject_labels_to_concatenate)
                    labels_set.append(subject_labels_to_concatenate)

                if self.args.partial_dataset_ninapro:
                    desired_gesture_labels = self.utils.partial_gesture_indices
                
                # Assuming labels are stored separately and need to be concatenated end-to-end
                concatenated_labels = np.concatenate(labels_set, axis=0) # (TRIAL)

                if self.args.partial_dataset_ninapro:
                    indices_for_partial_dataset = np.array([indices for indices, label in enumerate(concatenated_labels) if label in desired_gesture_labels])
                    concatenated_labels = concatenated_labels[indices_for_partial_dataset]
                    concatenated_trials = concatenated_trials[indices_for_partial_dataset]
                    # convert labels to indices
                    label_to_index = {label: index for index, label in enumerate(desired_gesture_labels)}
                    concatenated_labels = [label_to_index[label] for label in concatenated_labels]
                
                self.numGestures = len(np.unique(concatenated_labels))

                # Convert to one hot encoding
                concatenated_labels = np.eye(np.max(concatenated_labels) + 1)[concatenated_labels] # (TRIAL, GESTURE)

                # Append the concatenated trials to the new_emg list
                new_emg.append(concatenated_trials)
                new_labels.append(concatenated_labels)

            self.emg = [torch.from_numpy(emg_np) for emg_np in new_emg]
            self.labels = [torch.from_numpy(labels_np) for labels_np in new_labels]

        else: # Not exercises
            if (self.args.target_normalize):
                mins, maxes = self.utils.getExtrema(self.args.leftout_subject + 1)
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                    if self.args.leave_one_session_out:
                        NotImplementedError("leave-one-session-out not implemented with target_normalize yet")
                    emg_async = pool.map_async(self.utils.getEMG, [(i+1, mins, maxes, self.args.leftout_subject + 1) for i in range(self.utils.num_subjects)])
                    self.emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
                    
                    labels_async = pool.map_async(self.utils.getLabels, [(i+1) for i in range(self.utils.num_subjects)])
                    self.labels = labels_async.get()

            else: # Not target_normalize
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                    if self.args.leave_one_session_out: # based on 2 sessions for each subject
                        total_number_of_sessions = 2
                        self.emg = []
                        self.labels = []
                        for i in range(1, total_number_of_sessions+1):
                            emg_async = pool.map_async(self.utils.getEMG_separateSessions, [(j+1, i) for j in range(self.utils.num_subjects)])
                            self.emg.extend(emg_async.get())
                            
                            labels_async = pool.map_async(self.utils.getLabels_separateSessions, [(j+1, i) for j in range(self.utils.num_subjects)])
                            self.labels.extend(labels_async.get())
                        
                    else: # Not leave one session out
                        emg_async = pool.map_async(self.utils.getEMG, [(i+1) for i in range(self.utils.num_subjects)])
                        self.emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
                        
                        self.labels_async = pool.map_async(self.utils.getLabels, [(i+1) for i in range(self.utils.num_subjects)])
                        self.labels = self.labels_async.get()

            print("subject 1 mean", torch.mean(self.emg[0]))
            self.numGestures = self.utils.numGestures

        self.length = self.emg[0].shape[1]
        self.width = self.emg[0].shape[2]
        print("Number of Samples (across all participants): ", sum([e.shape[0] for e in self.emg]))
        print("Number of Electrode Channels: ", self.length)
        print("Number of Timesteps per Trial:", self.width)

    def reshape_data(self):
        """
        Modifies data to be normalized, reshaped, and grouped as needed based on whether n_subjects_out_randomly, held_out_test, k_folds, scalar_normalization, target_normalization are selected.
        """

        # These can be tuned to change the normalization
        # This is the coefficient for the standard deviation
        # used for the magnitude images. In practice, these
        # should be fairly small so that images have more
        # contrast
        if self.args.turn_on_rms:
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
        if self.args.leave_n_subjects_out_randomly != 0 and (not self.args.turn_off_scaler_normalization and not self.args.target_normalize):
            self.leaveOut = self.args.leave_n_subjects_out_randomly
            print(f"Leaving out {self.leaveOut} subjects randomly")
            # subject indices to leave out randomly
            leaveOutIndices = np.random.choice(range(self.utils.num_subjects), self.leaveOut, replace=False)
            print(f"Leaving out subjects {np.sort(leaveOutIndices)}")
            emg_in = np.concatenate([np.array(i.view(len(i), self.length*self.width)) for i in self.emg if i not in leaveOutIndices], axis=0, dtype=np.float32)
            
            global_low_value = emg_in.mean() - sigma_coefficient*emg_in.std()
            global_high_value = emg_in.mean() + sigma_coefficient*emg_in.std()
            
            # Normalize by electrode
            emg_in_by_electrode = emg_in.reshape(-1, self.length, self.width)
            
            # Assuming emg is your initial data of shape (SAMPLES, 16, 50)
            # Reshape data to (SAMPLES*50, 16)
            emg_reshaped = emg_in_by_electrode.reshape(-1, self.utils.numElectrodes)
            
            # Initialize and fit the scaler on the reshaped data
            # This will compute the mean and std dev for each electrode across all samples and features
            scaler = preprocessing.StandardScaler()
            scaler.fit(emg_reshaped)
            
            # Repeat means and std_devs for each time point using np.repeat
            scaler.mean_ = np.repeat(scaler.mean_, self.width)
            scaler.scale_ = np.repeat(scaler.scale_, self.width)
            scaler.var_ = np.repeat(scaler.var_, self.width)
            scaler.n_features_in_ = self.width*self.utils.numElectrodes

            del emg_in
            del emg_in_by_electrode
            del emg_reshaped

        else: # Not leave n subjects out randomly
            if (self.args.held_out_test):
                if self.args.turn_on_kfold:
                    skf = StratifiedKFold(n_splits=self.args.kfold, shuffle=True, random_state=self.args.seed)
                    
                    emg_in = np.concatenate([np.array(i.reshape(-1, self.length*self.width)) for i in self.emg], axis=0, dtype=np.float32)
                    labels_in = np.concatenate([np.array(i) for i in self.labels], axis=0, dtype=np.float16)
                    
                    labels_for_folds = np.argmax(labels_in, axis=1)
                    
                    fold_count = 1
                    for train_index, test_index in skf.split(emg_in, labels_for_folds):
                        if fold_count == self.args.fold_index:
                            train_indices = train_index
                            validation_indices = test_index
                            break
                        fold_count += 1

                    # Normalize by electrode
                    emg_in_by_electrode = emg_in[train_indices].reshape(-1, self.length, self.width)
                    # s = preprocessing.StandardScaler().fit(emg_in[train_indices])
                    global_low_value = emg_in[train_indices].mean() - sigma_coefficient*emg_in[train_indices].std()
                    global_high_value = emg_in[train_indices].mean() + sigma_coefficient*emg_in[train_indices].std()

                    # Assuming emg is your initial data of shape (SAMPLES, 16, 50)
                    # Reshape data to (SAMPLES*50, 16)
                    emg_reshaped = emg_in_by_electrode.reshape(-1, self.utils.numElectrodes)

                    # Initialize and fit the scaler on the reshaped data
                    # This will compute the mean and std dev for each electrode across all samples and features
                    scaler = preprocessing.StandardScaler()
                    scaler.fit(emg_reshaped)
                    
                    # Repeat means and std_devs for each time point using np.repeat
                    scaler.mean_ = np.repeat(scaler.mean_, self.width)
                    scaler.scale_ = np.repeat(scaler.scale_, self.width)
                    scaler.var_ = np.repeat(scaler.var_, self.width)
                    scaler.n_features_in_ = self.width*self.utils.numElectrodes

                    del emg_in
                    del labels_in

                    del emg_in_by_electrode
                    del emg_reshaped

                else: 
                    # Reshape and concatenate EMG data
                    # Flatten each subject's data from (TRIAL, CHANNEL, TIME) to (TRIAL, CHANNEL*TIME)
                    # Then concatenate along the subject dimension (axis=0)
                    emg_in = np.concatenate([np.array(i.reshape(-1, self.length*self.width)) for i in self.emg], axis=0, dtype=np.float32)
                    labels_in = np.concatenate([np.array(i) for i in self.labels], axis=0, dtype=np.float16)
                    indices = np.arange(self.emg_in.shape[0])
                    train_indices, validation_indices = model_selection.train_test_split(indices, test_size=0.2, stratify=labels_in)
                    train_emg_in = emg_in[train_indices]  # Select only the train indices
                    # s = preprocessing.StandardScaler().fit(train_emg_in)

                    # Normalize by electrode
                    emg_in_by_electrode = train_emg_in.reshape(-1, self.length, self.width)
                    global_low_value = emg_in[train_indices].mean() - sigma_coefficient*emg_in[train_indices].std()
                    global_high_value = emg_in[train_indices].mean() + sigma_coefficient*emg_in[train_indices].std()

                    # Assuming emg is your initial data of shape (SAMPLES, 16, 50)
                    # Reshape data to (SAMPLES*50, 16)
                    emg_reshaped = emg_in_by_electrode.reshape(-1, self.utils.numElectrodes)

                    # Initialize and fit the scaler on the reshaped data
                    # This will compute the mean and std dev for each electrode across all samples and features
                    scaler = preprocessing.StandardScaler()
                    scaler.fit(emg_reshaped)
                    
                    # Repeat means and std_devs for each time point using np.repeat
                    scaler.mean_ = np.repeat(scaler.mean_, self.width)
                    scaler.scale_ = np.repeat(scaler.scale_, self.width)
                    scaler.var_ = np.repeat(scaler.var_, self.width)
                    scaler.n_features_in_ = self.width*self.utils.numElectrodes

                    del emg_in
                    del labels_in

                    del train_emg_in
                    del indices

                    del emg_in_by_electrode
                    del emg_reshaped

            elif (not self.args.turn_off_scaler_normalization and not self.args.target_normalize): # Running LOSO standardization
                emg_in = np.concatenate([np.array(i.view(len(i), self.length*self.width)) for i in self.emg[:(self.leaveOut-1)]] + [np.array(i.view(len(i), self.length*self.width)) for i in self.emg[self.leaveOut:]], axis=0, dtype=np.float32)
                # s = preprocessing.StandardScaler().fit(emg_in)
                global_low_value = emg_in.mean() - sigma_coefficient*emg_in.std()
                global_high_value = emg_in.mean() + sigma_coefficient*emg_in.std()

                # Normalize by electrode
                emg_in_by_electrode = emg_in.reshape(-1, self.length, self.width)

                # Assuming emg is your initial data of shape (SAMPLES, 16, 50)
                # Reshape data to (SAMPLES*50, 16)
                emg_reshaped = emg_in_by_electrode.reshape(-1, self.utils.numElectrodes)

                # Initialize and fit the scaler on the reshaped data
                # This will compute the mean and std dev for each electrode across all samples and features
                scaler = preprocessing.StandardScaler()
                scaler.fit(emg_reshaped)
                
                # Repeat means and std_devs for each time point using np.repeat
                scaler.mean_ = np.repeat(scaler.mean_, self.width)
                scaler.scale_ = np.repeat(scaler.scale_, self.width)
                scaler.var_ = np.repeat(scaler.var_, self.width)
                scaler.n_features_in_ = self.width*self.utils.numElectrodes

                del emg_in
                del emg_in_by_electrode
                del emg_reshaped

            else: 
                global_low_value = None
                global_high_value = None
                scaler = None

    def rest(self):
            

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
        print("Width of EMG data: ", self.width)
        print("Length of EMG data: ", self.length)

        base_foldername_zarr = ""

        if self.args.leave_n_subjects_out_randomly != 0:
            base_foldername_zarr = f'leave_n_subjects_out_randomly_images_zarr/{self.args.dataset}/leave_{self.args.leave_n_subjects_out_randomly}_subjects_out_randomly_seed-{self.args.seed}/'
        else:
            if self.args.held_out_test:
                base_foldername_zarr = f'heldout_images_zarr/{self.args.dataset}/'
            elif self.args.leave_one_session_out:
                base_foldername_zarr = f'Leave_one_session_out_images_zarr/{self.args.dataset}/'
            elif self.args.turn_off_scaler_normalization:
                base_foldername_zarr = f'LOSOimages_zarr/{self.args.dataset}/'
            elif self.args.leave_one_subject_out:
                base_foldername_zarr = f'LOSOimages_zarr/{self.args.dataset}/'

        if self.args.turn_off_scaler_normalization:
            if self.args.leave_n_subjects_out_randomly != 0:
                base_foldername_zarr = base_foldername_zarr + 'leave_n_subjects_out_randomly_no_scaler_normalization/'
            else: 
                if self.args.held_out_test:
                    base_foldername_zarr = base_foldername_zarr + 'no_scaler_normalization/'
                else: 
                    base_foldername_zarr = base_foldername_zarr + 'LOSO_no_scaler_normalization/'
            scaler = None
        else:
            base_foldername_zarr = base_foldername_zarr + 'LOSO_subject' + str(self.leaveOut) + '/'

        if self.args.turn_on_rms:
            base_foldername_zarr += 'RMS_input_windowsize_' + str(self.args.rms_input_windowsize) + '/'
        elif self.args.turn_on_spectrogram:
            base_foldername_zarr += 'spectrogram/'
        elif self.args.turn_on_cwt:
            base_foldername_zarr += 'cwt/'
        elif self.args.turn_on_hht:
            base_foldername_zarr += 'hht/'

        if self.exercises:
            if self.args.partial_dataset_ninapro:
                base_foldername_zarr += 'partial_dataset_ninapro/'
            else:
                exercises_numbers_filename = '-'.join(map(str, self.args.exercises))
                base_foldername_zarr += f'exercises{exercises_numbers_filename}/'
            
        if self.args.save_images: 
            if not os.path.exists(base_foldername_zarr):
                os.makedirs(base_foldername_zarr)

        for x in tqdm(range(len(self.emg)), desc="Number of Subjects "):
            if self.args.held_out_test:
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
                if self.args.load_few_images:
                    data += [dataset[:10]]
                else: 
                    data += [dataset[:]]
            else:
                print(f"Could not find dataset for subject {x} at {foldername_zarr}")
                # Get images and create the dataset
                if (self.args.target_normalize):
                    scaler = None
                images = self.utils.getImages(self.emg[x], scaler, self.length, self.width, 
                                        turn_on_rms=self.args.turn_on_rms, rms_windows=self.args.rms_input_windowsize, 
                                        turn_on_magnitude=self.args.turn_on_magnitude, global_min=global_low_value, global_max=global_high_value, 
                                        turn_on_spectrogram=self.args.turn_on_spectrogram, turn_on_cwt=self.args.turn_on_cwt, 
                                        turn_on_hht=self.args.turn_on_hht)
                images = np.array(images, dtype=np.float16)
                
                # Save the dataset
                if self.args.save_images:
                    os.makedirs(foldername_zarr, exist_ok=True)
                    dataset = zarr.open(foldername_zarr, mode='w', shape=images.shape, dtype=images.dtype, chunks=True)
                    dataset[:] = images
                    print(f"Saved dataset for subject {x} at {foldername_zarr}")
                else:
                    print(f"Did not save dataset for subject {x} at {foldername_zarr} because save_images is set to False")
                data += [images]

        if self.args.leave_n_subjects_out_randomly != 0:
            
            # Instead of the below code, leave n subjects out randomly to be used as the 
            # validation set and the rest as the training set using leaveOutIndices
            
            X_validation = np.concatenate([np.array(data[i]) for i in range(self.utils.num_subjects) if i in leaveOutIndices], axis=0, dtype=np.float16)
            Y_validation = np.concatenate([np.array(self.labels[i]) for i in range(self.utils.num_subjects) if i in leaveOutIndices], axis=0, dtype=np.float16)
            X_validation = torch.from_numpy(X_validation).to(torch.float16)
            Y_validation = torch.from_numpy(Y_validation).to(torch.float16)
            
            X_train = np.concatenate([np.array(data[i]) for i in range(self.utils.num_subjects) if i not in leaveOutIndices], axis=0, dtype=np.float16)
            Y_train = np.concatenate([np.array(self.labels[i]) for i in range(self.utils.num_subjects) if i not in leaveOutIndices], axis=0, dtype=np.float16)
            X_train = torch.from_numpy(X_train).to(torch.float16)
            Y_train = torch.from_numpy(Y_train).to(torch.float16)
            
            print("Size of X_train:", X_train.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
            print("Size of Y_train:", Y_train.size()) # (SAMPLE, GESTURE)
            print("Size of X_validation:", X_validation.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
            print("Size of Y_validation:", Y_validation.size()) # (SAMPLE, GESTURE)
            
            del data
            del self.emg
            del self.labels

        else: 
            if self.args.held_out_test:
                combined_labels = np.concatenate([np.array(i) for i in self.labels], axis=0, dtype=np.float16)
                combined_images = np.concatenate([np.array(i) for i in data], axis=0, dtype=np.float16)
                X_train = combined_images[train_indices]
                Y_train = combined_labels[train_indices]
                X_validation = combined_images[validation_indices]
                Y_validation = combined_labels[validation_indices]
                X_validation, X_test, Y_validation, Y_test = model_selection.train_test_split(X_validation, Y_validation, test_size=0.5, stratify=Y_validation)
                del combined_images
                del combined_labels
                del data
                del self.emg
                
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
            elif self.args.leave_one_session_out:
                total_number_of_sessions = 2 # all datasets used in our benchmark have at most 2 sessions but this can be changed using a variable from dataset-specific utils instead
                left_out_subject_last_session_index = (total_number_of_sessions - 1) * self.utils.num_subjects + self.leaveOut-1
                left_out_subject_first_n_sessions_indices = [i for i in range(total_number_of_sessions * self.utils.num_subjects) if i % self.utils.num_subjects == (self.leaveOut-1) and i != left_out_subject_last_session_index]
                print("left_out_subject_last_session_index:", left_out_subject_last_session_index)
                print("left_out_subject_first_n_sessions_indices:", left_out_subject_first_n_sessions_indices)

                X_pretrain = []
                Y_pretrain = []
                if self.args.proportion_unlabeled_data_from_training_subjects>0:
                    X_pretrain_unlabeled_list = []
                    Y_pretrain_unlabeled_list = []

                X_finetune = []
                Y_finetune = []
                if self.args.proportion_unlabeled_data_from_leftout_subject>0:
                    X_finetune_unlabeled_list = []
                    Y_finetune_unlabeled_list = []

                for i in range(total_number_of_sessions * self.utils.num_subjects):
                    X_train_temp = data[i]
                    Y_train_temp = self.labels[i]
                    if i != left_out_subject_last_session_index and i not in left_out_subject_first_n_sessions_indices:
                        if self.args.proportion_data_from_training_subjects<1.0:
                            X_train_temp, _, Y_train_temp, _ = tts.train_test_split(
                                X_train_temp, Y_train_temp, train_size=self.args.proportion_data_from_training_subjects, stratify=Y_train_temp, random_state=self.args.seed, shuffle=(not self.args.cross_validation_for_time_series))
                        if self.args.proportion_unlabeled_data_from_training_subjects>0:
                            X_pretrain_labeled, X_pretrain_unlabeled, Y_pretrain_labeled, Y_pretrain_unlabeled = tts.train_test_split(
                                X_train_temp, Y_train_temp, train_size=1-self.args.proportion_unlabeled_data_from_training_subjects, stratify=self.labels[i], random_state=self.args.seed, shuffle=(not self.args.cross_validation_for_time_series))
                            X_pretrain.append(np.array(X_pretrain_labeled))
                            Y_pretrain.append(np.array(Y_pretrain_labeled))
                            X_pretrain_unlabeled_list.append(np.array(X_pretrain_unlabeled))
                            Y_pretrain_unlabeled_list.append(np.array(Y_pretrain_unlabeled))
                        else:
                            X_pretrain.append(np.array(X_train_temp))
                            Y_pretrain.append(np.array(Y_train_temp))
                    elif i in left_out_subject_first_n_sessions_indices:
                        if self.args.proportion_unlabeled_data_from_leftout_subject>0:
                            X_finetune_labeled, X_finetune_unlabeled, Y_finetune_labeled, Y_finetune_unlabeled = tts.train_test_split(
                                X_train_temp, Y_train_temp, train_size=1-self.args.proportion_unlabeled_data_from_leftout_subject, stratify=self.labels[i], random_state=self.args.seed, shuffle=(not self.args.cross_validation_for_time_series))
                            X_finetune.append(np.array(X_finetune_labeled))
                            Y_finetune.append(np.array(Y_finetune_labeled))
                            X_finetune_unlabeled_list.append(np.array(X_finetune_unlabeled))
                            Y_finetune_unlabeled_list.append(np.array(Y_finetune_unlabeled))
                        else:
                            X_finetune.append(np.array(X_train_temp))
                            Y_finetune.append(np.array(Y_train_temp))

                X_pretrain = np.concatenate(X_pretrain, axis=0, dtype=np.float16)
                Y_pretrain = np.concatenate(Y_pretrain, axis=0, dtype=np.float16)
                X_finetune = np.concatenate(X_finetune, axis=0, dtype=np.float16)
                Y_finetune = np.concatenate(Y_finetune, axis=0, dtype=np.float16)
                X_validation = np.array(data[left_out_subject_last_session_index])
                Y_validation = np.array(self.labels[left_out_subject_last_session_index])
                if self.args.proportion_unlabeled_data_from_training_subjects>0:
                    X_pretrain_unlabeled = np.concatenate(X_pretrain_unlabeled_list, axis=0, dtype=np.float16)
                    Y_pretrain_unlabeled = np.concatenate(Y_pretrain_unlabeled_list, axis=0, dtype=np.float16)
                if self.args.proportion_unlabeled_data_from_leftout_subject>0:
                    X_finetune_unlabeled = np.concatenate(X_finetune_unlabeled_list, axis=0, dtype=np.float16)
                    Y_finetune_unlabeled = np.concatenate(Y_finetune_unlabeled_list, axis=0, dtype=np.float16)
                
                X_train = torch.from_numpy(X_pretrain).to(torch.float16)
                Y_train = torch.from_numpy(Y_pretrain).to(torch.float16)
                X_train_finetuning = torch.from_numpy(X_finetune).to(torch.float16)
                Y_train_finetuning = torch.from_numpy(Y_finetune).to(torch.float16)
                X_validation = torch.from_numpy(X_validation).to(torch.float16)
                Y_validation = torch.from_numpy(Y_validation).to(torch.float16)
                if self.args.proportion_unlabeled_data_from_training_subjects>0:
                    X_train_unlabeled = torch.from_numpy(X_pretrain_unlabeled).to(torch.float16)
                    Y_train_unlabeled = torch.from_numpy(Y_pretrain_unlabeled).to(torch.float16)
                if self.args.proportion_unlabeled_data_from_leftout_subject>0:
                    X_train_finetuning_unlabeled = torch.from_numpy(X_finetune_unlabeled).to(torch.float16)
                    Y_train_finetuning_unlabeled = torch.from_numpy(Y_finetune_unlabeled).to(torch.float16)

                del X_finetune
                del Y_finetune
                
                if self.args.turn_on_unlabeled_domain_adaptation: # while in leave one session out
                    proportion_to_keep_of_leftout_subject_for_training = self.args.proportion_transfer_learning_from_leftout_subject
                    proportion_unlabeled_of_proportion_to_keep_of_leftout = self.args.proportion_unlabeled_data_from_leftout_subject
                    proportion_unlabeled_of_training_subjects = self.args.proportion_unlabeled_data_from_training_subjects

                    if self.args.proportion_unlabeled_data_from_training_subjects>0:
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

                    if self.args.proportion_unlabeled_data_from_leftout_subject>0:
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
                        
                    if not self.args.pretrain_and_finetune:
                        X_train = torch.concat((X_train, X_train_finetuning), axis=0)
                        Y_train = torch.concat((Y_train, Y_train_finetuning), axis=0)
                        
                else: 
                    if not self.args.pretrain_and_finetune:
                        X_train = torch.concat((X_train, X_train_finetuning), axis=0)
                        Y_train = torch.concat((Y_train, Y_train_finetuning), axis=0)
                    
                print("Size of X_train:     ", X_train.size())
                print("Size of Y_train:     ", Y_train.size())
                if not self.args.turn_on_unlabeled_domain_adaptation:
                    print("Size of X_train_finetuning:  ", X_train_finetuning.size())
                    print("Size of Y_train_finetuning:  ", Y_train_finetuning.size())
                print("Size of X_validation:", X_validation.size())
                print("Size of Y_validation:", Y_validation.size())
                
                del data
                del self.emg
                del self.labelslabels
                
            elif self.args.leave_one_subject_out: # Running LOSO
                if self.args.reduce_training_data_size:
                    reduced_size_per_subject = self.args.reduced_training_data_size // (self.utils.num_subjects - 1)

                X_validation = np.array(data[self.leaveOut-1])
                Y_validation = np.array(self.labels[self.leaveOut-1])

                X_train_list = []
                Y_train_list = []

                if self.args.proportion_unlabeled_data_from_training_subjects>0:
                    X_train_unlabeled_list = []
                    Y_train_unlabeled_list = []
                
                for i in range(len(data)):
                    if i == self.leaveOut-1:
                        continue
                    current_data = np.array(data[i])
                    current_labels = np.array(self.labels[i])

                    if self.args.reduce_training_data_size:
                        proportion_to_keep = reduced_size_per_subject / current_data.shape[0]
                        current_data, _, current_labels, _ = model_selection.train_test_split(current_data, current_labels, 
                                                                                                train_size=proportion_to_keep, stratify=current_labels, 
                                                                                                random_state=self.args.seed, shuffle=(not self.args.cross_validation_for_time_series))
                        
                    if self.args.proportion_data_from_training_subjects<1.0:
                        current_data, _, current_labels, _ = tts.train_test_split(
                            current_data, current_labels, train_size=self.args.proportion_data_from_training_subjects, stratify=current_labels, random_state=self.args.seed, shuffle=(not self.args.cross_validation_for_time_series))
                        
                    if self.args.proportion_unlabeled_data_from_training_subjects>0:
                        X_train_labeled, X_train_unlabeled, Y_train_labeled, Y_train_unlabeled = tts.train_test_split(
                            current_data, current_labels, train_size=1-self.args.proportion_unlabeled_data_from_training_subjects, stratify=current_labels, random_state=self.args.seed, shuffle=(not self.args.cross_validation_for_time_series))
                        current_data = X_train_labeled
                        current_labels = Y_train_labeled

                        X_train_unlabeled_list.append(X_train_unlabeled)
                        Y_train_unlabeled_list.append(Y_train_unlabeled)

                    X_train_list.append(current_data)
                    Y_train_list.append(current_labels)
                    
                X_train = torch.from_numpy(np.concatenate(X_train_list, axis=0)).to(torch.float16)
                Y_train = torch.from_numpy(np.concatenate(Y_train_list, axis=0)).to(torch.float16)
                if self.args.proportion_unlabeled_data_from_training_subjects>0:
                    X_train_unlabeled = torch.from_numpy(np.concatenate(X_train_unlabeled_list, axis=0)).to(torch.float16)
                    Y_train_unlabeled = torch.from_numpy(np.concatenate(Y_train_unlabeled_list, axis=0)).to(torch.float16)
                X_validation = torch.from_numpy(X_validation).to(torch.float16)
                Y_validation = torch.from_numpy(Y_validation).to(torch.float16)

                if self.args.transfer_learning: # while in leave one subject out
                    proportion_to_keep_of_leftout_subject_for_training = self.args.proportion_transfer_learning_from_leftout_subject
                    proportion_unlabeled_of_proportion_to_keep_of_leftout = self.args.proportion_unlabeled_data_from_leftout_subject
                    proportion_unlabeled_of_training_subjects = self.args.proportion_unlabeled_data_from_training_subjects
                    
                    if proportion_to_keep_of_leftout_subject_for_training>0.0:
                        if self.args.cross_validation_for_time_series:
                            X_train_partial_leftout_subject, X_validation_partial_leftout_subject, Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject = tts.train_test_split(
                                X_validation, Y_validation, train_size=proportion_to_keep_of_leftout_subject_for_training, stratify=Y_validation, random_state=self.args.seed, shuffle=False)
                        else:
                            # Split the validation data into train and validation subsets
                            X_train_partial_leftout_subject, X_validation_partial_leftout_subject, Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject = tts.train_test_split(
                                X_validation, Y_validation, train_size=proportion_to_keep_of_leftout_subject_for_training, stratify=Y_validation, random_state=self.args.seed, shuffle=True)
                    else:
                        X_validation_partial_leftout_subject = X_validation
                        Y_validation_partial_leftout_subject = Y_validation
                        X_train_partial_leftout_subject = torch.tensor([])
                        Y_train_partial_leftout_subject = torch.tensor([])
                        
                    if self.args.turn_on_unlabeled_domain_adaptation and proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
                        if self.args.cross_validation_for_time_series:
                            X_train_labeled_partial_leftout_subject, X_train_unlabeled_partial_leftout_subject, \
                            Y_train_labeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject = tts.train_test_split(
                                X_train_partial_leftout_subject, Y_train_partial_leftout_subject, train_size=1-proportion_unlabeled_of_proportion_to_keep_of_leftout, stratify=Y_train_partial_leftout_subject, random_state=self.args.seed, shuffle=False)
                        else:
                            X_train_labeled_partial_leftout_subject, X_train_unlabeled_partial_leftout_subject, \
                            Y_train_labeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject = tts.train_test_split(
                                X_train_partial_leftout_subject, Y_train_partial_leftout_subject, train_size=1-proportion_unlabeled_of_proportion_to_keep_of_leftout, stratify=Y_train_partial_leftout_subject, random_state=self.args.seed, shuffle=True)
                    
                    # if args.turn_on_unlabeled_domain_adaptation and proportion_unlabeled_of_training_subjects>0: #DELETE
                    #     if args.cross_validation_for_time_series:
                    #         X_train_labeled, X_train_unlabeled, Y_train_labeled, Y_train_unlabeled = tts.train_test_split(
                    #             X_train, Y_train, train_size=1-proportion_unlabeled_of_training_subjects, stratify=Y_train, random_state=args.seed, shuffle=False)
                    #     else: 
                    #         X_train_labeled, X_train_unlabeled, Y_train_labeled, Y_train_unlabeled = tts.train_test_split(
                    #             X_train, Y_train, train_size=1-proportion_unlabeled_of_training_subjects, stratify=Y_train, random_state=args.seed, shuffle=True)

                    print("Size of X_train_partial_leftout_subject:     ", X_train_partial_leftout_subject.shape) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
                    print("Size of Y_train_partial_leftout_subject:     ", Y_train_partial_leftout_subject.shape) # (SAMPLE, GESTURE)

                    if not self.args.turn_on_unlabeled_domain_adaptation:
                        # Append the partial validation data to the training data
                        if proportion_to_keep_of_leftout_subject_for_training>0:
                            if not self.args.pretrain_and_finetune:
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

                        if proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
                            if not self.args.pretrain_and_finetune:
                                X_train = torch.tensor(np.concatenate((X_train, X_train_labeled_partial_leftout_subject), axis=0))
                                Y_train = torch.tensor(np.concatenate((Y_train, Y_train_labeled_partial_leftout_subject), axis=0))
                                X_train_finetuning_unlabeled = torch.tensor(X_train_unlabeled_partial_leftout_subject)
                                Y_train_finetuning_unlabeled = torch.tensor(Y_train_unlabeled_partial_leftout_subject)
                            else:
                                X_train_finetuning = torch.tensor(X_train_labeled_partial_leftout_subject)
                                Y_train_finetuning = torch.tensor(Y_train_labeled_partial_leftout_subject)
                                X_train_finetuning_unlabeled = torch.tensor(X_train_unlabeled_partial_leftout_subject)
                                Y_train_finetuning_unlabeled = torch.tensor(Y_train_unlabeled_partial_leftout_subject)
                                
                        else:
                            if proportion_to_keep_of_leftout_subject_for_training>0:
                                if not self.args.pretrain_and_finetune:
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
                
                if self.args.turn_on_unlabeled_domain_adaptation and proportion_unlabeled_of_training_subjects>0:
                    print("Size of X_train_unlabeled:     ", X_train_unlabeled.shape)
                    print("Size of Y_train_unlabeled:     ", Y_train_unlabeled.shape)
                if self.args.turn_on_unlabeled_domain_adaptation and proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
                    print("Size of X_train_finetuning_unlabeled:     ", X_train_finetuning_unlabeled.shape)
                    print("Size of Y_train_finetuning_unlabeled:     ", Y_train_finetuning_unlabeled.shape)
                    
                if self.args.pretrain_and_finetune:
                    print("Size of X_train_finetuning:     ", X_train_finetuning.shape)
                    print("Size of Y_train_finetuning:     ", Y_train_finetuning.shape)
                    
            else: 
                ValueError("Please specify the type of test you want to run")

        model_name = self.args.model

        if self.args.model == "vit_tiny_patch2_32":
            pretrain_path = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth"
        elif self.args.model == "resnet50":
            pretrain_path = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
        else:
            pretrain_path = f"https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/{model_name}_mlp_im_1k_224.pth"

        def ceildiv(a, b):
                return -(a // -b)
            
        if self.args.turn_on_unlabeled_domain_adaptation:
            print("Number of total batches in training data:", X_train.shape[0] // self.args.batch_size)
            current_date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            assert (self.args.transfer_learning and self.args.cross_validation_for_time_series) or self.args.leave_one_session_out, \
                "Unlabeled Domain Adaptation requires transfer learning and cross validation for time series or leave one session out"
            
            semilearn_config_dict = {
                'algorithm': self.args.unlabeled_algorithm,
                'net': self.args.model,
                'use_pretrain': True,  
                'pretrain_path': pretrain_path,
                'seed': self.args.seed,

                # optimization configs
                'epoch': self.args.epochs,  # set to 100
                'num_train_iter': self.args.epochs * ceildiv(X_train.shape[0], self.args.batch_size),
                'num_eval_iter': ceildiv(X_train.shape[0], self.args.batch_size),
                'num_log_iter': ceildiv(X_train.shape[0], self.args.batch_size),
                'optim': 'AdamW',   # AdamW optimizer
                'lr': self.args.learning_rate,  # Learning rate
                'layer_decay': 0.5,  # Layer-wise decay learning rate  
                'momentum': 0.9,  # Momentum
                'weight_decay': 0.0005,  # Weight decay
                'amp': True,  # Automatic mixed precision
                'train_sampler': 'RandomSampler',  # Random sampler
                'rank': 0,  # Rank
                'batch_size': self.args.batch_size,  # Batch size
                'eval_batch_size': self.args.batch_size, # Evaluation batch size
                'use_wandb': True,
                'ema_m': 0.999,
                'save_dir': './saved_models/unlabeled_domain_adaptation/',
                'save_name': f'{self.args.unlabeled_algorithm}_{self.args.model}_{self.args.dataset}_seed_{self.args.seed}_leave_{self.leaveOut}_unlabeled_domain_adaptation_{current_date_and_time}',
                'resume': True,
                'overwrite': True,
                'load_path': f'./saved_models/unlabeled_domain_adaptation/{self.args.unlabeled_algorithm}_{self.args.model}_{self.args.dataset}_seed_{self.args.seed}_leave_{self.leaveOut}_unlabeled_domain_adaptation_{current_date_and_time}/latest_model.pth',
                'scheduler': None,

                # dataset configs
                'dataset': 'none',
                'num_labels': X_train.shape[0],
                'num_classes': self.numGestures,
                'input_size': 224,
                'data_dir': './data',

                # algorithm specific configs
                'hard_label': True,
                'uratio': 1.5,
                'ulb_loss_ratio': 1.0,

                # device configs
                'gpu': self.args.gpu,
                'world_size': 1,
                'distributed': False,
            } 
            
            semilearn_config = get_config(semilearn_config_dict)
            semilearn_algorithm = get_algorithm(semilearn_config, get_net_builder(semilearn_config.net, from_name=False), tb_log=None, logger=None)
            semilearn_algorithm.model = send_model_cuda(semilearn_config, semilearn_algorithm.model)
            semilearn_algorithm.ema_model = send_model_cuda(semilearn_config, semilearn_algorithm.ema_model, clip_batch=False)

            if self.args.model == 'vit_tiny_patch2_32':
                semilearn_transform = transforms.Compose([transforms.Resize((32,32)), ToNumpy()])
            else: 
                semilearn_transform = transforms.Compose([transforms.Resize((224,224)), ToNumpy()])
            
            labeled_dataset = BasicDataset(semilearn_config, X_train, torch.argmax(Y_train, dim=1), semilearn_config.num_classes, semilearn_transform, is_ulb=False)
            if proportion_unlabeled_of_training_subjects>0:
                unlabeled_dataset = BasicDataset(semilearn_config, X_train_unlabeled, torch.argmax(Y_train_unlabeled, dim=1), semilearn_config.num_classes, semilearn_transform, 
                                                is_ulb=True, strong_transform=semilearn_transform)
                proportion_unlabeled_to_use = self.args.proportion_unlabeled_data_from_training_subjects
            elif proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
                unlabeled_dataset = BasicDataset(semilearn_config, X_train_finetuning_unlabeled, torch.argmax(Y_train_finetuning_unlabeled, dim=1), semilearn_config.num_classes, semilearn_transform, 
                                                is_ulb=True, strong_transform=semilearn_transform)
                proportion_unlabeled_to_use = self.args.proportion_unlabeled_data_from_leftout_subject
            if self.args.pretrain_and_finetune:
                finetune_dataset = BasicDataset(semilearn_config, X_train_finetuning, torch.argmax(Y_train_finetuning, dim=1), semilearn_config.num_classes, semilearn_transform, is_ulb=False)
                finetune_unlabeled_dataset = BasicDataset(semilearn_config, X_train_finetuning_unlabeled, torch.argmax(Y_train_finetuning_unlabeled, dim=1), 
                                                        semilearn_config.num_classes, semilearn_transform, is_ulb=True, strong_transform=semilearn_transform)
            validation_dataset = BasicDataset(semilearn_config, X_validation, torch.argmax(Y_validation, dim=1), semilearn_config.num_classes, semilearn_transform, is_ulb=False)

            labeled_batch_size = int(semilearn_config.batch_size * (1-proportion_unlabeled_to_use))
            unlabeled_batch_size = int(semilearn_config.batch_size * proportion_unlabeled_to_use)
            labeled_iters = len(labeled_dataset) * ceildiv(self.args.epochs, labeled_batch_size)
            unlabeled_iters = len(unlabeled_dataset) * ceildiv(self.args.epochs, unlabeled_batch_size)
            iters_for_loader = max(labeled_iters, unlabeled_iters)
            train_labeled_loader = get_data_loader(semilearn_config, labeled_dataset, labeled_batch_size, num_workers=multiprocessing.cpu_count()//8, 
                                                num_epochs=self.args.epochs, num_iters=iters_for_loader)
            if proportion_unlabeled_of_training_subjects>0 or proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
                train_unlabeled_loader = get_data_loader(semilearn_config, unlabeled_dataset, unlabeled_batch_size, num_workers=multiprocessing.cpu_count()//8,
                                                        num_epochs=self.args.epochs, num_iters=iters_for_loader)
                
            if self.args.pretrain_and_finetune:
                if self.args.proportion_unlabeled_data_from_leftout_subject>0:
                    proportion_unlabeled_to_use = self.args.proportion_unlabeled_data_from_leftout_subject
                elif self.args.proportion_unlabeled_data_from_training_subjects>0:
                    proportion_unlabeled_to_use = self.args.proportion_unlabeled_data_from_training_subjects
                labeled_batch_size = int(semilearn_config.batch_size * (1-proportion_unlabeled_to_use))
                unlabeled_batch_size = int(semilearn_config.batch_size * proportion_unlabeled_to_use)
                labeled_iters = len(finetune_dataset) * ceildiv(self.args.epochs, labeled_batch_size)
                unlabeled_iters = len(finetune_unlabeled_dataset) * ceildiv(self.args.epochs, unlabeled_batch_size)
                iters_for_loader = max(labeled_iters, unlabeled_iters)
                train_finetuning_loader = get_data_loader(semilearn_config, finetune_dataset, labeled_batch_size, num_workers=multiprocessing.cpu_count()//8,
                                                        num_epochs=self.args.epochs, num_iters=iters_for_loader)
                train_finetuning_unlabeled_loader = get_data_loader(semilearn_config, finetune_unlabeled_dataset, unlabeled_batch_size, num_workers=multiprocessing.cpu_count()//8,
                                                                    num_epochs=self.args.epochs, num_iters=iters_for_loader)
            validation_loader = get_data_loader(semilearn_config, validation_dataset, semilearn_config.eval_batch_size, num_workers=multiprocessing.cpu_count()//8)

        else:
            if self.args.model == 'resnet50_custom':
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
                model.add_module('fc3', nn.Linear(512, self.numGestures))
                model.add_module('softmax', nn.Softmax(dim=1))
            elif self.args.model == 'resnet50':
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
                # Replace the last fully connected layer
                num_ftrs = model.fc.in_features  # Get the number of input features of the original fc layer
                model.fc = nn.Linear(num_ftrs, self.numGestures)  # Replace with a new linear layer
            elif self.args.model == 'convnext_tiny_custom':
                # %% Referencing: https://medium.com/exemplifyml-ai/image-classification-with-resnet-convnext-using-pytorch-f051d0d7e098
                class LayerNorm2d(nn.LayerNorm):
                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        x = x.permute(0, 2, 3, 1)
                        x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
                        x = x.permute(0, 3, 1, 2)
                        return x

                n_inputs = 768
                hidden_size = 128 # default is 2048
                n_outputs = self.numGestures

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
            elif self.args.model == 'vit_tiny_patch2_32':
                pretrain_path = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth"
                model = VisualTransformer.vit_tiny_patch2_32(pretrained=True, pretrained_path=pretrain_path, num_classes=self.numGestures)
            elif self.args.model == 'MLP' or self.args.model == 'SVC' or self.args.model == 'RF':
                model = None # Will be initialized later
            else: 
                # model_name = 'efficientnet_b0'  # or 'efficientnet_b1', ..., 'efficientnet_b7'
                # model_name = 'tf_efficientnet_b3.ns_jft_in1k'
                model = timm.create_model(model_name, pretrained=True, num_classes=self.numGestures)
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

        if not self.args.turn_on_unlabeled_domain_adaptation:
            if self.args.model not in ['MLP', 'SVC', 'RF']:
                num = 0
                for name, param in model.named_parameters():
                    num += 1
                    if (num > 0):
                    #if (num > 72): # for -3
                    #if (num > 33): # for -4
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            batch_size = self.args.batch_size

            if self.args.model == 'vit_tiny_patch2_32':
                resize_transform = transforms.Compose([transforms.Resize((32,32)), ToNumpy()])
            else:
                resize_transform = transforms.Compose([transforms.Resize((224,224)), ToNumpy()])

            train_dataset = CustomDataset(X_train, Y_train, transform=resize_transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=self.utils.seed_worker, pin_memory=True)
            val_dataset = CustomDataset(X_validation, Y_validation, transform=resize_transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=self.utils.seed_worker, pin_memory=True)
            if (self.args.held_out_test):
                test_dataset = CustomDataset(X_test, Y_test, transform=resize_transform)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=self.utils.seed_worker, pin_memory=True)

            # Define the loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            learn = self.args.learning_rate
            if self.args.model not in ['MLP', 'SVC', 'RF']:
                optimizer = torch.optim.Adam(model.parameters(), lr=learn)

            num_epochs = self.args.epochs
            if self.args.turn_on_cosine_annealing:
                number_cycles = 5
                annealing_multiplier = 2
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.utils.periodLengthForAnnealing(num_epochs, annealing_multiplier, number_cycles),
                                                                                    T_mult=annealing_multiplier, eta_min=1e-5, last_epoch=-1)
            elif self.args.turn_on_cyclical_lr:
                # Define the cyclical learning rate scheduler
                step_size = len(train_loader) * 6  # Number of iterations in half a cycle
                base_lr = 1e-4  # Minimum learning rate
                max_lr = 1e-3  # Maximum learning rate
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=step_size, mode='triangular2', cycle_momentum=False)

            # Training loop
            gc.collect()
            torch.cuda.empty_cache()

        wandb_runname = 'CNN_seed-'+str(self.args.seed)
        if self.args.turn_on_kfold:
            wandb_runname += '_k-fold-'+str(self.args.kfold)+'_fold-index-'+str(self.args.fold_index)
        if self.args.turn_on_cyclical_lr:
            wandb_runname += '_cyclical-lr'
        if self.args.turn_on_cosine_annealing: 
            wandb_runname += '_cosine-annealing'
        if self.args.turn_on_rms:
            wandb_runname += '_rms-'+str(self.args.rms_input_windowsize)
        if self.args.turn_on_magnitude:  
            wandb_runname += '_mag-'
        if self.args.leftout_subject != 0:
            wandb_runname += '_LOSO-'+str(self.args.leftout_subject)
        wandb_runname += '_' + model_name
        if (self.exercises and not self.args.partial_dataset_ninapro):
            wandb_runname += '_exer-' + ''.join(character for character in str(self.args.exercises) if character.isalnum())
        if self.args.dataset == "OzdemirEMG":
            if self.args.full_dataset_ozdemir:
                wandb_runname += '_full'
            else:
                wandb_runname += '_partial'
        if self.args.dataset == "ninapro-db2" or self.args.dataset == "ninapro-db5":
            if self.args.partial_dataset_ninapro:
                wandb_runname += '_partial'
        if self.args.turn_on_spectrogram:
            wandb_runname += '_spect'
        if self.args.turn_on_cwt:
            wandb_runname += '_cwt'
        if self.args.turn_on_hht:
            wandb_runname += '_hht'
        if self.args.reduce_training_data_size:
            wandb_runname += '_reduced-training-data-size-' + str(self.args.reduced_training_data_size)
        if self.args.leave_n_subjects_out_randomly != 0:
            wandb_runname += '_leave_n_subjects_out-'+str(self.args.leave_n_subjects_out_randomly)
        if self.args.turn_off_scaler_normalization:
            wandb_runname += '_no-scal-norm'
        if self.args.target_normalize:
            wandb_runname += '_targ-norm'
        if self.args.load_few_images:
            wandb_runname += '_load-few'
        if self.args.transfer_learning:
            wandb_runname += '_tran-learn'
            wandb_runname += '-prop-' + str(self.args.proportion_transfer_learning_from_leftout_subject)
        if self.args.cross_validation_for_time_series:   
            wandb_runname += '_cv-for-ts'
        if self.args.reduce_data_for_transfer_learning != 1:
            wandb_runname += '_red-data-for-tran-learn-' + str(self.args.reduce_data_for_transfer_learning)
        if self.args.leave_one_session_out:
            wandb_runname += '_leave-one-sess-out'
        if self.args.leave_one_subject_out:
            wandb_runname += '_loso'
        if self.args.one_subject_for_training_set_for_session_test:
            wandb_runname += '_one-subj-for-training-set'
        if self.args.held_out_test:
            wandb_runname += '_held-out'
        if self.args.pretrain_and_finetune:
            wandb_runname += '_pretrain-finetune'
        if self.args.turn_on_unlabeled_domain_adaptation:
            wandb_runname += '_unlabeled-adapt'
            wandb_runname += '-algo-' + self.args.unlabeled_algorithm
            wandb_runname += '-prop-unlabel-leftout' + str(self.args.proportion_unlabeled_data_from_leftout_subject)
        if self.args.proportion_data_from_training_subjects<1.0:
            wandb_runname += '_train-subj-prop-' + str(self.args.proportion_data_from_training_subjects)
        if self.args.proportion_unlabeled_data_from_training_subjects>0:
            wandb_runname += '_unlabel-subj-prop-' + str(self.args.proportion_unlabeled_data_from_training_subjects)

        if (self.args.held_out_test):
            if self.args.turn_on_kfold:
                self.project_name += '_k-fold-'+str(self.args.kfold)
            else:
                self.project_name += '_heldout'
        elif self.args.leave_one_subject_out:
            self.project_name += '_LOSO'
        elif self.args.leave_one_session_out:
            self.project_name += '_leave-one-session-out'
            
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

        self.project_name += self.args.project_name_suffix

        run = wandb.init(name=wandb_runname, project=self.project_name, entity='jehanyang')
        wandb.config.lr = self.args.learning_rate

        if self.args.leave_n_subjects_out_randomly != 0:
            wandb.config.left_out_subjects = leaveOutIndices

        device = torch.device("cuda:" + str(self.args.gpu) if torch.cuda.is_available() else "cpu")
        print("Device:", device)
        if not self.args.turn_on_unlabeled_domain_adaptation and self.args.model not in ['MLP', 'SVC', 'RF']:
            model.to(device)

            wandb.watch(model)

        testrun_foldername = f'test/{self.project_name}/{wandb_runname}/{self.formatted_datetime}/'
        # Make folder if it doesn't exist
        if not os.path.exists(testrun_foldername):
            os.makedirs(testrun_foldername)
        model_filename = f'{testrun_foldername}model_{self.formatted_datetime}.pth'

        if (self.exercises):
            if not self.args.partial_dataset_ninapro:
                gesture_labels = self.argsutils.gesture_labels['Rest']
                for exercise_set in self.args.exercises:
                    gesture_labels = gesture_labels + self.utils.gesture_labels[exercise_set]
            else:
                gesture_labels = self.utils.partial_gesture_labels
        else:
            gesture_labels = self.utils.gesture_labels
            
        # if X_train, validation, test or Y_train validation, test are numpy arrays, convert them to tensors
        X_train = torch.from_numpy(X_train).to(torch.float16) if isinstance(X_train, np.ndarray) else X_train
        Y_train = torch.from_numpy(Y_train).to(torch.float16) if isinstance(Y_train, np.ndarray) else Y_train
        X_validation = torch.from_numpy(X_validation).to(torch.float16) if isinstance(X_validation, np.ndarray) else X_validation
        Y_validation = torch.from_numpy(Y_validation).to(torch.float16) if isinstance(Y_validation, np.ndarray) else Y_validation
        if self.args.held_out_test:
            X_test = torch.from_numpy(X_test).to(torch.float16) if isinstance(X_test, np.ndarray) else X_test
            Y_test = torch.from_numpy(Y_test).to(torch.float16) if isinstance(Y_test, np.ndarray) else Y_test

        if self.args.held_out_test:
            # Plot and log images
            self.utils.plot_average_images(X_test, np.argmax(Y_test.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, self.args, self.formatted_datetime, 'test')
            self.utils.plot_first_fifteen_images(X_test, np.argmax(Y_test.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, self.args, self.formatted_datetime, 'test')

        self.utils.plot_average_images(X_validation, np.argmax(Y_validation.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, self.args, self.formatted_datetime, 'validation')
        self.utils.plot_first_fifteen_images(X_validation, np.argmax(Y_validation.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, self.args, self.formatted_datetime, 'validation')

        self.utils.plot_average_images(X_train, np.argmax(Y_train.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, self.args, self.formatted_datetime, 'train')
        self.utils.plot_first_fifteen_images(X_train, np.argmax(Y_train.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, self.args, self.formatted_datetime, 'train')

        if self.args.pretrain_and_finetune:
            self.utils.plot_average_images(X_train_finetuning, np.argmax(Y_train_finetuning.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, self.args, self.formatted_datetime, 'train_finetuning')
            self.utils.plot_first_fifteen_images(X_train_finetuning, np.argmax(Y_train_finetuning.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, self.args, self.formatted_datetime, 'train_finetuning')

        if self.args.turn_on_unlabeled_domain_adaptation:
            semilearn_algorithm.loader_dict = {}
            semilearn_algorithm.loader_dict['train_lb'] = train_labeled_loader
            if proportion_unlabeled_of_training_subjects>0 or proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
                semilearn_algorithm.loader_dict['train_ulb'] = train_unlabeled_loader
            semilearn_algorithm.loader_dict['eval'] = validation_loader
            semilearn_algorithm.scheduler = None
            
            semilearn_algorithm.train()
            
            if self.args.pretrain_and_finetune:
                run = wandb.init(name=wandb_runname, project=self.project_name, entity='jehanyang')
                wandb.config.lr = self.args.learning_rate
                
                semilearn_config_dict['num_train_iter'] = semilearn_config_dict['num_train_iter'] + self.args.finetuning_epochs * ceildiv(X_train_finetuning.shape[0], self.args.batch_size)
                semilearn_config_dict['num_eval_iter'] = ceildiv(X_train_finetuning.shape[0], self.args.batch_size)
                semilearn_config_dict['num_log_iter'] = ceildiv(X_train_finetuning.shape[0], self.args.batch_size)
                semilearn_config_dict['epoch'] = self.args.finetuning_epochs + self.args.epochs
                semilearn_config_dict['algorithm'] = self.args.unlabeled_algorithm
                
                semilearn_config = get_config(semilearn_config_dict)
                semilearn_algorithm = get_algorithm(semilearn_config, get_net_builder(semilearn_config.net, from_name=False), tb_log=None, logger=None)
                semilearn_algorithm.epochs = self.args.epochs + self.args.finetuning_epochs # train for the same number of epochs as the previous training
                semilearn_algorithm.model = send_model_cuda(semilearn_config, semilearn_algorithm.model)
                semilearn_algorithm.load_model(semilearn_config.load_path)
                semilearn_algorithm.ema_model = send_model_cuda(semilearn_config, semilearn_algorithm.ema_model, clip_batch=False)
                semilearn_algorithm.loader_dict = {}
                semilearn_algorithm.loader_dict['train_lb'] = train_finetuning_loader
                semilearn_algorithm.scheduler = None
                
                if proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
                    semilearn_algorithm.loader_dict['train_ulb'] = train_finetuning_unlabeled_loader
                elif proportion_unlabeled_of_training_subjects>0:
                    semilearn_algorithm.loader_dict['train_ulb'] = train_unlabeled_loader

                semilearn_algorithm.loader_dict['eval'] = validation_loader
                semilearn_algorithm.train()

        else: 
            if self.args.model in ['MLP', 'SVC', 'RF']:
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
                
                if self.args.model == 'MLP':
                    # PyTorch MLP model
                    input_size = 3 * 224 * 224  # Change according to your input size
                    hidden_sizes = [512, 256]  # Example hidden layer sizes
                    output_size = 10  # Number of classes
                    model = MLP(input_size, hidden_sizes, output_size).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    criterion = nn.CrossEntropyLoss()
                
                elif self.args.model == 'SVC':
                    model = SVC(probability=True)
                
                elif self.args.model == 'RF':
                    model = RandomForestClassifier()
                    
                if self.args.model == 'MLP':
                    # PyTorch training loop for MLP
                    for epoch in tqdm(range(num_epochs), desc="Epoch"):
                        model.train()
                        train_acc = 0.0
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
                                preds = torch.argmax(output, dim=1)
                                train_acc += torch.mean((preds == Y_batch).type(torch.float)).item()

                                if t.n % 10 == 0:
                                    t.set_postfix({"Batch Loss": loss.item(), "Batch Acc": train_acc / (t.n + 1)})

                                del X_batch, Y_batch, output, preds
                                torch.cuda.empty_cache()

                        # Validation
                        model.eval()
                        val_loss = 0.0
                        val_acc = 0.0
                        with torch.no_grad():
                            for X_batch, Y_batch in val_loader:
                                X_batch = X_batch.view(X_batch.size(0), -1).to(device).to(torch.float32)
                                Y_batch = torch.argmax(Y_batch, dim=1).to(device).to(torch.int64)

                                output = model(X_batch)
                                val_loss += criterion(output, Y_batch).item()
                                preds = torch.argmax(output, dim=1)
                                val_acc += torch.mean((preds == Y_batch).type(torch.float)).item()

                                del X_batch, Y_batch
                                torch.cuda.empty_cache()

                        train_loss /= len(train_loader)
                        train_acc /= len(train_loader)
                        val_loss /= len(val_loader)
                        val_acc /= len(val_loader)

                        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                        print(f"Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")
                        wandb.log({
                            "Epoch": epoch,
                            "Train Loss": train_loss,
                            "Train Acc": train_acc,
                            "Valid Loss": val_loss,
                            "Valid Acc": val_acc,
                            "Learning Rate": optimizer.param_groups[0]['lr']
                        })

                    torch.save(model.state_dict(), model_filename)
                    wandb.save(f'model/modelParameters_{self.formatted_datetime}.pth')

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
                            if isinstance(output, dict):
                                output = output['logits']
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
                            if isinstance(output, dict):
                                output = output['logits']
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
                wandb.save(f'model/modelParameters_{self.formatted_datetime}.pth')

                if self.args.pretrain_and_finetune:
                    num_epochs = self.args.finetuning_epochs
                    # train more on fine tuning dataset
                    finetune_dataset = CustomDataset(X_train_finetuning, Y_train_finetuning, transform=resize_transform)
                    finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)
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
                                if isinstance(output, dict):
                                    output = output['logits']
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
                                if isinstance(output, dict):
                                    output = output['logits']
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
                if (self.args.held_out_test):
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
                            if isinstance(output, dict):
                                output = output['logits']
                            test_loss += criterion(output, Y_batch).item()

                            test_acc += np.mean(np.argmax(output.cpu().detach().numpy(), axis=1) == np.argmax(Y_batch.cpu().detach().numpy(), axis=1))

                            output = np.argmax(output.cpu().detach().numpy(), axis=1)
                            pred.extend(output)
                            self.add_lableslabels = np.argmax(Y_batch.cpu().detach().numpy(), axis=1)
                            true.extend(self.labels)

                    test_loss /= len(test_loader)
                    test_acc /= len(test_loader)
                    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
                    
                    wandb.log({
                        "Test Loss": test_loss,
                        "Test Acc": test_acc}) 
                    
                    
                    # %% Confusion Matrix
                    # Plot and log confusion matrix in wandb
                    self.utils.plot_confusion_matrix(true, pred, gesture_labels, testrun_foldername, self.args, self.formatted_datetime, 'test')

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

                self.utils.plot_confusion_matrix(np.argmax(Y_validation.cpu().detach().numpy(), axis=1), np.array(validation_predictions), gesture_labels, testrun_foldername, self.args, self.formatted_datetime, 'validation')   

                # Load training in smaller batches for memory purposes
                torch.cuda.empty_cache()  # Clear cache if needed

                model.eval()
                with torch.no_grad():
                    train_predictions = []
                    for X_batch, Y_batch in tqdm(train_loader, desc="Training Batch Loading"):
                        X_batch = X_batch.to(device).to(torch.float32)
                        outputs = model(X_batch)
                        if isinstance(outputs, dict):
                                outputs = outputs['logits']
                        preds = torch.argmax(outputs, dim=1)
                        train_predictions.extend(preds.cpu().detach().numpy())

                self.utils.plot_confusion_matrix(np.argmax(Y_train.cpu().detach().numpy(), axis=1), np.array(train_predictions), gesture_labels, testrun_foldername, self.args, self.formatted_datetime, 'train')
                    
            run.finish()
            
            

        
a = my_class()
print("args:", a.args) 
a.run()






