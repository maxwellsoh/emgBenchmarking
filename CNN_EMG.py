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
import utils_MCS_EMG as utils
from sklearn.model_selection import StratifiedKFold
import os

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import timm
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import zarr
from Split_Strategies.cross_validation_utilities import train_test_split as tts # custom train test split to split stratified without shuffling
import gc
import datetime
from PIL import Image
from torch.utils.data import Dataset
#import VisualTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix, classification_report 
import torch.nn.functional as F
import subprocess
import get_datasets
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer, split_ssl_data, BasicDataset
from semilearn.core.utils import send_model_cuda
import torchmetrics
import ml_metrics_utils as ml_utils
from sklearn.metrics import confusion_matrix, classification_report
import VisualTransformer

from Hook_Manager import Hook_Manager

# Imports for Setup_Run
from Setup.Parse_Arguments import Parse_Arguments

# Imports for Data_Initializer
from Data.X_Data import X_Data
from Data.Y_Data import Y_Data
from Data.Label_Data import Label_Data
from Data.Combined_Data import Combined_Data

# Imports for Data_Splitter
from importlib import import_module

# Imports for Run_Model
from Model.CNN_Trainer import CNN_Trainer
from Model.Unlabeled_Domain_Adaptation_Trainer import Unlabeled_Domain_Adaptation_Trainer
from Model.MLP_Trainer import MLP_Trainer
from Model.SVC_RF_Trainer import SVC_RF_Trainer



class Run_Setup():
    """
    Sets up the run by reading in arguments, setting the dataset source, conducting safety checks, printing values and setting up env.

    Returns:
        env: Setup object which contains necessary information for the run 
    """
   

    def __init__(self):
        pass

    def set_seeds_for_reproducibility(self, env):
        """ Set seeds for reproducibility. 
        """

        seed = env.args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_run(self):

        run = Parse_Arguments() # TODO: case if config
    
        run.setup_args()
        run.setup_for_dataset()
        run.set_exercise()
        run.print_params()
        
        env = run.set_env()
        self.set_seeds_for_reproducibility(env)

        return env


class Data_Initializer():
    """
    Loads in the data for the run. (EMG/labels/forces)
    """

    def __init__(self, env):
        self.args = env.args
        self.utils = env.utils
        self.exercises = env.exercises
        self.leaveOut = env.leaveOut
        self.env = env
        self.num_gestures = env.num_gestures
                
        self.X = None
        self.Y = None
        self.label = None

    def initialize_data(self):
        """
        Loads data for X, Y, and Label classes. Scaler normalizes EMG data, sets leave out indices, creates new folder name for images, acquires images, and prints data information.
        """

        # Initialize class objects
        self.X = X_Data(self.env)
        self.Y = Y_Data(self.env)
        self.label = Label_Data(self.env)

        # Wrapper class to call operations on all three
        all_data = Combined_Data(self.X, self.Y, self.label, self.env)
        all_data.load_data(self.exercises)
        all_data.scaler_normalize_emg()
        
        base_foldername_zarr = self.create_foldername_zarr()
        self.X.load_images(base_foldername_zarr)
        self.X.print_data_information()

        return self.X, self.Y, self.label
    
    # Helper functions for Data_Initializer class

    def create_foldername_zarr(self):
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
            if self.args.target_normalize > 0:
                base_foldername_zarr += 'target_normalize_' + str(self.args.target_normalize) + '/'  

        if self.args.turn_on_rms:
            base_foldername_zarr += 'RMS_input_windowsize_' + str(self.args.rms_input_windowsize) + '/'
        elif self.args.turn_on_spectrogram:
            base_foldername_zarr += 'spectrogram/'
        elif self.args.turn_on_cwt:
            base_foldername_zarr += 'cwt/'
        elif self.args.turn_on_hht:
            base_foldername_zarr += 'hht/'
        else:
            base_foldername_zarr += 'raw/'

        if self.exercises:
            if self.args.partial_dataset_ninapro:
                base_foldername_zarr += 'partial_dataset_ninapro/'
            else:
                exercises_numbers_filename = '-'.join(map(str, self.args.exercises))
                base_foldername_zarr += f'exercises{exercises_numbers_filename}/'
            
        if self.args.save_images: 
            if not os.path.exists(base_foldername_zarr):
                os.makedirs(base_foldername_zarr)

        return base_foldername_zarr


class Data_Splitter():

    def __init__(self, env):

        self.args = env.args
        self.utils = env.utils
        self.env = env

    def split_data(self, X_data, Y_data, label_data):

        if self.args.leave_n_subjects_out_randomly:
            split_strategy = "Leave_N_Subjects_Out_Randomly"
        elif self.args.held_out_test:
            split_strategy = "Held_Out_Test"
        elif self.args.leave_one_session_out:
            split_strategy = "Leave_One_Session_Out"
        elif self.args.leave_one_subject_out:
            split_strategy = "Leave_One_Subject_Out"
        elif self.utils.num_subjects == 1:
            split_strategy = "Single_Subject"
        else:
            raise ValueError("Please specify the type of test you want to run")

        strategy_module = import_module(f"Split_Strategies.{split_strategy}")
        strategy_class = getattr(strategy_module, split_strategy)
        strategy_class = strategy_class(X_data, Y_data, label_data, self.env)
        strategy_class.split()

class Run_Model():

    def __init__(self, env):
        self.args = env.args
        self.utils = env.utils
        self.exercises = env.exercises
        self.leaveOut = env.leaveOut
        self.env = env
        self.num_gestures = env.num_gestures

    def run_model(self, X, Y, label):

        # get the model 
        if self.args.turn_on_unlabeled_domain_adaptation:
            model_trainer = Unlabeled_Domain_Adaptation_Trainer(X, Y, label, self.env)
        else:
            if self.args.model == "MLP":
                model_trainer = MLP_Trainer(X, Y, label, self.env)
            elif self.args.model in ["SVC", "RF"]:
                model_trainer = SVC_RF_Trainer(X, Y, label, self.env)
            else:
                model_trainer = CNN_Trainer(X, Y, label, self.env)

        model_trainer.setup_model()
        model_trainer.model_loop()



hooks = Hook_Manager()
run_setup = Run_Setup()
hooks.register_hook("setup_run", run_setup.setup_run)
env = hooks.call_hook("setup_run")

data_initializer = Data_Initializer(env)
hooks.register_hook("initialize_data", data_initializer.initialize_data)
X, Y, label = hooks.call_hook("initialize_data")



data_splitter = Data_Splitter(env)
hooks.register_hook("split_data", data_splitter.split_data)
hooks.call_hook("split_data", X, Y, label)

run_model = Run_Model(env)
hooks.register_hook("run_model", run_model.run_model)
hooks.call_hook("run_model", X, Y, label)


