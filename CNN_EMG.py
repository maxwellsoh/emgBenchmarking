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
import datetime
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

class Setup():
    """
    Object to store run information shared across all Data and Model classes. 
    """

    def __init__(self):
        self.args = None
        import utils_MCS_EMG as utils
        self.utils = utils
        self.exercises = None
        self.project_name = None
        self.formatted_datetime = None
        self.leaveOut = None
        self.seed = None

class Run_Setup():
    """
    Sets up the run by reading in arguments, setting the dataset source, conducting safety checks, printing values and setting up env.

    Returns:
        env: Setup object which contains necessary information for the run 
    """


    def __init__(self):
        pass

    # Helper routines for main (can likely be moved into their own class)
    def parse_args(self): 
        """Argument parser for configuring different trials. 

        Returns:
            ArgumentParser: argument parser 
        """
        
        import utils_MCS_EMG as utils

        def list_of_ints(arg):
            """Define a custom argument type for a list of integers"""
            return list(map(int, arg.split(',')))

        ## Argument parser with optional argumenets

        # Create the parser
        parser = argparse.ArgumentParser(description="Include arguments for running different trials")
        parser.add_argument("--multiprocessing", type=utils.str2bool, help="Whether or not to use multiprocessing when acquiring data. Set to True by default.", default=True)
        parser.add_argument("--force_regression", type=utils.str2bool, help="Regression between EMG and force data", default=False)
        parser.add_argument('--dataset', help='dataset to test. Set to MCS_EMG by default', default="MCS_EMG")
        # Add argument for doing leave-one-subject-out
        parser.add_argument('--leave_one_subject_out', type=utils.str2bool, help='whether or not to do leave one subject out. Set to False by default.', default=False)
        # Add argument for leftout subject (indexed from 1)
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
        # Add argument for full or partial dataset for MCS EMG dataset
        parser.add_argument('--full_dataset_mcs', type=utils.str2bool, help='whether or not to use the full dataset for MCS EMG Dataset. Set to False by default.', default=False)
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
        parser.add_argument('--target_normalize', type=float, help='use a poportion of leftout data for normalization. Set to 0 by default.', default=0.0)
        # Test with transfer learning by using some data from the validation dataset
        parser.add_argument('--transfer_learning', type=utils.str2bool, help='use some data from the validation dataset for transfer learning. Set to False by default.', default=False)
        # Add argument for cross validation for time series
        parser.add_argument('--train_test_split_for_time_series', type=utils.str2bool, help='whether or not to use data split for time series. Set to False by default.', default=False)
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
        parser.add_argument('--proportion_unlabeled_data_from_leftout_subject', type=float, help='proportion of data from left-out-subject to keep as unlabeled data. Set to 0.75 by default.', default=0.75) # TODO: fix, we note that this affects leave-one-session-out even when fully supervised
        # Add argument to specify batch size
        parser.add_argument('--batch_size', type=int, help='batch size. Set to 64 by default.', default=64)
        # Add argument for whether to use unlabeled data for subjects used for training as well
        parser.add_argument('--proportion_unlabeled_data_from_training_subjects', type=float, help='proportion of data from training subjects to use as unlabeled data. Set to 0.0 by default.', default=0.0)
        # Add argument for cutting down amount of total data for training subjects
        parser.add_argument('--proportion_data_from_training_subjects', type=float, help='proportion of data from training subjects to use. Set to 1.0 by default.', default=1.0)
        # Add argument for loading unlabeled data from flexwear-hd dataset
        parser.add_argument('--load_unlabeled_data_flexwearhd', type=utils.str2bool, help='whether or not to load unlabeled data from FlexWear-HD dataset. Set to False by default.', default=False)

        parser.add_argument('--target_normalize_subject', type=int, help='number of subject that is left out for target normalization, starting from subject 1', default=0)

        args = parser.parse_args()
        return args
        
    def setup_for_dataset(self, args):
        """ Conducts safety checks on the args, downloads needed datasets, and imports the correct utils file. """

        
        exercises = False
        args.dataset = args.dataset.lower()

        if args.target_normalize_subject == 0:
            args.target_normalize_subject = args.leftout_subject
            print("Target normalize subject defaulting to leftout subject.")

        if args.model == "MLP" or args.model == "SVC" or args.model == "RF":
            print("Warning: not using pytorch, many arguments will be ignored")
            if args.turn_on_unlabeled_domain_adaptation:
                raise NotImplementedError("Cannot use unlabeled domain adaptation with MLP, SVC, or RF")
            if args.pretrain_and_finetune:
                raise NotImplementedError("Cannot use pretrain and finetune with MLP, SVC, or RF")

        if args.force_regression:
            assert args.dataset in {"ninapro-db3", "ninapro_db3"}, "Regression only implemented for Ninapro DB2 and DB3 dataset."

        if (args.dataset in {"uciemg", "uci"}):
            if (not os.path.exists("./uciEMG")):
                print("uciEMG dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--UCI'])
            import utils_UCI as utils
            project_name = 'emg_benchmarking_uci'
            args.dataset = "uciemg"

        elif (args.dataset in {"ninapro-db2", "ninapro_db2"}):
            if (not os.path.exists("./NinaproDB2")):
                print("NinaproDB2 dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--NinaproDB2'])
            import utils_NinaproDB2 as utils
            project_name = 'emg_benchmarking_ninapro-db2'
            exercises = True
            if args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for ninapro-db2; only one session exists")
            if args.force_regression:
                assert args.exercises == [3], "Regression only implemented for exercise 3"
            args.dataset = 'ninapro-db2'

        elif (args.dataset in { "ninapro-db5", "ninapro_db5"}):
            if (not os.path.exists("./NinaproDB5")):
                print("NinaproDB5 dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--NinaproDB5'])
                subprocess.run(['python', './process_NinaproDB5.py'])
            import utils_NinaproDB5 as utils
            project_name = 'emg_benchmarking_ninapro-db5'
            exercises = True
            if args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for ninapro-db5; only one session exists")
            args.dataset = 'ninapro-db5'

        elif (args.dataset in {"ninapro-db3", "ninapro_db3"}):
            import utils_NinaproDB3 as utils

            assert args.exercises == [1] or args.partial_dataset_ninapro or (args.exercises == [3] and args.force_regression), "Exercise C cannot be used for classification due to missing data."
            project_name = 'emg_benchmarking_ninapro-db3'
            exercises = True
            if args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for ninapro-db3; only one session exists")
            
            if args.force_regression:
                print("NOTE: Subject 10 is missing gesture data for exercise 3 and is not used for regression.")
            
            assert not(args.force_regression and args.leftout_subject == 10), "Subject 10 is missing gesture data for exercise 3 and cannot be used. Please choose another subject."

            if args.force_regression and args.leftout_subject == 11: 
                args.leftout_subject = 10
                # subject 10 is missing force data and is deleted internally 

            args.dataset = 'ninapro-db3'

        elif (args.dataset.lower() == "myoarmbanddataset"):
            if (not os.path.exists("./myoarmbanddataset")):
                print("myoarmbanddataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--MyoArmbandDataset'])
            import utils_MyoArmbandDataset as utils
            project_name = 'emg_benchmarking_myoarmbanddataset'
            if args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for myoarmbanddataset; only one session exists")
            args.dataset = 'myoarmbanddataset'

        elif (args.dataset.lower() == "hyser"):
            if (not os.path.exists("./hyser")):
                print("Hyser dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--Hyser'])
            import utils_Hyser as utils
            project_name = 'emg_benchmarking_hyser'
            args.dataset = 'hyser'

        elif (args.dataset.lower() == "capgmyo"):
            if (not os.path.exists("./CapgMyo_B")):
                print("CapgMyo_B dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--CapgMyo_B'])
            import utils_CapgMyo as utils
            project_name = 'emg_benchmarking_capgmyo'
            if args.leave_one_session_out:
                utils.num_subjects = 10
            args.dataset = 'capgmyo'

        elif (args.dataset.lower() == "flexwear-hd"):
            if (not os.path.exists("./FlexWear-HD")):
                print("FlexWear-HD dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--FlexWearHD_Dataset'])
            import utils_FlexWearHD as utils
            project_name = 'emg_benchmarking_flexwear-hd_dataset'
            # if args.leave_one_session_out:
                # raise ValueError("leave-one-session-out not implemented for FlexWear-HDDataset; only one session exists")
            args.dataset = 'flexwear-hd'

        elif (args.dataset.lower() == "sci"):
            import utils_SCI as utils
            project_name = 'emg_benchmarking_sci'
            args.dataset = 'sci'
            assert not args.transfer_learning, "Transfer learning not implemented for SCI dataset"
            assert not args.leave_one_subject_out, "Leave one subject out not implemented for SCI dataset"

        elif (args.dataset.lower() == "mcs"):
            if (not os.path.exists("./MCS_EMG")):
                print("MCS dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--MCS_EMG'])

            project_name = 'emg_benchmarking_mcs'
            if args.full_dataset_mcs:
                print(f"Using the full dataset for MCS EMG")
                utils.gesture_labels = utils.gesture_labels_full
                utils.numGestures = len(utils.gesture_labels)
            else: 
                print(f"Using the partial dataset for MCS EMG")
                utils.gesture_labels = utils.gesture_labels_partial
                utils.numGestures = len(utils.gesture_labels)
            if args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for MCS_EMG; only one session exists")
            args.dataset = 'mcs'
            
        else: 
            raise ValueError("Dataset not recognized. Please choose from 'uciemg', 'ninapro-db2', 'ninapro-db5', 'myoarmbanddataset', 'hyser'," +
                            "'capgmyo', 'flexwear-hd', 'sci', or 'mcs'")
            
        # Safety Check 
        if args.turn_off_scaler_normalization:
            assert args.target_normalize == 0.0, "Cannot turn off scaler normalization and turn on target normalize at the same time"

        if utils.num_subjects == 1:
                assert not args.pretrain_and_finetune, "Cannot pretrain and finetune with only one subject"

        
            

        # Add date and time to filename
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        print("------------------------------------------------------------------------------------------------------------------------")
        print("Starting run at", formatted_datetime)
        print("------------------------------------------------------------------------------------------------------------------------")

        

        return args, exercises, project_name, formatted_datetime, utils

    def print_params(self, args):
        for param, value in vars(args).items():
            if getattr(args, param):
                print(f"The value of --{param} is {value}")

    def set_exercise(self, args):
        """ Set the exercises for the partial dataset for Ninapro datasets. 
        """

        if args.partial_dataset_ninapro:
            if args.dataset == "ninapro-db2":
                args.exercises = [1]
            elif args.dataset == "ninapro-db5":
                args.exercises = [2]
            elif args.dataset == "ninapro-db3":
                args.exercises = [1]

        return args

    def set_seeds_for_reproducibility(self, args):
        """ Set seeds for reproducibility. 
        """

        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_run(self):

        args = self.parse_args()
        args, exercises, project_name, formatted_datetime, utils = self.setup_for_dataset(args)
        args = self.set_exercise(args)
        self.print_params(args)
        self.set_seeds_for_reproducibility(args)
        
        env = Setup()
        env.args = args
        env.exercises = exercises
        env.project_name = project_name
        env.formatted_datetime = formatted_datetime
        env.utils = utils
        env.leaveOut = int(args.leftout_subject)
        env.num_gestures = utils.numGestures
        env.seed = args.seed

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


