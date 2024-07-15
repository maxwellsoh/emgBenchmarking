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

# THIS SHOULD BE IN A MAIN FUNCTION


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



# TEMPORARY VALUES FOR TESTING (SHOULD EVENTUALLY TRANSITION OUT OF USING THEM SINCE THEY'RE INITIALIZED IN EACH CLASS)
exercises = env.exercises
project_name = env.project_name
formatted_datetime = env.formatted_datetime
utils = env.utils
args = env.args
num_gestures = env.num_gestures
leaveOut = env.leaveOut

data = X.data
labels = label.data
if args.force_regression:
    forces = Y.data
else:
    labels = Y.data
    

numGestures = env.num_gestures

if args.leave_n_subjects_out_randomly != 0 and (not args.turn_off_scaler_normalization and not (args.target_normalize > 0)):
    leaveOutIndices = X.leaveOutIndices

X_train = X.train if not None else None
Y_train = Y.train if not None else None
label_train = label.train if not None else None

X_train_unlabeled = X.train_unlabeled if not None else None
Y_train_unlabeled = Y.train_unlabeled if not None else None
label_train_unlabeled = label.train_unlabeled if not None else None

X_validation = X.validation if not None else None
Y_validation = Y.validation if not None else None
label_validation = label.validation if not None else None

X_test = X.test if not None else None
Y_test = Y.test if not None else None
label_test = label.test if not None else None

X_train_finetuning = X.train_finetuning if not None else None
Y_train_finetuning = Y.train_finetuning if not None else None
label_train_finetuning = label.train_finetuning if not None else None

X_train_finetuning_unlabeled = X.train_finetuning_unlabeled if not None else None
Y_train_finetuning_unlabeled = Y.train_finetuning_unlabeled if not None else None
label_train_finetuning_unlabeled = label.train_finetuning_unlabeled if not None else None


# =================================================================================================


# class ToNumpy:
#         """Custom transformation to convert PIL Images or Tensors to NumPy arrays."""
#         def __call__(self, pic):
#             if isinstance(pic, Image.Image):
#                 return np.array(pic)
#             elif isinstance(pic, torch.Tensor):
#                 # Make sure the tensor is in CPU and convert it
#                 return np.float32(pic.cpu().detach().numpy())
#             else:
#                 raise TypeError("Unsupported image type")

# model_name = args.model

# if args.model == "vit_tiny_patch2_32":
#     pretrain_path = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth"
# elif args.model == "resnet50":
#     pretrain_path = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
# else:
#     pretrain_path = f"https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/{model_name}_mlp_im_1k_224.pth"

# def ceildiv(a, b):
#         return -(a // -b)
    
# if args.turn_on_unlabeled_domain_adaptation: # set up datasets and config for unlabeled domain adaptation
#     current_date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     assert (args.transfer_learning and args.train_test_split_for_time_series) or args.leave_one_session_out, \
#         "Unlabeled Domain Adaptation requires transfer learning and cross validation for time series or leave one session out"
    
#     semilearn_config_dict = {
#         'algorithm': args.unlabeled_algorithm,
#         'net': args.model,
#         'use_pretrain': True,  
#         'pretrain_path': pretrain_path,
#         'seed': args.seed,

#         # optimization configs
#         'epoch': args.epochs,  # set to 100
#         # 'num_train_iter': args.epochs * ceildiv(X_train.shape[0], args.batch_size),
#         # 'num_eval_iter': ceildiv(X_train.shape[0], args.batch_size),
#         # 'num_log_iter': ceildiv(X_train.shape[0], args.batch_size),
#         'optim': 'AdamW',   # AdamW optimizer
#         'lr': args.learning_rate,  # Learning rate
#         'layer_decay': 0.5,  # Layer-wise decay learning rate  
#         'momentum': 0.9,  # Momentum
#         'weight_decay': 0.0005,  # Weight decay
#         'amp': True,  # Automatic mixed precision
#         'train_sampler': 'RandomSampler',  # Random sampler
#         'rank': 0,  # Rank
#         'batch_size': args.batch_size,  # Batch size
#         'eval_batch_size': args.batch_size, # Evaluation batch size
#         'use_wandb': True,
#         'ema_m': 0.999,
#         'save_dir': './saved_models/unlabeled_domain_adaptation/',
#         'save_name': f'{args.unlabeled_algorithm}_{args.model}_{args.dataset}_seed_{args.seed}_leave_{leaveOut}_unlabeled_domain_adaptation_{current_date_and_time}',
#         'resume': True,
#         'overwrite': True,
#         'load_path': f'./saved_models/unlabeled_domain_adaptation/{args.unlabeled_algorithm}_{args.model}_{args.dataset}_seed_{args.seed}_leave_{leaveOut}_unlabeled_domain_adaptation_{current_date_and_time}/latest_model.pth',
#         'scheduler': None,

#         # dataset configs
#         'dataset': 'none',
#         'num_labels': X_train.shape[0],
#         'num_classes': numGestures,
#         'input_size': 224,
#         'data_dir': './data',

#         # algorithm specific configs
#         'hard_label': True,
#         'uratio': 1.5,
#         'ulb_loss_ratio': 1.0,

#         # device configs
#         'gpu': args.gpu,
#         'world_size': 1,
#         'distributed': False,
#     } 
    
#     semilearn_config = get_config(semilearn_config_dict)

#     if args.model == 'vit_tiny_patch2_32':
#         semilearn_transform = transforms.Compose([transforms.Resize((32,32)), ToNumpy()])
#     else: 
#         semilearn_transform = transforms.Compose([transforms.Resize((224,224)), ToNumpy()])
    
#     labeled_dataset = BasicDataset(semilearn_config, X_train, torch.argmax(Y_train, dim=1), semilearn_config.num_classes, semilearn_transform, is_ulb=False)
#     if proportion_unlabeled_of_training_subjects>0:
#         unlabeled_dataset = BasicDataset(semilearn_config, X_train_unlabeled, torch.argmax(Y_train_unlabeled, dim=1), semilearn_config.num_classes, semilearn_transform, 
#                                         is_ulb=True, strong_transform=semilearn_transform)
#         # proportion_unlabeled_to_use = args.proportion_unlabeled_data_from_training_subjects
#     elif proportion_unlabeled_of_proportion_to_keep_of_leftout>0 or args.load_unlabeled_data_flexwearhd:
#         unlabeled_dataset = BasicDataset(semilearn_config, X_train_finetuning_unlabeled, torch.argmax(Y_train_finetuning_unlabeled, dim=1), semilearn_config.num_classes, semilearn_transform, 
#                                         is_ulb=True, strong_transform=semilearn_transform)
#         # proportion_unlabeled_to_use = args.proportion_unlabeled_data_from_leftout_subject
#     if args.pretrain_and_finetune:
#         finetune_dataset = BasicDataset(semilearn_config, X_train_finetuning, torch.argmax(Y_train_finetuning, dim=1), semilearn_config.num_classes, semilearn_transform, is_ulb=False)
#         finetune_unlabeled_dataset = BasicDataset(semilearn_config, X_train_finetuning_unlabeled, torch.argmax(Y_train_finetuning_unlabeled, dim=1), 
#                                                   semilearn_config.num_classes, semilearn_transform, is_ulb=True, strong_transform=semilearn_transform)
        
#     validation_dataset = BasicDataset(semilearn_config, X_validation, torch.argmax(Y_validation, dim=1), semilearn_config.num_classes, semilearn_transform, is_ulb=False)

#     proportion_unlabeled_to_use = len(unlabeled_dataset) / (len(labeled_dataset) + len(unlabeled_dataset)) 
#     labeled_batch_size = int(semilearn_config.batch_size * (1-proportion_unlabeled_to_use))
#     unlabeled_batch_size = int(semilearn_config.batch_size * proportion_unlabeled_to_use)
#     if labeled_batch_size + unlabeled_batch_size < semilearn_config.batch_size:
#         if labeled_batch_size < unlabeled_batch_size:
#             labeled_batch_size += 1
#         else:
#             unlabeled_batch_size += 1
        
#     labeled_iters = args.epochs * ceildiv(len(labeled_dataset), labeled_batch_size)
#     unlabeled_iters = args.epochs * ceildiv(len(unlabeled_dataset), unlabeled_batch_size)
#     iters_for_loader = max(labeled_iters, unlabeled_iters)
#     train_labeled_loader = get_data_loader(semilearn_config, labeled_dataset, labeled_batch_size, num_workers=multiprocessing.cpu_count()//8, 
#                                            num_epochs=args.epochs, num_iters=iters_for_loader)
#     if proportion_unlabeled_of_training_subjects>0 or proportion_unlabeled_of_proportion_to_keep_of_leftout>0 or args.load_unlabeled_data_flexwearhd:
#         train_unlabeled_loader = get_data_loader(semilearn_config, unlabeled_dataset, unlabeled_batch_size, num_workers=multiprocessing.cpu_count()//8,
#                                                  num_epochs=args.epochs, num_iters=iters_for_loader)
        
#     semilearn_config.num_train_iter = iters_for_loader
#     semilearn_config.num_eval_iter = ceildiv(iters_for_loader, args.epochs)
#     semilearn_config.num_log_iter = ceildiv(iters_for_loader, args.epochs)
    
#     semilearn_algorithm = get_algorithm(semilearn_config, get_net_builder(semilearn_config.net, from_name=False), tb_log=None, logger=None)
#     semilearn_algorithm.model = send_model_cuda(semilearn_config, semilearn_algorithm.model)
#     semilearn_algorithm.ema_model = send_model_cuda(semilearn_config, semilearn_algorithm.ema_model, clip_batch=False)
    
#     print("Batches per epoch:", semilearn_config.num_eval_iter)
        
#     if args.pretrain_and_finetune:
#         proportion_unlabeled_to_use = len(finetune_unlabeled_dataset) / (len(finetune_dataset) + len(finetune_unlabeled_dataset))
#         labeled_batch_size = int(semilearn_config.batch_size * (1-proportion_unlabeled_to_use))
#         unlabeled_batch_size = int(semilearn_config.batch_size * proportion_unlabeled_to_use)
#         if labeled_batch_size + unlabeled_batch_size < semilearn_config.batch_size:
#             if labeled_batch_size < unlabeled_batch_size:
#                 labeled_batch_size += 1
#             else:
#                 unlabeled_batch_size += 1
#         labeled_iters = args.epochs * ceildiv(len(finetune_dataset), labeled_batch_size)
#         unlabeled_iters = args.epochs * ceildiv(len(finetune_unlabeled_dataset), unlabeled_batch_size)
#         iters_for_loader = max(labeled_iters, unlabeled_iters)
#         train_finetuning_loader = get_data_loader(semilearn_config, finetune_dataset, labeled_batch_size, num_workers=multiprocessing.cpu_count()//8,
#                                                   num_epochs=args.epochs, num_iters=iters_for_loader)
#         train_finetuning_unlabeled_loader = get_data_loader(semilearn_config, finetune_unlabeled_dataset, unlabeled_batch_size, num_workers=multiprocessing.cpu_count()//8,
#                                                             num_epochs=args.epochs, num_iters=iters_for_loader)
#     validation_loader = get_data_loader(semilearn_config, validation_dataset, semilearn_config.eval_batch_size, num_workers=multiprocessing.cpu_count()//8)

# # [START] Model Initialization
# else:
#     if args.model == 'resnet50_custom':
#         model = resnet50(weights=ResNet50_Weights.DEFAULT)
#         model = nn.Sequential(*list(model.children())[:-4])
#         # #model = nn.Sequential(*list(model.children())[:-4])
#         num_features = model[-1][-1].conv3.out_channels
#         # #num_features = model.fc.in_features
#         dropout = 0.5
#         model.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
#         model.add_module('fc1', nn.Linear(num_features, 512))
#         model.add_module('relu', nn.ReLU())
#         model.add_module('dropout1', nn.Dropout(dropout))
#         model.add_module('fc3', nn.Linear(512, numGestures))
#         model.add_module('softmax', nn.Softmax(dim=1))
#     elif args.model == 'resnet50':
#         model = resnet50(weights=ResNet50_Weights.DEFAULT)
#         # Replace the last fully connected layer
#         num_ftrs = model.fc.in_features  # Get the number of input features of the original fc layer
#         model.fc = nn.Linear(num_ftrs, numGestures)  # Replace with a new linear layer
#     elif args.model == 'convnext_tiny_custom':
#         class LayerNorm2d(nn.LayerNorm):
#             def forward(self, x: torch.Tensor) -> torch.Tensor:
#                 x = x.permute(0, 2, 3, 1)
#                 x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#                 x = x.permute(0, 3, 1, 2)
#                 return x
            
#         # Referencing: https://medium.com/exemplifyml-ai/image-classification-with-resnet-convnext-using-pytorch-f051d0d7e098


#         n_inputs = 768
#         hidden_size = 128 # default is 2048
#         n_outputs = numGestures

#         # model = timm.create_model(model_name, pretrained=True, num_classes=10)
#         model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
#         #model = nn.Sequential(*list(model.children())[:-4])
#         #model = nn.Sequential(*list(model.children())[:-3])
#         #num_features = model[-1][-1].conv3.out_channels
#         #num_features = model.fc.in_features
#         dropout = 0.1 # was 0.5

#         sequential_layers = nn.Sequential(
#             LayerNorm2d((n_inputs,), eps=1e-06, elementwise_affine=True),
#             nn.Flatten(start_dim=1, end_dim=-1),
#             nn.Linear(n_inputs, hidden_size, bias=True),
#             nn.BatchNorm1d(hidden_size),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.GELU(),
#             nn.Linear(hidden_size, n_outputs),
#             nn.LogSoftmax(dim=1)
#         )
#         model.classifier = sequential_layers
#     elif args.model == 'vit_tiny_patch2_32':
#         pretrain_path = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth"
#         model = VisualTransformer.vit_tiny_patch2_32(pretrained=True, pretrained_path=pretrain_path, num_classes=numGestures)
#     elif args.model == 'MLP' or args.model == 'SVC' or args.model == 'RF':
#         model = None # Will be initialized later
#     else: 
#         # model_name = 'efficientnet_b0'  # or 'efficientnet_b1', ..., 'efficientnet_b7'
#         # model_name = 'tf_efficientnet_b3.ns_jft_in1k'

#         if args.force_regression: 
#             model = timm.create_model(model_name, pretrained=True, num_classes=Y_validation.shape[1])
#         else: 
#             model = timm.create_model(model_name, pretrained=True, num_classes=numGestures)
#         # # Load the Vision Transformer model
#         # model_name = 'vit_base_patch16_224'  # This is just one example, many variations exist
#         # model = timm.create_model(model_name, pretrained=True, num_classes=utils.numGestures)

# # [END] Model Initialization
# class CustomDataset(Dataset):
#     def __init__(self, X, Y, transform=None):
#         self.X = X
#         self.Y = Y
#         self.transform = transform

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, index):
#         x = self.X[index]
#         y = self.Y[index]
#         if self.transform:
#             x = self.transform(x)
#         return x, y

# if not args.turn_on_unlabeled_domain_adaptation: # set up datasets, loaders, schedulers, and optimizer
#     if args.model not in ['MLP', 'SVC', 'RF']:
#         num = 0
#         for name, param in model.named_parameters():
#             num += 1
#             if (num > 0):
#             #if (num > 72): # for -3
#             #if (num > 33): # for -4
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False

#     batch_size = args.batch_size

 
#     from torchvision import transforms
#     from PIL import Image

#     class ToVector:
    
#         def __call__(self, img):
#             # Convert image to a tensor and flatten it
#             return img.flatten()

#     if args.model == 'vit_tiny_patch2_32':
#         resize_transform = transforms.Compose([transforms.Resize((32,32)), ToNumpy()])
#     else:
        

#         resize_transform = transforms.Compose([transforms.Resize((224,224)), ToNumpy()])
#         if args.model == "MLP":
#             resize_transform = transforms.Compose([resize_transform, ToVector()])

#     train_dataset = CustomDataset(X_train, Y_train, transform=resize_transform)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)
#     val_dataset = CustomDataset(X_validation, Y_validation, transform=resize_transform)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)
#     test_dataset = CustomDataset(X_test, Y_test, transform=resize_transform)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)

#     # Define the loss function and optimizer
#     if args.force_regression:
#         criterion = nn.MSELoss()
#     else:
#         criterion = nn.CrossEntropyLoss()
#     learn = args.learning_rate
#     if args.model not in ['MLP', 'SVC', 'RF']:
#         optimizer = torch.optim.Adam(model.parameters(), lr=learn)

#     num_epochs = args.epochs
#     if args.turn_on_cosine_annealing:
#         number_cycles = 5
#         annealing_multiplier = 2
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=utils.periodLengthForAnnealing(num_epochs, annealing_multiplier, number_cycles),
#                                                                             T_mult=annealing_multiplier, eta_min=1e-5, last_epoch=-1)
#     elif args.turn_on_cyclical_lr:
#         # Define the cyclical learning rate scheduler
#         step_size = len(train_loader) * 6  # Number of iterations in half a cycle
#         base_lr = 1e-4  # Minimum learning rate
#         max_lr = 1e-3  # Maximum learning rate
#         scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=step_size, mode='triangular2', cycle_momentum=False)

#     # Training loop
#     gc.collect()
#     torch.cuda.empty_cache()

# def create_wandb_runname():
#     wandb_runname = 'CNN_seed-'+str(args.seed)
#     if args.turn_on_kfold:
#         wandb_runname += '_k-fold-'+str(args.kfold)+'_fold-index-'+str(args.fold_index)
#     if args.turn_on_cyclical_lr:
#         wandb_runname += '_cyclical-lr'
#     if args.turn_on_cosine_annealing: 
#         wandb_runname += '_cosine-annealing'
#     if args.turn_on_rms:
#         wandb_runname += '_rms-'+str(args.rms_input_windowsize)
#     if args.turn_on_magnitude:  
#         wandb_runname += '_mag-'
#     if args.leftout_subject != 0:
#         wandb_runname += '_LOSO-'+str(args.leftout_subject)
#     wandb_runname += '_' + model_name
#     if (exercises and not args.partial_dataset_ninapro):
#         wandb_runname += '_exer-' + ''.join(character for character in str(args.exercises) if character.isalnum())
#     if args.dataset == "mcs":
#         if args.full_dataset_mcs:
#             wandb_runname += '_full'
#         else:
#             wandb_runname += '_partial'
#     if args.dataset == "ninapro-db2" or args.dataset == "ninapro-db5":
#         if args.partial_dataset_ninapro:
#             wandb_runname += '_partial'
#     if args.turn_on_spectrogram:
#         wandb_runname += '_spect'
#     if args.turn_on_cwt:
#         wandb_runname += '_cwt'
#     if args.turn_on_hht:
#         wandb_runname += '_hht'
#     if args.reduce_training_data_size:
#         wandb_runname += '_reduced-training-data-size-' + str(args.reduced_training_data_size)
#     if args.leave_n_subjects_out_randomly != 0:
#         wandb_runname += '_leave_n_subjects_out-'+str(args.leave_n_subjects_out_randomly)
#     if args.turn_off_scaler_normalization:
#         wandb_runname += '_no-scal-norm'
#     if args.target_normalize > 0:
#         wandb_runname += '_targ-norm-' + str(args.target_normalize)
#     if args.load_few_images:
#         wandb_runname += '_load-few'
#     if args.transfer_learning:
#         wandb_runname += '_tran-learn'
#         wandb_runname += '-prop-' + str(args.proportion_transfer_learning_from_leftout_subject)
#     if args.train_test_split_for_time_series:   
#         wandb_runname += '_cv-for-ts'
#     if args.reduce_data_for_transfer_learning != 1:
#         wandb_runname += '_red-data-for-tran-learn-' + str(args.reduce_data_for_transfer_learning)
#     if args.leave_one_session_out:
#         wandb_runname += '_leave-one-sess-out'
#     if args.leave_one_subject_out:
#         wandb_runname += '_loso'
#     if args.one_subject_for_training_set_for_session_test:
#         wandb_runname += '_one-subj-for-training-set'
#     if args.held_out_test:
#         wandb_runname += '_held-out'
#     if args.pretrain_and_finetune:
#         wandb_runname += '_pretrain-finetune'
#     if args.turn_on_unlabeled_domain_adaptation:
#         wandb_runname += '_unlabeled-adapt'
#         wandb_runname += '-algo-' + args.unlabeled_algorithm
#         wandb_runname += '-prop-unlabel-leftout' + str(args.proportion_unlabeled_data_from_leftout_subject)
#     if args.proportion_data_from_training_subjects<1.0:
#         wandb_runname += '_train-subj-prop-' + str(args.proportion_data_from_training_subjects)
#     if args.proportion_unlabeled_data_from_training_subjects>0:
#         wandb_runname += '_unlabel-subj-prop-' + str(args.proportion_unlabeled_data_from_training_subjects)
#     if args.load_unlabeled_data_flexwearhd:
#         wandb_runname += '_load-unlabel-data-flexwearhd'
#     if args.force_regression:
#         wandb_runname += '_force-regression'

#     return wandb_runname
# wandb_runname = create_wandb_runname()

# if (args.held_out_test):

#     if args.turn_on_kfold:
#         project_name += '_k-fold-'+str(args.kfold)
#     else:
#         project_name += '_heldout'
# elif args.leave_one_subject_out:
#     project_name += '_LOSO'
# elif args.leave_one_session_out:
#     project_name += '_leave-one-session-out'
    
# def freezeAllLayersButLastLayer(model):
#     # Convert model children to a list
#     children = list(model.children())

#     # Freeze parameters in all children except the last one
#     for child in children[:-1]:
#         for param in child.parameters():
#             param.requires_grad = False

#     # Unfreeze the last layer
#     for param in children[-1].parameters():
#         param.requires_grad = True

# def unfreezeAllLayers(model):
#     for name, param in model.named_parameters():
#         param.requires_grad = True
#     return model

# project_name += args.project_name_suffix

# run = wandb.init(name=wandb_runname, project=project_name)
# wandb.config.lr = args.learning_rate

# if args.leave_n_subjects_out_randomly != 0:
#     wandb.config.left_out_subjects = leaveOutIndices

# device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
# print("Device:", device)
# if not args.turn_on_unlabeled_domain_adaptation and args.model not in ['MLP', 'SVC', 'RF']:
#     model.to(device)

#     wandb.watch(model)

# testrun_foldername = f'test/{project_name}/{wandb_runname}/{formatted_datetime}/'
# # Make folder if it doesn't exist
# if not os.path.exists(testrun_foldername):
#     os.makedirs(testrun_foldername)
# model_filename = f'{testrun_foldername}model_{formatted_datetime}.pth'

# if (exercises):
#     if not args.partial_dataset_ninapro:
#         gesture_labels = utils.gesture_labels['Rest']
#         for exercise_set in args.exercises:
#             gesture_labels = gesture_labels + utils.gesture_labels[exercise_set]
#     else:
#         gesture_labels = utils.partial_gesture_labels
# else:
#     gesture_labels = utils.gesture_labels

# # Plot and log images
# if args.force_regression:
#     test = label_test
# else:
#     test = Y_test

# utils.plot_average_images(X_test, np.argmax(test.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'test')
# utils.plot_first_fifteen_images(X_test, np.argmax(test.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'test')
    
# def get_metrics(testing=True):
#     """
#     Constructs training and validation metric arrays based on whether it is a regression or classification task. Also returns testing metrics based on testing flag.

#     All changes to metrics should be done here. 
#     """
#     def get_regression_metrics():
#         regression_metrics = [
#             torchmetrics.MeanSquaredError().to(device), 
#             torchmetrics.MeanSquaredError(squared=False).to(device), 
#             torchmetrics.MeanAbsoluteError().to(device),
#             torchmetrics.R2Score(num_outputs=6, multioutput="uniform_average").to(device), 
#             torchmetrics.R2Score(num_outputs=6, multioutput="raw_values").to(device)
#         ]
#         for metric, name in zip(regression_metrics, ["MeanSquaredError", "RootMeanSquaredError","MeanAbsoluteError", "R2Score_UniformAverage", "R2Score_RawValues"]):
#             metric.name = name

#         return regression_metrics

#     def get_classification_metrics():
#         classification_metrics = [

#             torchmetrics.Accuracy(task="multiclass", num_classes=numGestures, average="macro").to(device),
#             torchmetrics.Precision(task="multiclass", num_classes=numGestures, average="macro").to(device),
#             torchmetrics.Recall(task="multiclass", num_classes=numGestures, average="macro").to(device),
#             torchmetrics.F1Score(task="multiclass", num_classes=numGestures, average="macro").to(device),
#             torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=numGestures, average="macro").to(device),
#             torchmetrics.Accuracy(task="multiclass", num_classes=numGestures, average="micro").to(device),
#             torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=numGestures, average="micro").to(device),
#             torchmetrics.AUROC(task="multiclass", num_classes=numGestures, average="macro").to(device),
#             torchmetrics.AveragePrecision(task="multiclass", num_classes=numGestures, average="macro").to(device)
#         ]
#         for metric, name in zip(classification_metrics, ["Macro_Acc", "Macro_Precision", "Macro_Recall", "Macro_F1Score", "Macro_Top5Accuracy", "Micro_Accuracy", "Micro_Top5Accuracy", "Macro_AUROC","Macro_AUPRC"]):
#             metric.name = name
        
#         return classification_metrics
        
#     if args.force_regression:
#         training_metrics = get_regression_metrics()
#         validation_metrics = get_regression_metrics()
#         testing_metrics = get_regression_metrics()

#     else: 
#         training_metrics = get_classification_metrics()
#         validation_metrics = get_classification_metrics()
#         testing_metrics = get_classification_metrics()

#     if not testing:
#         return training_metrics, validation_metrics
#     return training_metrics, validation_metrics, testing_metrics
       
# if args.force_regression: 
#     validation = label_validation
#     train = label_train
# else:
#     validation = Y_validation
#     train = Y_train

# utils.plot_average_images(X_validation, np.argmax(validation.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'gesture validation')
# utils.plot_first_fifteen_images(X_validation, np.argmax(validation.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'gesture validation')

# utils.plot_average_images(X_train, np.argmax(train.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'gesture train')
# utils.plot_first_fifteen_images(X_train, np.argmax(train.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'gesture train')

# # Initialize number of classes based on regressions or classification
# if args.force_regression:
#     num_classes = Y_train.shape[1]
#     assert num_classes == 6
# else: 
#     num_classes = numGestures

# # def get_metrrics  
# if args.pretrain_and_finetune:
    
#     if args.force_regression:
#         train_finetuning = label_train_finetuning
#     else:
#         train_finetuning = Y_train_finetuning

#     utils.plot_average_images(X_train_finetuning, np.argmax(train_finetuning.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'gesture train_finetuning')
#     utils.plot_first_fifteen_images(X_train_finetuning, np.argmax(train_finetuning.cpu().detach().numpy(), axis=1), gesture_labels, testrun_foldername, args, formatted_datetime, 'gesture train_finetuning')

# if args.turn_on_unlabeled_domain_adaptation:
#     print("Pretraining the model...")
#     semilearn_algorithm.loader_dict = {}
#     semilearn_algorithm.loader_dict['train_lb'] = train_labeled_loader
#     if proportion_unlabeled_of_training_subjects>0 or proportion_unlabeled_of_proportion_to_keep_of_leftout>0 or args.load_unlabeled_data_flexwearhd:
#         semilearn_algorithm.loader_dict['train_ulb'] = train_unlabeled_loader
#     semilearn_algorithm.loader_dict['eval'] = validation_loader
#     semilearn_algorithm.scheduler = None
    
#     semilearn_algorithm.train()

#     if args.model == 'vit_tiny_patch2_32':
#         resize_transform = transforms.Compose([transforms.Resize((32,32)), ToNumpy()])
#     else:
#         resize_transform = transforms.Compose([transforms.Resize((224,224)), ToNumpy()])
#     test_dataset = CustomDataset(X_test, Y_test, transform=resize_transform)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)
#     criterion = nn.CrossEntropyLoss()

#     _, _, testing_metrics = get_metrics()

#     wandb.init(name=wandb_runname+"_unlab_test", project=project_name)
#     ml_utils.evaluate_model_on_test_set(semilearn_algorithm.model, test_loader, device, numGestures, criterion, args, testing_metrics)
#     wandb.finish()

#     if args.pretrain_and_finetune:
#         print("Finetuning the model...")
#         run = wandb.init(name=wandb_runname+"_unlab_finetune", project=project_name)
#         wandb.config.lr = args.learning_rate
        
#         semilearn_config_dict['num_train_iter'] = semilearn_config.num_train_iter + iters_for_loader
#         semilearn_config_dict['num_eval_iter'] = ceildiv(iters_for_loader, args.finetuning_epochs)
#         semilearn_config_dict['num_log_iter'] = ceildiv(iters_for_loader, args.finetuning_epochs)
#         semilearn_config_dict['epoch'] = args.finetuning_epochs + args.epochs
#         semilearn_config_dict['algorithm'] = args.unlabeled_algorithm
        
#         semilearn_config = get_config(semilearn_config_dict)
#         semilearn_algorithm = get_algorithm(semilearn_config, get_net_builder(semilearn_config.net, from_name=False), tb_log=None, logger=None)
#         semilearn_algorithm.epochs = args.epochs + args.finetuning_epochs # train for the same number of epochs as the previous training
#         semilearn_algorithm.model = send_model_cuda(semilearn_config, semilearn_algorithm.model)
#         semilearn_algorithm.load_model(semilearn_config.load_path)
#         semilearn_algorithm.ema_model = send_model_cuda(semilearn_config, semilearn_algorithm.ema_model, clip_batch=False)
#         semilearn_algorithm.loader_dict = {}
#         semilearn_algorithm.loader_dict['train_lb'] = train_finetuning_loader
#         semilearn_algorithm.scheduler = None
        
#         if proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
#             semilearn_algorithm.loader_dict['train_ulb'] = train_finetuning_unlabeled_loader
#         elif proportion_unlabeled_of_training_subjects>0 or args.load_unlabeled_data_flexwearhd:
#             semilearn_algorithm.loader_dict['train_ulb'] = train_unlabeled_loader

#         semilearn_algorithm.loader_dict['eval'] = validation_loader
#         semilearn_algorithm.train()

#         wandb.init(name=wandb_runname+"_unlab_finetune_test", project=project_name)
#         ml_utils.evaluate_model_on_test_set(semilearn_algorithm.model, test_loader, device, numGestures, criterion, args, testing_metrics)
#         wandb.finish()

# else: 
#     if args.model in ['MLP', 'SVC', 'RF']:
#         # New block of code to add 
#         class MLP(nn.Module):
#             def __init__(self, input_size, hidden_sizes, output_size):
#                 super(MLP, self).__init__()
#                 self.hidden_layers = nn.ModuleList()
#                 self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
#                 for i in range(1, len(hidden_sizes)):
#                     self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
#                 self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

#             def forward(self, x):
#                 for hidden in self.hidden_layers:
#                     x = F.relu(hidden(x))
#                 x = self.output_layer(x)
#                 return x
            
#         def get_data_from_loader(loader):
#             X = []
#             Y = []
#             for X_batch, Y_batch in tqdm(loader, desc="Batches convert to Numpy"):
#                 # Flatten each image from [batch_size, 3, 224, 224] to [batch_size, 3*224*224]
#                 # X_batch_flat = X_batch.view(X_batch.size(0), -1).cpu().numpy().astype(np.float64)
#                 Y_batch_indices = torch.argmax(Y_batch, dim=1)  # Convert one-hot to class indices
#                 X.append(X_batch)
#                 Y.append(Y_batch_indices.cpu().numpy().astype(np.int64))
#             return np.vstack(X), np.hstack(Y)
        
#         if args.model == 'MLP':
#             # PyTorch MLP model
#             input_size = 3 * 224 * 224  # Change according to your input size
#             hidden_sizes = [512, 256]  # Example hidden layer sizes
#             output_size = num_classes  # Number of classes
#             model = MLP(input_size, hidden_sizes, output_size).to(device)
#             optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#             if args.force_regression:
#                 criterion = nn.MSELoss()
#             else:
#                 criterion = nn.CrossEntropyLoss()
        
#         elif args.model == 'SVC':
#             model = SVC(probability=True)
        
#         elif args.model == 'RF':
#             model = RandomForestClassifier()
            
#         if args.model == 'MLP':
#             # PyTorch training loop for MLP
#             training_metrics, validation_metrics = get_metrics(testing=False)

#             for epoch in tqdm(range(num_epochs), desc="Epoch"):
#                 model.train()
#                 # Metrics

#                 # Initialize metrics 
#                 for train_metric in training_metrics:
#                     train_metric.reset()

#                 train_loss = 0.0

#                 with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as t:
#                     if not args.force_regression:
#                         ground_truth_train_all = []
#                         outputs_all = [] # NOTE: why is this inside the loop for MLP but outside in CNN
                    
#                     for X_batch, Y_batch in t:
#                         X_batch = X_batch.view(X_batch.size(0), -1).to(device).to(torch.float32)
#                         if args.force_regression:
#                             Y_batch = Y_batch.to(device).to(torch.float32) 
#                         else:
#                             Y_batch = torch.argmax(Y_batch, dim=1).to(device).to(torch.int64)

#                         optimizer.zero_grad()
#                         output = model(X_batch)
#                         loss = criterion(output, Y_batch)
#                         loss.backward()
#                         optimizer.step()

#                         train_loss += loss.item()
#                         for train_metric in training_metrics:
#                             if train_metric.name != "Macro_AUROC" and train_metric.name != "Macro_AUPRC":
#                                 train_metric(output, Y_batch)
                        
#                         if not args.force_regression:
#                             outputs_all.append(output)
#                             # ground_truth_train_all.append(torch.argmax(Y_batch, dim=1)) 
#                             # NOTE: This code was double flattening it raising dimension errors.
#                             ground_truth_train_all.append(Y_batch)
                        
#                         if not args.force_regression:
#                             if t.n % 10 == 0:
#                                 train_micro_acc = next(metric for metric in training_metrics if metric.name == "Micro_Accuracy")
#                                 t.set_postfix({"Batch Loss": loss.item(), "Batch Acc": train_micro_acc.compute().item()})

#                         del X_batch, Y_batch, output
#                         torch.cuda.empty_cache()
                    
#                     if not args.force_regression:
#                         outputs_all = torch.cat(outputs_all, dim=0).to(device)
#                         ground_truth_train_all = torch.cat(ground_truth_train_all, dim=0).to(device)

#                         train_macro_auroc = next(metric for metric in training_metrics if metric.name == "Macro_AUROC")
#                         train_macro_auprc = next(metric for metric in training_metrics if metric.name == "Macro_AUPRC")

#                         train_macro_auroc(outputs_all, ground_truth_train_all)
#                         train_macro_auprc(outputs_all, ground_truth_train_all)


#                 # Validation
#                 model.eval()

#                 for val_metric in validation_metrics:
#                     val_metric.reset()
                
#                 if not args.force_regression:
#                     all_val_outputs = []
#                     all_val_labels = []

#                 val_loss = 0.0
#                 with torch.no_grad():
        
#                     for X_batch, Y_batch in val_loader:
#                         X_batch = X_batch.view(X_batch.size(0), -1).to(device).to(torch.float32)
#                         if args.force_regression:
#                             Y_batch =  Y_batch.to(device).to(torch.float32)
#                         else: 
#                             Y_batch = torch.argmax(Y_batch, dim=1).to(device).to(torch.int64)

#                         output = model(X_batch)
#                         for validation_metric in validation_metrics:
#                             validation_metric(output, Y_batch)

#                         val_loss += criterion(output, Y_batch).item()
            
#                         if not args.force_regression:
#                             all_val_outputs.append(output)
#                             all_val_labels.append(Y_batch)

#                         del X_batch, Y_batch
#                         torch.cuda.empty_cache()

#                 if not args.force_regression:
#                     all_val_outputs = torch.cat(all_val_outputs, dim=0).to(device)
#                     all_val_labels = torch.cat(all_val_labels, dim=0)

#                     Y_validation_long = torch.argmax(Y_validation, dim=1).to(device).to(torch.int64)

#                     true_labels = Y_validation_long.cpu().detach().numpy()
#                     test_predictions = np.argmax(all_val_outputs.cpu().detach().numpy(), axis=1)
#                     conf_matrix = confusion_matrix(true_labels, test_predictions)
#                     print("Confusion Matrix:")
#                     print(conf_matrix)

#                     val_macro_auroc = next(metric for metric in validation_metrics if metric.name == "Macro_AUROC")
#                     val_macro_auprc = next(metric for metric in validation_metrics if metric.name == "Macro_AUPRC")

#                     val_macro_auroc(all_val_outputs, Y_validation_long)
#                     val_macro_auprc(all_val_outputs, Y_validation_long)
                    

#                 # Average the losses and print the metrics
#                 train_loss /= len(train_loader)
#                 val_loss /= len(val_loader)

#                 if not args.force_regression:
#                     tpr_results = ml_utils.evaluate_model_tpr_at_fpr(model, val_loader, device, numGestures)
#                     fpr_results = ml_utils.evaluate_model_fpr_at_tpr(model, val_loader, device, numGestures)
#                     confidence_levels, proportions_above_confidence_threshold = ml_utils.evaluate_confidence_thresholding(model, val_loader, device)

#                 # Compute the metrics and store them in dictionaries (to prevent multiple calls to compute)
#                 training_metrics_values = {metric.name: metric.compute() for metric in training_metrics}
#                 validation_metrics_values = {metric.name: metric.compute() for metric in validation_metrics}

                
#                 print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#                 training_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in training_metrics_values.items())
#                 print(f"Train Metrics: {training_metrics_str}")

#                 val_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in validation_metrics_values.items())
#                 print(f"Val Metrics: {val_metrics_str}")
                
#                 if not args.force_regression:
#                     for fpr, tprs in tpr_results.items():
#                         print(f"Val TPR at {fpr}: {', '.join(f'{tpr:.4f}' for tpr in tprs)}")
#                     for confidence_level, accuracy in confidence_levels.items():
#                         print(f"Val Accuracy at confidence level >{confidence_level}: {accuracy:.4f}")

#                 training_metrics_values = {metric.name: metric.compute() for metric in training_metrics}
#                 validation_metrics_values = {metric.name: metric.compute() for metric in validation_metrics}

#                 # Log metrics to wandb or any other tracking tool
#                 wandb.log({
#                 "train/Loss": train_loss,
#                 "train/Learning Rate": optimizer.param_groups[0]['lr'],
#                 "train/Epoch": epoch,
#                 "validation/Loss": val_loss,
#                 **{
#                     f"train/{name}": value.item() 
#                     for name, value in training_metrics_values.items() 
#                     if name != 'R2Score_RawValues'
#                 },
#                 **{
#                     f"train/R2Score_RawValues_{i+1}": v.item() 
#                     for name, value in training_metrics_values.items() 
#                     if name == 'R2Score_RawValues'
#                     for i, v in enumerate(value)
#                 },
#                 **{
#                     f"validation/{name}": value.item() 
#                     for name, value in validation_metrics_values.items() 
#                     if name != 'R2Score_RawValues'
#                 },
#                 **{
#                     f"validation/R2Score_RawValues_{i+1}": v.item() 
#                     for name, value in validation_metrics_values.items() 
#                     if name == 'R2Score_RawValues'
#                     for i, v in enumerate(value)
#                 },


#                 **({f"tpr_at_fixed_fpr/Average Val TPR at {fpr} FPR": np.mean(tprs) for fpr, tprs in tpr_results.items()} if not args.force_regression else {}),
#                 **({f"fpr_at_fixed_tpr/Average Val FPR at {tpr} TPR": np.mean(fprs) for tpr, fprs in fpr_results.items()}if not args.force_regression else {}),
#                 **({f"confidence_level_accuracies/Val Accuracy at {int(confidence_level*100)}% confidence": acc for confidence_level, acc in confidence_levels.items()} if not args.force_regression else {}),
#                 **({f"proportion_above_confidence_threshold/Val Proportion above {int(confidence_level*100)}% confidence": prop for confidence_level, prop in proportions_above_confidence_threshold.items()} if not args.force_regression else {})
#             })
                
#             torch.save(model.state_dict(), model_filename)
#             wandb.save(f'model/modelParameters_{formatted_datetime}.pth')

#         else: # SVC or RF
#             X_train, Y_train = get_data_from_loader(train_loader)
#             X_val, Y_val = get_data_from_loader(val_loader)
#             # X_test, Y_test = get_data_from_loader(test_loader)

#             print("Data loaded")
#             model.fit(X_train, Y_train)
#             print("Model trained")
#             train_preds = model.predict(X_train)
#             print("Train predictions made")
#             val_preds = model.predict(X_val)
#             print("Validation predictions made")
#             # test_preds = model.predict(X_test)

#             train_acc = accuracy_score(Y_train, train_preds)
#             val_acc = accuracy_score(Y_val, val_preds)
#             # test_acc = accuracy_score(Y_test, test_preds)

#             train_loss = log_loss(Y_train, model.predict_proba(X_train))
#             val_loss = log_loss(Y_val, model.predict_proba(X_val))
#             # test_loss = log_loss(Y_test, model.predict_proba(X_test))

#             print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
#             print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
#             # print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

#             wandb.log({
#                 "Train Loss": train_loss,
#                 "Train Acc": train_acc,
#                 "Val Loss": val_loss,
#                 "Val Acc": val_acc,
#                 # "Test Loss": test_loss,
#                 # "Test Acc": test_acc
#             })

  
#     else: # CNN training

#         # Initialize metrics
#         if args.held_out_test or args.pretrain_and_finetune:
#             training_metrics, validation_metrics, testing_metrics = get_metrics()
#         else: 
#             training_metrics, validation_metrics = get_metrics(testing=False)

#         for metric in training_metrics:
#             print("metric_name:", metric.name)
        
#         for epoch in tqdm(range(num_epochs), desc="Epoch"):
#             model.train()
#             train_loss = 0.0

#             # Reset training metrics at the start of each epoch
#             for train_metric in training_metrics:
#                 train_metric.reset()
            
#             if not args.force_regression:
#                 outputs_train_all = []
#                 ground_truth_train_all = []

#             with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as t:

#                 for X_batch, Y_batch in t:
#                     X_batch = X_batch.to(device).to(torch.float32)
#                     Y_batch = Y_batch.to(device).to(torch.float32) # ground truth

#                     if args.force_regression:
#                         Y_batch_long = Y_batch
#                     else: 
#                         Y_batch_long = torch.argmax(Y_batch, dim=1)

#                     optimizer.zero_grad()
#                     output = model(X_batch)
#                     if isinstance(output, dict):
#                         output = output['logits']
#                     loss = criterion(output, Y_batch)
#                     loss.backward()
#                     optimizer.step()

#                     if not args.force_regression:
#                         outputs_train_all.append(output)
#                         ground_truth_train_all.append(torch.argmax(Y_batch, dim=1))

#                     train_loss += loss.item()

#                     for train_metric in training_metrics:
#                         if train_metric.name != "Macro_AUROC" and train_metric.name != "Macro_AUPRC":
#                             train_metric(output, Y_batch_long)

#                     if not args.force_regression:
#                         micro_accuracy_metric = next(metric for metric in training_metrics if metric.name == "Micro_Accuracy")
#                         if t.n % 10 == 0:
#                             t.set_postfix({
#                                 "Batch Loss": loss.item(), 
#                                 "Batch Acc": micro_accuracy_metric.compute().item()
#                             })
                
#                 if not args.force_regression:
#                     outputs_train_all = torch.cat(outputs_train_all, dim=0).to(device)
#                     ground_truth_train_all = torch.cat(ground_truth_train_all, dim=0).to(device)

#             if not args.force_regression: 
#                 train_macro_auroc_metric = next(metric for metric in training_metrics if metric.name == "Macro_AUROC")
#                 train_macro_auprc_metric = next(metric for metric in training_metrics if metric.name == "Macro_AUPRC")

#                 train_macro_auroc_metric(outputs_train_all,   ground_truth_train_all)
#                 train_macro_auprc_metric(outputs_train_all, ground_truth_train_all)

#             # Validation phase
#             model.eval()
#             val_loss = 0.0

#             for val_metric in validation_metrics:
#                 val_metric.reset()

#             all_val_outputs = []
#             all_val_labels = []

#             with torch.no_grad():
            
#                 for X_batch, Y_batch in val_loader:
#                     X_batch = X_batch.to(device).to(torch.float32)
#                     Y_batch = Y_batch.to(device).to(torch.float32)
#                     if args.force_regression:
#                         Y_batch_long = Y_batch
#                     else: 
#                         Y_batch_long = torch.argmax(Y_batch, dim=1)

#                     output = model(X_batch)
#                     if isinstance(output, dict):
#                         output = output['logits']

#                     if not args.force_regression:
#                         all_val_outputs.append(output)
#                         all_val_labels.append(Y_batch_long)
                 
#                     val_loss += criterion(output, Y_batch).item()

#                     for val_metric in validation_metrics:
#                         if val_metric.name != "Macro_AUROC" and val_metric.name != "Macro_AUPRC":
#                             val_metric(output, Y_batch_long)

#             if not args.force_regression:
#                 all_val_outputs = torch.cat(all_val_outputs, dim=0)
#                 all_val_labels = torch.cat(all_val_labels, dim=0)
#                 Y_validation_long = torch.argmax(Y_validation, dim=1).to(device)
#                 true_labels = Y_validation_long.cpu().detach().numpy()
#                 test_predictions = np.argmax(all_val_outputs.cpu().detach().numpy(), axis=1)


#                 conf_matrix = confusion_matrix(true_labels, test_predictions)
#                 print("Confusion Matrix:")
#                 print(conf_matrix)

#                 val_macro_auroc_metric = next(metric for metric in validation_metrics if metric.name == "Macro_AUROC")
#                 val_macro_auprc_metric = next(metric for metric in validation_metrics if metric.name == "Macro_AUPRC")

#                 val_macro_auroc_metric(all_val_outputs, Y_validation_long)
#                 val_macro_auprc_metric(all_val_outputs, Y_validation_long)

#             # Calculate average loss and metrics
#             train_loss /= len(train_loader)
#             val_loss /= len(val_loader)

#             # Compute the metrics and store them in dictionaries (to prevent multiple calls to compute)
#             training_metrics_values = {metric.name: metric.compute() for metric in training_metrics}
#             validation_metrics_values = {metric.name: metric.compute() for metric in validation_metrics}

#             if not args.force_regression:
#                 tpr_results = ml_utils.evaluate_model_tpr_at_fpr(model, val_loader, device, num_classes)
#                 confidence_levels, proportions_above_confidence_threshold = ml_utils.evaluate_confidence_thresholding(model, val_loader, device)

#             # Print metric values
#             print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#             train_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in training_metrics_values.items())
#             print(f"Train Metrics: {train_metrics_str}")

#             val_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in validation_metrics_values.items())
#             print(f"Val Metrics: {val_metrics_str}")

#             if not args.force_regression: 
#                 for fpr, tprs in tpr_results.items():
#                     print(f"Val TPR at {fpr}: {', '.join(f'{tpr:.4f}' for tpr in tprs)}")
#                 for confidence_level, acc in confidence_levels.items():
#                     print(f"Val Accuracy at {confidence_level} confidence level: {acc:.4f}")

#             wandb.log({
#                 "train/Loss": train_loss,
#                 "train/Learning Rate": optimizer.param_groups[0]['lr'],
#                 "train/Epoch": epoch,
#                 "validation/Loss": val_loss,
#                 **{
#                     f"train/{name}": value.item() 
#                     for name, value in training_metrics_values.items() 
#                     if name != 'R2Score_RawValues'
#                 },
#                 **{
#                     f"train/R2Score_RawValues_{i+1}": v.item() 
#                     for name, value in training_metrics_values.items() 
#                     if name == 'R2Score_RawValues'
#                     for i, v in enumerate(value)
#                 },
#                 **{
#                     f"validation/{name}": value.item() 
#                     for name, value in validation_metrics_values.items() 
#                     if name != 'R2Score_RawValues'
#                 },
#                 **{
#                     f"validation/R2Score_RawValues_{i+1}": v.item() 
#                     for name, value in validation_metrics_values.items() 
#                     if name == 'R2Score_RawValues'
#                     for i, v in enumerate(value)
#                 },
#                 **({f"tpr_at_fixed_fpr/Average Val TPR at {fpr} FPR": np.mean(tprs) for fpr, tprs in tpr_results.items()} if not args.force_regression else {}),
#                 **({f"confidence_level_accuracies/Val Accuracy at {int(confidence_level*100)}% confidence": acc for confidence_level, acc in confidence_levels.items()} if not args.force_regression else {}),
#                 **({f"proportion_above_confidence_threshold/Val Proportion above {int(confidence_level*100)}% confidence": prop for confidence_level, prop in proportions_above_confidence_threshold.items()} if not args.force_regression else {}),
#             })

#         torch.save(model.state_dict(), model_filename)
#         wandb.save(f'model/modelParameters_{formatted_datetime}.pth')

#         if args.pretrain_and_finetune:
#             ### Finish the current run and start a new run for finetuning
#             ml_utils.evaluate_model_on_test_set(model, test_loader, device, numGestures, criterion, args, testing_metrics)

#             model.eval()
#             with torch.no_grad():
#                 test_predictions = []
#                 for X_batch, Y_batch in tqdm(test_loader, desc="Test Batch Loading for Confusion Matrix"):
#                     X_batch = X_batch.to(device).to(torch.float32)
#                     outputs = model(X_batch)
#                     if isinstance(outputs, dict):
#                         outputs = outputs['logits']
#                     preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
#                     test_predictions.extend(preds)

#             # Print confusion matrix before plotting
#             # Convert lists to numpy arrays
            
#             if not args.force_regression: 
#                 true_labels = np.argmax(Y_test.cpu().detach().numpy(), axis=1)
#                 test_predictions = np.array(test_predictions)

#                 # Calculate and print the confusion matrix
#                 conf_matrix = confusion_matrix(true_labels, test_predictions)
#                 print("Confusion Matrix:")
#                 print(conf_matrix)

#                 print("Classification Report:")
#                 print(classification_report(true_labels, test_predictions))
                
#                 utils.plot_confusion_matrix(np.argmax(Y_test.cpu().detach().numpy(), axis=1), np.array(test_predictions), gesture_labels, testrun_foldername, args, formatted_datetime, 'test')   

#             torch.cuda.empty_cache()  # Clear cache if needed

#             model.eval()
#             with torch.no_grad():
#                 validation_predictions = []
#                 for X_batch, Y_batch in tqdm(val_loader, desc="Validation Batch Loading for Confusion Matrix"):
#                     X_batch = X_batch.to(device).to(torch.float32)
#                     outputs = model(X_batch)
#                     if isinstance(outputs, dict):
#                         outputs = outputs['logits']
#                     preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
#                     validation_predictions.extend(preds)
#             if not args.force_regression: 
#                 utils.plot_confusion_matrix(np.argmax(Y_validation.cpu().detach().numpy(), axis=1), np.array(validation_predictions), gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')   

#             # Load training in smaller batches for memory purposes
#             torch.cuda.empty_cache()  # Clear cache if needed

#             model.eval()
#             train_loader_unshuffled = DataLoader(train_dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)
            
#             with torch.no_grad():
#                 train_predictions = []
#                 for X_batch, Y_batch in tqdm(train_loader_unshuffled, desc="Training Batch Loading for Confusion Matrix"):
#                     X_batch = X_batch.to(device).to(torch.float32)
#                     outputs = model(X_batch)
#                     if isinstance(outputs, dict):
#                             outputs = outputs['logits']
#                     preds = torch.argmax(outputs, dim=1)
#                     train_predictions.extend(preds.cpu().detach().numpy())
            
#             if not args.force_regression:
#                 utils.plot_confusion_matrix(np.argmax(Y_train.cpu().detach().numpy(), axis=1), np.array(train_predictions), gesture_labels, testrun_foldername, args, formatted_datetime, 'train')
#                 run.finish()

#             ### Initiate new logging of finetuning phase
#             run = wandb.init(name=wandb_runname+"_finetune", project=project_name) 
#             num_epochs = args.finetuning_epochs
#             # train more on fine tuning dataset
#             finetune_dataset = CustomDataset(X_train_finetuning, Y_train_finetuning, transform=resize_transform)
#             finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)

#             # Initialize metrics for finetuning training and validation
#             ft_training_metrics, ft_validation_metrics, testing_metrics = get_metrics()
    
#             for epoch in tqdm(range(num_epochs), desc="Finetuning Epoch"):
#                 model.train()
#                 train_loss = 0.0

#                 # Reset finetuning training metrics at the start of each epoch
#                 for ft_train_metric in ft_training_metrics:
#                     ft_train_metric.reset()
#                 print()

#                 with tqdm(finetune_loader, desc=f"Finetuning Epoch {epoch+1}/{num_epochs}", leave=False) as t:
#                     for X_batch, Y_batch in t:
#                         X_batch = X_batch.to(device).to(torch.float32)
#                         Y_batch = Y_batch.to(device).to(torch.float32)
#                         if args.force_regression:
#                             Y_batch_long = Y_batch
#                         else: 
#                             Y_batch_long = torch.argmax(Y_batch, dim=1)

#                         optimizer.zero_grad()
#                         output = model(X_batch)
#                         if isinstance(output, dict):
#                             output = output['logits']
#                         loss = criterion(output, Y_batch_long)
#                         loss.backward()
#                         optimizer.step()

#                         train_loss += loss.item()

#                         for ft_train_metric in ft_training_metrics:
#                             ft_train_metric(output, Y_batch_long)

#                         if not args.force_regression: 
#                             if t.n % 10 == 0:
#                                 ft_accuracy_metric = next(metric for metric in ft_training_metrics if metric.name =="Micro_Accuracy")

#                                 t.set_postfix({
#                                     "Batch Loss": loss.item(), 
#                                     "Batch Acc": ft_accuracy_metric.compute().item()
#                                 })

#                 # Finetuning Validation
#                 model.eval()
#                 val_loss = 0.0

#                 for ft_val_metric in ft_validation_metrics:
#                     ft_val_metric.reset()

#                 if not args.force_regression:
#                     all_val_outputs = []
#                     all_val_labels = []
    
#                 with torch.no_grad():
#                     for X_batch, Y_batch in val_loader:
#                         X_batch = X_batch.to(device).to(torch.float32)
#                         Y_batch = Y_batch.to(device).to(torch.float32)
#                         if args.force_regression:
#                             Y_batch_long = Y_batch
#                         else: 
#                             Y_batch_long = torch.argmax(Y_batch, dim=1)

#                         output = model(X_batch)
#                         if isinstance(output, dict):
#                             output = output['logits']
#                         val_loss += criterion(output, Y_batch).item()

#                         for ft_val_metric in ft_validation_metrics:
#                             if ft_val_metric.name != "Macro_AUROC" and ft_val_metric.name != "Macro_AUPRC":
#                                 ft_val_metric(output, Y_batch_long)

#                         if not args.force_regression:
#                             all_val_outputs.append(output)
#                             all_val_labels.append(Y_batch_long)

                
#                 if not args.force_regression:
#                     all_val_outputs = torch.cat(all_val_outputs, dim=0)
#                     all_val_labels = torch.cat(all_val_labels, dim=0)  
#                     Y_validation_long = torch.argmax(Y_validation, dim=1).to(device)

#                     true_labels = Y_validation_long.cpu().detach().numpy()
#                     test_predictions = np.argmax(all_val_outputs.cpu().detach().numpy(), axis=1)
#                     conf_matrix = confusion_matrix(true_labels, test_predictions)
#                     print("Confusion Matrix:")
#                     print(conf_matrix)

#                     finetune_val_macro_auroc_metric = next(metric for metric in ft_validation_metrics if metric.name == "Macro_AUROC")
#                     finetune_val_macro_auprc_metric = next(metric for metric in ft_validation_metrics if metric.name == "Macro_AUPRC")

#                     finetune_val_macro_auroc_metric(all_val_outputs, Y_validation_long)
#                     finetune_val_macro_auprc_metric(all_val_outputs, Y_validation_long)

#                 # Calculate average loss and metrics
#                 train_loss /= len(finetune_loader)
#                 val_loss /= len(val_loader)
#                 if not args.force_regression: 
#                     tpr_results = ml_utils.evaluate_model_tpr_at_fpr(model, val_loader, device, numGestures)
#                     fpr_results = ml_utils.evaluate_model_fpr_at_tpr(model, val_loader, device, numGestures)
#                     confidence_levels, proportions_above_confidence_threshold = ml_utils.evaluate_confidence_thresholding(model, val_loader, device)

#                 # Compute the metrics and store them in dictionaries (to prevent multiple calls to compute)
#                 ft_training_metrics_values = {ft_metric.name: ft_metric.compute() for ft_metric in ft_training_metrics}
#                 ft_validation_metrics_values = {ft_metric.name: ft_metric.compute() for ft_metric in ft_validation_metrics}

#                 # Print metric values
#                 print(f"Finetuning Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#                 ft_train_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in ft_training_metrics_values.items())
#                 print(f"Train Metrics | {ft_train_metrics_str}")

#                 ft_val_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in ft_validation_metrics_values.items())
#                 print(f"Val Metrics | {ft_val_metrics_str}")

#                 wandb.log({
#                     "train/Loss": train_loss,
#                     **{
#                         f"train/{name}": value.item() 
#                         for name, value in ft_training_metrics_values.items() 
#                         if name != 'R2Score_RawValues'
#                     },
#                     **{
#                         f"train/R2Score_RawValues_{i+1}": v.item() 
#                         for name, value in ft_training_metrics_values.items() 
#                         if name == 'R2Score_RawValues'
#                         for i, v in enumerate(value)
#                     },
#                     "train/Learning Rate": optimizer.param_groups[0]['lr'],
#                     "train/Epoch": epoch,
#                     "validation/Loss": val_loss,
#                     **{
#                         f"validation/{name}": value.item() 
#                         for name, value in ft_validation_metrics_values.items() 
#                         if name != 'R2Score_RawValues'
#                     },
#                     **{
#                         f"validation/R2Score_RawValues_{i+1}": v.item() 
#                         for name, value in ft_validation_metrics_values.items() 
#                         if name == 'R2Score_RawValues'
#                         for i, v in enumerate(value)
#                     },

#                     **({f"tpr_at_fixed_fpr/Average Val TPR at {fpr} FPR": np.mean(tprs) for fpr, tprs in tpr_results.items()} if not args.force_regression else {}),
#                     **({f"fpr_at_fixed_tpr/Average Val FPR at {tpr} TPR": np.mean(fprs) for tpr, fprs in fpr_results.items()} if not args.force_regression else {}),
#                     **({f"confidence_level_accuracies/Val Accuracy at {int(confidence_level*100)}% confidence": acc for confidence_level, acc in confidence_levels.items()} if not args.force_regression else {}),
#                     **({f"proportion_above_confidence_threshold/Val Proportion above {int(confidence_level*100)}% confidence": prop for confidence_level, prop in proportions_above_confidence_threshold.items()} if not args.force_regression else {})
#                 })
            
#             torch.save(model.state_dict(), model_filename)
#             wandb.save(f'model/modelParameters_{formatted_datetime}.pth')

#             # Evaluate the model on the test set
#             ml_utils.evaluate_model_on_test_set(model, test_loader, device, numGestures, criterion, args, testing_metrics)

#         torch.cuda.empty_cache()  # Clear cache if needed

#         model.eval()
#         with torch.no_grad():
#             # assert args.force_regression == False
#             test_predictions = []
#             for X_batch, Y_batch in tqdm(test_loader, desc="Test Batch Loading for Confusion Matrix"):
#                 X_batch = X_batch.to(device).to(torch.float32)
#                 outputs = model(X_batch)
#                 if isinstance(outputs, dict):
#                     outputs = outputs['logits']
#                 preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
#                 test_predictions.extend(preds)

#         # Print confusion matrix before plotting
#         # Convert lists to numpy arrays
#         if not args.force_regression: 
#             true_labels = np.argmax(Y_test.cpu().detach().numpy(), axis=1)
#             test_predictions = np.array(test_predictions)

#             # Calculate and print the confusion matrix
#             conf_matrix = confusion_matrix(true_labels, test_predictions)
#             print("Confusion Matrix:")
#             print(conf_matrix)

#             print("Classification Report:")
#             print(classification_report(true_labels, test_predictions))
                
#             utils.plot_confusion_matrix(np.argmax(Y_test.cpu().detach().numpy(), axis=1), np.array(test_predictions), gesture_labels, testrun_foldername, args, formatted_datetime, 'test')   

#         torch.cuda.empty_cache()  # Clear cache if needed

#         model.eval()
#         with torch.no_grad():
#             validation_predictions = []
#             for X_batch, Y_batch in tqdm(val_loader, desc="Validation Batch Loading for Confusion Matrix"):
#                 X_batch = X_batch.to(device).to(torch.float32)
#                 outputs = model(X_batch)
#                 if isinstance(outputs, dict):
#                     outputs = outputs['logits']
#                 preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
#                 validation_predictions.extend(preds)

#         if not args.force_regression:
#             utils.plot_confusion_matrix(np.argmax(Y_validation.cpu().detach().numpy(), axis=1), np.array(validation_predictions), gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')   

#         # Load training in smaller batches for memory purposes
#         torch.cuda.empty_cache()  # Clear cache if needed

#         model.eval()
#         train_loader_unshuffled = DataLoader(train_dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=utils.seed_worker, pin_memory=True)
#         with torch.no_grad():
#             train_predictions = []
#             for X_batch, Y_batch in tqdm(train_loader_unshuffled, desc="Training Batch Loading for Confusion Matrix"):
#                 X_batch = X_batch.to(device).to(torch.float32)
#                 outputs = model(X_batch)
#                 if isinstance(outputs, dict):
#                         outputs = outputs['logits']
#                 preds = torch.argmax(outputs, dim=1)
#                 train_predictions.extend(preds.cpu().detach().numpy())

#         if not args.force_regression: 
#             utils.plot_confusion_matrix(np.argmax(Y_train.cpu().detach().numpy(), axis=1), np.array(train_predictions), gesture_labels, testrun_foldername, args, formatted_datetime, 'train')
            
#     run.finish()