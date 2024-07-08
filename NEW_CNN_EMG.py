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

args = None # make args a global variable
utils = None

# need to make args and utils a global variable 

# class CNN_EMG():

#     def __init__(self):
#         self.X = X_Data()
#         self.Y = Y_Data()
#         self.labels = Labels_Data()

#     # def parse args and initialize here

#         self.X.load_data(exercises) 
#         self.Y.load_data(exercises)
#         self.labels.load_data(exercises)

#         assert len(self.X.data[-1]) == len(self.Y.data[-1]), "Number of trials for X and Y do not match."
#         assert len(self.Y.data[-1]) == len(self.labels.data[-1]), "Number of trials for Y and Labels do not match."


# Classes for splitting data

class Data_Split_Strategy():
    """
    Serves as a wrapper to hold X, Y, and label objects. 
    """

    def __init__(self, combined):
        assert(isinstance(combined, Combined))

        global args
        global utils

        self.X = combined.X
        self.Y = combined.Y
        self.label = combined.label

        self.leaveOut = int(args.leftout_subject) # LOSO helper variable

    def convert_to_16_tensors(self, set_to_convert):
        self.X.convert_to_16_tensors(set_to_convert)
        self.Y.convert_to_16_tensors(set_to_convert)
        self.label.convert_to_16_tensors(set_to_convert)

    def concatenate_sessions(self, set_to_assign, set_to_concat):
        self.X.concatenate_sessions(set_to_assign, set_to_concat)
        self.Y.concatenate_sessions(set_to_assign, set_to_concat)
        self.label.concatenate_sessions(set_to_assign, set_to_concat)

    def del_data(self):
        self.X.del_data()
        self.Y.del_data()
        self.label.del_data()

    def print_set_shapes(self):
        self.X.print_set_shapes()
        self.Y.print_set_shapes()
        self.label.print_set_shapes()

class Split_Leave_N_Subjects_Out_Randomly(Data_Split_Strategy):
    """
    Splits such that validate uses left out subjects (leaveOutIndices) and train uses the rest.
    """
    
    # Helper Routines
    def validation_from_leave_out_indices(self):
        """
        Create validation sets by taking leaveOutIndices from data. 
        """
        self.X.validation_from_leave_out_indices()
        self.Y.validation_from_leave_out_indices()
        self.label.validation_from_leave_out_indices()

    def convert_datasets_to_tensors(self):
        """
        Convert validation and training sets to tensors.
        """
        super().convert_to_16_tensors("validation")
        super().convert_to_16_tensors("train")

    def split(self):
        """
        Split data into training and validation sets.

        This method creates validation sets from leave out subjects (leaveOutIndices) and training sets from the rest of the subjects, converts the data to tensors, and deletes the original data. 
        """

        self.validation_from_leave_out_indices()
        self.train_from_non_leave_out_indices()
        self.convert_datasets_to_tensors()
        self.del_data()
        
class Split_Held_Out_Test(Data_Split_Strategy):
    """
    Splits such that train uses train_indices, validation uses validation_indices, and test uses test_indices.
    """

    def combine_data(self):
        """
        Combines data across the 0th axis.
        """

        X_combined_data = np.concatenate([np.array(i) for i in self.X.data], axis=0, dtype=np.float16)
        Y_combined_data = np.concatenate([np.array(i) for i in self.Y.data], axis=0, dtype=np.float16)
        label_combined_data = np.concatenate([np.array(i) for i in self.label.data], axis=0, dtype=np.float16)

        return X_combined_data, Y_combined_data, label_combined_data

    def train_from_train_indices(self, X_combined_data, Y_combined_data, label_combined_data):
        """
        Create train set by taking train indices from data.
        """

        self.X.train_from_train_indices(X_combined_data)
        self.Y.train_from_train_indices(Y_combined_data)
        self.label.train_from_train_indices(label_combined_data)
    
    def validation_from_validation_indices(self, X_combined_data, Y_combined_data, label_combined_data):
        """
        Create validation set by taking validation indices from data.
        """

        self.X.validation_from_validation_indices(X_combined_data)
        self.Y.validation_from_validation_indices(Y_combined_data)
        self.label.validation_from_validation_indices(label_combined_data)

    def convert_datasets_to_tensors(self):
        """
        Covert train, validation, and test to tensors.
        """
        super().convert_to_16_tensors("train")
        super().convert_to_16_tensors("validation")
        super().convert_to_16_tensors("test")

    def del_combined_data(self):
        """
        Delete combined_data variable.
        """
        self.X.del_combined_data()
        self.Y.del_combined_data()
        self.label.del_combined_data()
    
    def split(self):
        """Train using train_indices. Split validation indices into validation and test using labels to stratify.""" 

        X_combined_data, Y_combined_data, label_combined_data = self.combine_data()
        self.train_from_train_indices(X_combined_data, Y_combined_data, label_combined_data)
        self.validation_from_validation_indices(X_combined_data, Y_combined_data, label_combined_data)
       
        # Split validation into validation and test 50/50
        self.X.validation, self.X.test, \
        self.Y.validation, self.Y.test, \
        self.label.validation, self.label.test \
        = model_selection.train_test_split(
            self.X.validation, 
            self.Y.validation, 
            self.label.validation, 
            test_size=0.5, 
            stratify=self.label.validation
        )

        self.convert_datasets_to_tensors()

        del X_combined_data
        del Y_combined_data
        del label_combined_data
      
        self.del_data()

class Split_Leave_One_Session_Out(Data_Split_Strategy):
    """ 
    Splits data based on leaving one session of leaveOut subject out. 
    """

    # Helper Routines

    def concatenate_pretrain_and_finetune(self):
        """
        Concatenate sets across the 0th axis (sessions) for pretrain, finetune, and if enabled, pretrain_unlabeled and finetune_unlabeled. 
        """

        super().concatenate_sessions(set_to_assign="pretrain", set_to_concat="pretrain")
        super().concatenate_sessions(set_to_assign="finetune", set_to_concat="finetune")

        if args.proportion_unlabeled_data_from_training_subjects:
            super().concatenate_sessions(set_to_assign="pretrain_unlabeled", set_to_concat="pretrain_unlabeled_list")
        if proportion_unlabeled_data_from_leftout_subject or args.load_unlabeled_data_flexwearhd:
            super().concatenate_sessions(set_to_assign="finetune_unlabeled", set_to_concat="finetune_unlabeled_list")

    def append_to_pretrain(self, X_new_data, Y_new_data, label_new_data):
        self.X.append_to_pretrain(X_new_data)
        self.Y.append_to_pretrain(Y_new_data)
        self.label.append_to_pretrain(label_new_data)

    def append_to_pretrain_unlabeled_list(self, X_new_data, Y_new_data, label_new_data):

        self.X.append_to_pretrain_unlabeled_list(X_new_data)
        self.Y.append_to_pretrain_unlabeled_list(Y_new_data)
        self.label.append_to_pretrain_unlabeled_list(label_new_data)

    def append_to_finetune(self, X_new_data, Y_new_data, label_new_data):   

        self.X.append_to_finetune(X_new_data)
        self.Y.append_to_finetune(Y_new_data)
        self.label.append_to_finetune(label_new_data)

    def append_to_finetune_unlabeled_list(self, X_new_data, Y_new_data, label_new_data, flexwear_unlabeled_data=False):

        if flexwear_unlabeled_data:
            self.X.append_flexwear_unlabeled_to_finetune_unlabeled_list(X_new_data)
            self.Y.append_flexwear_unlabeled_to_finetune_unlabeled_list(Y_new_data)
            self.label.append_flexwear_unlabeled_to_finetune_unlabeled_list(label_new_data)

        self.X.append_to_finetune_unlabeled_list(X_new_data)
        self.Y.append_to_finetune_unlabeled_list(Y_new_data)
        self.label.append_to_finetune_unlabeled_list(label_new_data)

    def pretrain_from_non_leftout(self, X_train_temp, Y_train_temp, label_train_temp):
        """
        Creates pretrain and pretrain_unlabeled sets from non left out subject's data.

        If proportion_data_from_training_subjects < 1.0, only takes a proportion of the data from the training subjects. If proportion_unlabeled_data_from_training_subjects > 0, splits the training set into labeled and unlabeled data.

        Args:
            X_train_temp: EMG data for the current subject and session
            Y_train_temp: Label/Force data for the current subject and session
            label_train_temp: Label data for the current subject and session

        Sets:
            self.pretrain: Non left out subject's labeled data
            self.pretrain_unlabeled_list: Non left out subject's unlabeled data (if proportion_unlabeled_data_from_training_subjects > 0)
        """

        if args.proportion_data_from_training_subjects<1.0:
            X_train_temp, _, \
            Y_train_temp, _, \
            label_train_temp, _ \
            = tts.train_test_split(
                X_train_temp, 
                Y_train_temp, 
                train_size=args.proportion_data_from_training_subjects, 
                stratify=label_train_temp, 
                random_state=args.seed, 
                shuffle=(not args.train_test_split_for_time_series)
            )
            
        if args.proportion_unlabeled_data_from_training_subjects>0:
            X_pretrain_labeled, X_pretrain_unlabeled, \
            Y_pretrain_labeled, Y_pretrain_unlabeled, \
            label_pretrain_labeled, label_pretrain_unlabeled \
            = tts.train_test_split(
                X_train_temp, 
                Y_train_temp, 
                train_size=1-args.proportion_unlabeled_data_from_training_subjects, 
                stratify=label_train_temp, 
                random_state=args.seed, 
                shuffle=(not args.train_test_split_for_time_series)
            )

            self.append_to_pretrain(X_pretrain_labeled, Y_pretrain_labeled, label_pretrain_labeled)

            self.append_to_pretrain_unlabeled_list(X_pretrain_unlabeled, Y_pretrain_unlabeled, label_pretrain_unlabeled)

        else:
            self.append_to_pretrain(X_train_temp, Y_train_temp, label_train_temp)

    def finetune_from_leftout_first_n(self, X_train_temp, Y_train_temp, label_train_temp):

        """ 
        Creates finetune and finetune_unlabeled sets from left out subject's first n sessions data.

        If proportion_unlabeld_data_from_leftout_subject > 0, splits the first n sessions data into labeled and unlabeled data.

        Args:
            X_train_temp: EMG data for the current subject and session
            Y_train_temp: Label/Force data for the current subject and session
            label_train_temp: Label data for the current subject and session

        Sets:
            self.finetune: Left out subject's first n sessions labeled data
            self.finetune_unlabeled_list: Left out subject's first n sessions unlabeled data (if proportion_unlabeled_data_from_leftout_subject > 0)    
        
        """

        if args.proportion_unlabeled_data_from_leftout_subject>0:

            X_finetune_labeled, X_finetune_unlabeled, \
            Y_finetune_labeled, Y_finetune_unlabeled, \
            label_finetune_labeled, label_finetune_unlabeled \
            = tts.train_test_split(
                X_train_temp, 
                Y_train_temp, 
                train_size=1-args.proportion_unlabeled_data_from_leftout_subject, 
                stratify=label_train_temp, 
                random_state=args.seed, 
                shuffle=(not args.train_test_split_for_time_series)
            )

            self.append_to_finetune(X_finetune_labeled, Y_finetune_labeled, label_finetune_labeled)
            self.append_to_finetune_unlabeled_list(X_finetune_unlabeled, Y_finetune_unlabeled, label_finetune_unlabeled)

        else:
            self.append_to_finetune(X_train_temp, Y_train_temp, label_train_temp)

    def validation_from_leftout_last(left_out_subject_last_session_index):   
        self.X.validation_from_leftout_last(left_out_subject_last_session_index)
        self.Y.validation_from_leftout_last(left_out_subject_last_session_index)
        self.label.validation_from_leftout_last(left_out_subject_last_session_index)

    def train_from_train_and_finetuning(self):
        """ If not args.pretrain_and_finetune, combine train and finetune sets into train. """

        self.X.train_from_train_and_finetuning()
        self.Y.train_from_train_and_finetuning()
        self.label.train_from_train_and_finetuning()

    def train_from_train_and(self):
        """ 
        If not args.turn_on_unlabeled_domain_adaptation, train using train and finetuning. """

        self.X.train_for_no_finetune()
        self.Y.train_for_no_finetune()
        self.label.train_for_no_finetune()
    
    def train_from_train_finetuning(self):
        """ 
        If not args.unlabeled_domain_adaptation, train from train_finetuning. """

        self.X.train_from_train_finetuning()
        self.Y.train_from_train_finetuning()
        self.label.train_from_train_finetuning()

    def create_pretrain_and_finetune(self, left_out_subject_last_session_index, left_out_subject_first_n_sessions_indices):
        """
        Creates pretrain, pretrain_unlabeled, finetune, finetune_unlabeled, and validation sets.

        Args:
            left_out_subject_last_session_index: index of leftout subject's last session
            left_out_subject_first_n_sessions_indices: indices of leftout subject's first n sessions

        Sets:
            pretrain: non left out subject's labeled
            pretrain_unlabeled: non left out subject's unlabeled (if proportion_unlabeled_data_from_training_subjects > 0)
            finetune: first n sessions of leftout subject labeled
            finetune_unlabeled: first n sessions of leftout subject unlabeled (if proportion_unlabeled_data_from_leftout_subject > 0)
            validation: last session of leftout subject
        """
        for i in range(utils.num_sessions*utils.num_subjects):

            X_train_temp = self.X.data[i]
            Y_train_temp = self.Y.data[i]
            label_train_temp = self.label.data[i]

            if i != left_out_subject_last_session_index and i not in left_out_subject_first_n_sessions_indices:
                self.pretrain_from_non_leftout(X_train_temp, Y_train_temp, label_train_temp)

            elif i in left_out_subject_first_n_sessions_indices:
                self.finetune_from_leftout_first_n(X_train_temp, Y_train_temp, label_train_temp)

        if args.load_unlabeled_data_flexwearhd:
            flexwear_unlabeled_data = self.X.flexwear_unlabeled_data
            self.append_to_finetune_unlabeled_list(flexwear_unlabeled_data, flexwear_unlabeled_data, flexwear_unlabeled_data, flexear_unlabeled_data=True)

        self.validation_from_leftout_last(left_out_subject_last_session_index)
    
    def reassign_sets(self):
        """
        Reassignes the temporary variables (pretrain, finetune, validation) to the train, train_finetuning, and validation sets (and their unlabeled counterparts). 
        """
        self.X.reassign_sets()
        self.Y.reassign_sets()
        self.label.reassign_sets()



    def convert_datasets_to_tensors(self):
        """
        Converts train, train_finetuning, validation, and if enabeled train_unlabeled and train_finetuning_unlabeled to 16 bit tensors.
        """
        super().convert_to_16_tensors("train")
        super().convert_to_16_tensors("train_finetuning")
        super().convert_to_16_tensors("validation")
        if args.proportion_unlabeled_data_from_training_subjects:
            super().convert_to_16_tensors("train_unlabeled")
        if args.proportion_unlabeled_data_from_leftout_subject or args.load_unlabeled_data_flexwearhd:
            super().convert_to_16_tensors("train_finetuning_unlabeled") 

    def finetuning_if_no_proportion(self):
        """
        If no proportion of unlabeled data from left out subject is to be taken, finetune should be only labeled (which is all the data) and there is no finetune_unlabeled.
        """
        self.X.finetuning_if_no_proportion()
        self.Y.finetuning_if_no_proportion()
        self.label.finetuning_if_no_proportion()

    def train_from_train_and_finetuning(self):
        """
        If not doing pretrain and finetune, train should use all of the data. (Concatenate train and finetune)
        """

        self.X.concatenate_sessions(set_to_assign="train", set_to_concat="train_finetuning")
        self.Y.concatenate_sessions(set_to_assign="train", set_to_concat="train_finetuning")
        self.label.concatenate_sessions(set_to_assign="train", set_to_concat="train_finetuning")

    def split(): 
        """ 
        Trains on non leftout subjects, finetunes with leftout subject's first n sessions, and validates with leftout subject's last session.
        
        Sets:
            train: non leftout subject's labeled data
            train_unlabeled: non leftout subject's unlabeled data (if unlabeled domain adaptation)
            train_finetuning: leftout subject's first n sessions labeled data
            train_finetuning_unlabeled: leftout subject's first n sessions unlabeled data (if unlabeled domain adaptation)
            validation: leftout subject's last session data

        """

        # Split leave out subject's data into first n sessions and last session
        left_out_subject_last_session_index = (utils.num_sessions-1) * utils.num_subjects + self.leaveOut-1
        left_out_subject_first_n_sessions_indices = [i for i in range(utils.num_sessions * utils.num_subjects) if i % utils.num_subjects == (self.leaveOut-1) and i != left_out_subject_last_session_index]

        print("left_out_subject_last_session_index:", left_out_subject_last_session_index)
        print("left_out_subject_first_n_sessions_indices:", left_out_subject_first_n_sessions_indices)

        self.create_pretrain_and_finetune(left_out_subject_last_session_index, left_out_subject_first_n_sessions_indices)
        self.concatenate_pretrain_and_finetune()
        self.reassign_sets()
        self.convert_datasets_to_tensors()
        
        # Undo any excess splitting
        if args.turn_on_unlabeled_domain_adaptation and not (args.proportion_unlabeled_data_from_leftout_subject > 0):
            self.finetuning_if_no_proportion()
        elif utils.num_subjects == 1:
            self.train_from_train_finetuning()

        if not args.pretrain_and_finetune:
            self.train_from_train_and_finetuning()

        self.del_data
        self.print_set_shapes()

class Split_Leave_One_Subject_Out(Data_Split_Strategy):

    def append_to_train_unlabeled_list(self, X_new_data, Y_new_data, label_new_data):
        self.X.append_to_train_unlabeled_list(X_new_data)
        self.Y.append_to_train_unlabeled_list(Y_new_data)
        self.label.append_to_train_unlabeled_list(label_new_data)

    def convert_datasets(self):
        super().concatenate_sessions(set_to_assign="train", set_to_concat="train_list")
        super().convert_to_16_tensors(set_to_convert="train")
        
        if args.proportion_unlabeled_data_from_training_subjects>0:
            super().concatenate_sessions(set_to_assign="train_unlabeled", set_to_concat="train_unlabeled_list")
            super().convert_to_16_tensors(set_to_convert="train_unlabeled")

        super().convert_to_16_tensors(set_to_convert="validation")

    def concatenate_to_train(self, X_new_data, Y_new_data, label_new_data):
        self.X.concatenate_to_train(X_new_data)
        self.Y.concatenate_to_train(Y_new_data)
        self.label.concatenate_to_train(label_new_data) 


    def train_from_non_left_out(self):

        for i in range(len(data)):
            if i == self.leaveOut-1:
                continue

            X_train_temp = np.array(self.X.data[i])
            Y_train_temp = np.array(self.Y.data[i])
            label_train_temp = np.array(self.label.data[i])

            if args.reduce_training_data_size:
                proportion_to_keep = reduced_size_per_subject / X_train_temp.shape[0]
                X_train_temp, _, \
                Y_train_temp, _, \
                label_train_temp, _ \
                = model_selection.train_test_split(
                    X_train_temp, 
                    Y_train_temp, 
                    train_size=proportion_to_keep, 
                    stratify=label_train_temp, 
                    random_state=args.seed, 
                    shuffle=(not args.train_test_split_for_time_series)
                )

            if args.proportion_data_from_training_subjects < 1.0:
                X_train_temp, _, \
                Y_train_temp, _, \
                label_train_temp, _ \
                = tts.train_test_split(
                    X_train_temp, 
                    Y_train_temp, 
                    train_size=args.proportion_data_from_training_subjects, 
                    stratify=label_train_temp, 
                    random_state=args.seed, 
                    shuffle=(not args.train_test_split_for_time_series)
                )
 
            if args.proportion_unlabeled_data_from_training_subjects>0:
                X_train_labeled, X_train_unlabeled, \
                Y_train_labeled, Y_train_unlabeled, \
                label_train_labeled, label_train_unlabeled = tts.train_test_split(
                    X_train_temp, 
                    Y_train_temp, 
                    train_size=1-args.proportion_unlabeled_data_from_training_subjects, 
                    stratify=label_train_temp, 
                    random_state=args.seed, 
                    shuffle=(not args.train_test_split_for_time_series)
                )

                self.append_to_train_list(X_train_labeled, Y_train_labeled, label_train_labeled)

                self.append_to_train_unlabeled_list(X_train_unlabeled, Y_train_unlabeled, label_train_unlabeled)

            else:
                self.append_to_train_list(X_train_temp, Y_train_temp, label_train_temp)

    def validation_from_left_out(self):
        X.validation_from_leave_out_subject()
        Y.validation_from_leave_out_subject()
        label.validation_from_leave_out_subject()


    def train_finetuning_from_set(self, X_new_data, Y_new_data, label_new_data):
        self.X.train_finetuning_from_set(X_new_data)
        self.Y.train_finetuning_from_set(Y_new_data)
        self.label.train_finetuning_from_set(label_new_data)




    def split_left_out(self): # L 1210 - 1344 
        """
        If doing transfer learning, splits left out subject's data into train_partial_leftout_subject and validation_partial_leftout_subject sets.
        """

        ## STEP ONE: SPLITS THEM INTO TRAIN_LABELED

        assert args.transfer_learning, "Transfer learning must be turned on to split left out subject's data."

        proportion_to_keep_of_leftout_subject_for_training = args.proportion_transfer_learning_from_leftout_subject
        
        proportion_unlabeled_of_proportion_to_keep_of_leftout = args.proportion_unlabeled_data_from_leftout_subject
        

        # SSplit leftout validation into train and validation
        if proportion_to_keep_of_leftout_subject_for_training>0.0:
            X_train_partial_leftout_subject, X_validation_partial_leftout_subject, \
            Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject, \
            label_train_partial_leftout_subject, label_validation_partial_leftout_subject = \
                tts.train_test_split(
                    X_validation, 
                    Y_validation, 
                    train_size=proportion_to_keep_of_leftout_subject_for_training, 
                    stratify=label_validation, 
                    random_state=args.seed, 
                    shuffle=(not args.train_test_splt_for_time_series), 
                    force_regression=args.force_regression
                )

        # Otherwise validate with all of left out subject's data
        else:
            X_validation_partial_leftout_subject = X_validation
            Y_validation_partial_leftout_subject = Y_validation
            label_validation_partial_leftout_subject = label_validation

            X_train_partial_leftout_subject = torch.tensor([])
            Y_train_partial_leftout_subject = torch.tensor([])
            label_train_partial_leftout_subject = torch.tensor([])


        # If unlabeled domain adaptation, split the training data into labeled and unlabeled
        if args.turn_on_unlabeled_domain_adaptation and proportion_unlabeled_of_proportion_to_keep_of_leftout>0:

            X_train_labeled_partial_leftout_subject, X_train_unlabeled_partial_leftout_subject, \
            Y_train_labeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject, \
            label_train_labeled_partial_leftout_subject, label_train_unlabeled_partial_leftout_subject = \
                tts.train_test_split(
                    X_train_partial_leftout_subject, 
                    Y_train_partial_leftout_subject, 
                    train_size=1-proportion_unlabeled_of_proportion_to_keep_of_leftout, 
                    stratify=label_train_partial_leftout_subject, 
                    random_state=args.seed, 
                    shuffle=(not args.train_test_split_for_time_series), 
                    force_regression=args.force_regression
                )


        # Flexwear unlabeled domain adaptatoin
        if args.load_unlabeled_data_flexwearhd:
            if proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
                X_train_unlabeled_partial_leftout_subject = np.concatenate([X_train_unlabeled_partial_leftout_subject, self.X.flexwear_unlabeled_data], axis=0)
                Y_train_unlabeled_partial_leftout_subject = np.concatenate([Y_train_unlabeled_partial_leftout_subject, np.zeros((self.X.flexwear_unlabeled_data.shape[0], utils.numGestures))], axis=0)
                label_train_unlabeled_partial_leftout_subject = Y_train_unlabeled_partial_leftout_subject

            else:
                X_train_unlabeled_partial_leftout_subject = self.X.flexwear_unlabeled_data
                Y_train_unlabeled_partial_leftout_subject = np.zeros((self.X.flexwear_unlabeled_data.shape[0], utils.numGestures))
                label_train_unlabeled_partial_leftout_subject = Y_train_unlabeled_partial_leftout_subject


        # Add the partial from leftout subject to train/finetune
        if not args.turn_on_unlabeled_domain_adaptation:
            # Append the partial validation data to the training data
            if proportion_to_keep_of_leftout_subject_for_training>0:
                
                if not args.pretrain_and_finetune:
                    self.concatenate_to_train(X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject)
            
                else:
                    self.train_finetuning_from_set(X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject)

                    
        else: # unlabeled domain adaptation
            if proportion_unlabeled_of_training_subjects>0:
                # creates copies
                X_train = torch.tensor(X_train)
                Y_train = torch.tensor(Y_train)
                label_train = torch.tensor(label_train)
                X_train_unlabeled = torch.tensor(X_train_unlabeled)
                Y_train_unlabeled = torch.tensor(Y_train_unlabeled)
                label_train_unlabeled = torch.tensor(label_train_unlabeled)

            if proportion_unlabeled_of_proportion_to_keep_of_leftout>0 or args.load_unlabeled_data_flexwearhd:
                if proportion_unlabeled_of_proportion_to_keep_of_leftout==0:
                    X_train_labeled_partial_leftout_subject = X_train_partial_leftout_subject
                    Y_train_labeled_partial_leftout_subject = Y_train_partial_leftout_subject
                    label_train_labeled_partial_leftout_subject = label_train_partial_leftout_subject
                if not args.pretrain_and_finetune:
                    X_train = torch.tensor(np.concatenate((X_train, X_train_labeled_partial_leftout_subject), axis=0))
                    Y_train = torch.tensor(np.concatenate((Y_train, Y_train_labeled_partial_leftout_subject), axis=0))
                    label_train = torch.tensor(np.concatenate((label_train, label_train_labeled_partial_leftout_subject), axis=0))
                    X_train_unlabeled = torch.tensor(np.concatenate((X_train_unlabeled, X_train_unlabeled_partial_leftout_subject), axis=0))
                    Y_train_unlabeled = torch.tensor(np.concatenate((Y_train_unlabeled, Y_train_unlabeled_partial_leftout_subject), axis=0))
                    label_train_unlabeled = torch.tensor(np.concatenate((label_train_unlabeled, label_train_unlabeled_partial_leftout_subject), axis=0))
                else:
                    X_train_finetuning = torch.tensor(X_train_labeled_partial_leftout_subject)
                    Y_train_finetuning = torch.tensor(Y_train_labeled_partial_leftout_subject)
                    label_train_finetuning = torch.tensor(label_train_labeled_partial_leftout_subject)
                    X_train_finetuning_unlabeled = torch.tensor(X_train_unlabeled_partial_leftout_subject)
                    Y_train_finetuning_unlabeled = torch.tensor(Y_train_unlabeled_partial_leftout_subject)
                    label_train_finetuning_unlabeled = torch.tensor(label_train_unlabeled_partial_leftout_subject)
            else:
                if proportion_to_keep_of_leftout_subject_for_training>0:
                    if not args.pretrain_and_finetune:
                        X_train = torch.tensor(np.concatenate((X_train, X_train_partial_leftout_subject), axis=0))
                        Y_train = torch.tensor(np.concatenate((Y_train, Y_train_partial_leftout_subject), axis=0))
                        label_train = torch.tensor(np.concatenate((label_train, label_train_partial_leftout_subject), axis=0))
                    else: 
                        X_train_finetuning = torch.tensor(X_train_partial_leftout_subject)
                        Y_train_finetuning = torch.tensor(Y_train_partial_leftout_subject)
                        label_train_finetuning = torch.tensor(label_train_partial_leftout_subject)

        # Update the validation data
        X_train = torch.tensor(X_train).to(torch.float16)
        Y_train = torch.tensor(Y_train).to(torch.float16)
        label_train = torch.tensor(label_train).to(torch.float16)
        X_validation = torch.tensor(X_validation_partial_leftout_subject).to(torch.float16)
        Y_validation = torch.tensor(Y_validation_partial_leftout_subject).to(torch.float16)
        label_validation = torch.tensor(label_validation_partial_leftout_subject).to(torch.float16)
        
        del X_train_partial_leftout_subject, X_validation_partial_leftout_subject, Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject, label_train_partial_leftout_subject, label_validation_partial_leftout_subject



            
    def split(self):

        # I feel like it honestly might be better to just move the split_left_out to be a helper function in validation_from_leave_out subject. the only problem is that it relies on convert_datasets but i guess we can just move it out 

        if args.reduce_training_data_size:
            reduced_size_per_subject = args.reduced_training_data_size // (utils.num_subjects-1)

        
        self.train_from_non_leave_out_subject()
        self.validation_from_leave_out_subject()
        self.convert_datasets()

        if args.transfer_learning:
            self.split_left_out()

        super().print_set_shapes()








            




    


class Combined():
    """Wrapper class that repeats a given functions for all the data. 
    """
    def __init(self, x_obj, y_obj, label_obj):

        global args
        global utils 

        self.X = x_obj
        self.Y = y_obj
        self.label = label_obj

    def load_data(self):
        self.X.load_data()
        self.Y.load_data()
        self.label.load_data()

        assert len(self.X.data[-1]) == len(self.Y.data[-1]), "Number of trials for X and Y do not match."
        assert len(self.Y.data[-1]) == len(self.label.data[-1]), "Number of trials for Y and Labels do not match."

    def scaler_normalize_emg(self):
        # Need to share indices/split across the board since randomly generated.
        train_indices, validation_indices, leaveOutIndices = self.X.scaler_normalize_emg()

        if args.leave_n_subjects_out_randomly != 0 and (not args.turn_off_scaler_normalization and not (args.target_normalize > 0)):
            self.X.leaveOutIndices = leaveOutIndices
            self.Y.leaveOutIndices = leaveOutIndices
            self.label.leaveOutIndices = leaveOutIndices
        
        elif not args.held_out_test:
            self.X.train_indices = train_indices
            self.X.validation_indices = validation_indices

            self.Y.train_indices = train_indices
            self.Y.validation_indices = validation_indices

            self.label.train_indices = train_indices
            self.label.validation_indices = validation_indices

    def load_images(self, base_foldername_zarr):
        flexwear_unlabeled_data = self.X.load_images(base_foldername_zarr)

        if flexwear_unlabeled_data:
            assert args.load_unlabeled_data_flexwearhd, "Unlabeled data should only be returned if load_unlabeled_data_flexwearhd is turned on."
            
    def create_train_test_sets(self):

        if args.leave_n_subjects_out_randomly:
            splitter = Split_Leave_N_Subjects_Out_Randomly(self)
        elif args.held_out_test:
            splitter = Split_Held_Out_Test(self)
        elif args.leave_one_session_out:
            splitter = Split_Leave_One_Session_Out(self)
        elif args.leave_one_subject_out:
            # splitter = Split_Leave_One_Subject_Out(self)
        elif utils.num_subjects == 1:
            # splitter = Split_Single_Subject(self)
        else:
            raise ValueError("Please specify the type of test you want to run")
        
        splitter.split()

    

class Data():
    """
    Class to load the data for the CNN. Should pull in the relevant data, save in zarr folder, and then create train, validation, and test datasets.

    Args:
        train: non leftout subject's labeled data
        train_unlabeled: 
        train_finetuning: leftout subject's first n sessions labeled data
        train_finetuning_unlabeled: leftout subject's first n sessions unlabeled data (if unlabeled domain adaptation)
        validation: leftout subject's last session data

    """

    def __init__(self, field):

        global args 
        global utils

        self.field = field
        self.data = None

        self.scaler = None # if not args.turn_off_scaler_normalization

        # Leave_N_Subjects_Out_Randomly Helper Variables
        self.leaveOutIndices = None 

        # LOSO Helper Variable
        self.leaveOut = int(args.leftout_subject)

        # Leave_One_Session_Out Helper Variables
        self.pretrain = []
        if args.proportion_unlabeled_data_from_training_subjects:
            self.pretrain_unlabeled_list = []
        self.finetune = []
        if args.proportion_unlabeled_data_from_leftout_subject:
            self.finetune_unlabeled_list= []

        # Leave_One_Subject_Out Helper Variables
        self.train_list = []
        if args.proportion_unlabeled_from_training_subjects > 0:
            self.train_unlabeled_list = []

        self.train = None
        self.train_unlabeled = None
        self.train_finetuning = None
        self.train_finetuning_unlabeled = None
        self.validation = None
        
    def load_data(self):
        raise NotImplementedError("Subclass must implement this method")

    def process_ninapro(self, data):
        """ Appends exercise sets together and adds dimensions to data if necessary for Ninapro dataset values. 

        Returns:
            Torch with all data concatenated for each subject for each exercise.
        
        """

        # Remove Subject 10 who is missing most of exercise 3 data in DB3 
        if args.force_regression and args.dataset == "ninapro-db3":
            MISSING_SUBJECT = 10
            utils.num_subjects -= 1 
            del data[0][MISSING_SUBJECT-1]

        new_data = []
        numGestures = 0

        for subject in range(utils.num_subjects):
            subject_data = []

            # Store data for this subject across all exercises
            for exercise_set in range(len(emg)):
                subject_data.append(data[exercise_set][subject])

                if args.force_regression and self.field == "Y": 
                    # Take the first gesture reading out of the 500 to reduce dimension
                    subject_data.append(data[exercise_set][subject][:, :, 0])
                else:
                    subject_data.append(data[exercise_set][subject])

            concatenated_data = self.concat_across_exercises(subject_data)
            new_data.append(concatenated_data)

        data = [torch.from_numpy(data_np) for data_np in new_data]
        return data       

    # NOTE: HELPER FUNCTIONS SHOULD ONLY EVER ASSIGN SPLIT SET VALUES. EVERYTHING ELSE SHOULD BE TEMPORARY VARIABLES USED WITHIN THE SCOPE OF THE APPRORPIRATE SPLIT FUNCTION

    # Helpers for Train, Test, and Validation Splits

    def convert_to_16_tensors(self, set_to_convert, set_to_assign=None):
        valid_set_types = {"train", "train_finetuning", "train_unlabeled", "validation", "train_finetuning_unlabeled", "test"}
        assert set_to_convert in valid_set_types, f"'{set_to_convert}' is not a valid set_to_convert. Must be one of {valid_set_types}"

        if set_to_assign is None:
            set_to_assign = set_to_convert

        converted_torch = torch.from_numpy(getattr(self, set_to_convert)).to(torch.float16)
        setattr(self, set_to_assign, converted_torch)


    def del_data(self):
        del self.data

    # Leave_N_Subjects_Out_Randomly Helper Functions
    def train_from_non_leave_out_indices(self):
        self.train = np.concatenate([np.array(self.data[i]) for i in range(utils.num_subjects) if i not in self.leaveOutIndices], axis=0, dtype=np.float16)

    def validation_from_leave_out_indices(self):
        self.validation = np.concatenate([np.array(self.data[i]) for i in range(utils.num_subjects) if i in self.leaveOutIndices], axis=0, dtype=np.float16)

    # Held_Out_Test Helper Functions
    def train_from_train_indices(self, combined_data):
        self.train = combined_data[self.train_indices]

    def validation_from_validation_indices(self, combined_data):
        self.validation = combined_data[self.validation_indices]

    # Leave_One_Session_Out Helper Functions
    def append_to_pretrain(self, new_data):
        self.pretrain.append(new_data)

    def append_to_pretrain_unlabeled_list(self, new_data):
        self.pretrain_unlabeled_list.append(new_data)

    def append_to_finetune(self, new_data):
        self.finetune.append(new_data)

    def append_to_finetune_unlabeled_list(self, new_data):
        self.finetune_unlabeled_list.append(new_data)

    def validation_from_leftout_last(self, left_out_subject_last_session_index):
        self.validation = np.array(self.data[left_out_subject_last_session_index])

    def append_flexwear_unlabeled_to_finetune_unlabeled_list(self):   
        raise NotImplementedError("Subclass must implement this method")

    def concatenate_sessions(self, set_to_assign, set_to_concat):
        """ 
        Sets set_to_assign to set_to_concat concatenated across the 0th axis.
        """
        valid_assign = {"pretrain", "finetune", "validation", "pretrain_unlabeled", "finetune_unlabeled", "train", "train_finetuning", "train_unlabeled", "validation", "train_finetuning_unlabeled", "test"}
        assert set_to_assign in valid_assign, f"' Cannot assign {set_to_assign}. Must be one of {valid_assign}"

        concatenated_data = np.concatenate(getattr(self, set_to_concat), axis=0, dtype=np.float16)
        setattr(self, set_to_assign, concatenated_data)

    def reassign_sets(self):

        self.train = self.pretrain
        if args.proportion_unlabeled_data_from_training_subjects:
            self.train_unlabeled = self.pretrain_unlabeled

        self.train_finetuning = self.finetune
        if args.proportion_unlabeled_data_from_leftout_subject or args.load_unlabeled_data_flexwearhd:
            self.train_finetuning_unlabeled = self.finetune_unlabeled

        self.validation = self.validation

        del self.pretrain
        del self.pretrain_unlabeled
        del self.finetune_unlabeled
        del self.finetune

    def train_from_train_and_finetuning(self):
        self.train = torch.concat((self.train, self.train_finetuning), axis=0)

    def train_from_train_finetuning(self):
        self.train = self.train_finetuning

    def print_set_shapes(self):
        for set_name in ['train', 'train_unlabeled', 'finetuning_unlabeled', 'validation'], "train_partial_leftout_subject"]:
            dataset = getattr(self, set_name, None)
            if dataset is not None:
                print(f"Size of {set_name}: {dataset.size()}")

    def finetuning_if_no_proportion(self):
        self.train_finetuning = self.train_finetuning
        self.train_finetuning_unlabeled = None

    # Leave_One_Subject_Out Helper Functions (LOSO)
    def append_to_train_unlabeled_list(self, new_data):
        self.train_unlabeled_list.append(new_data)
    
    def validation_from_leave_out_subject(self):
        self.validation = np.array(self.data[self.leaveOut-1])

    def concatenate_to_tain(self, new_data):
        self.train = np.concatenate(self.train, new_data, axis=0)

    def train_finetuning_from_set(self, new_data):
        self.train_finetuning = torch.tensor(new_data)
        
class X_Data(Data):

    def __init__(self):
        super().__init__("X")
        self.width = None
        self.length = None

        self.global_low_value = None
        self.global_high_value = None
        self.scaler = None
        self.train_indices = None
        self.validation_indices = None

    # Load EMG Data
    def load_data(self, exercises):
        """ Sets self.data to EMG data. (emg) """

        def load_EMG_ninapro():
            """Gets the EMG data for Ninapro datasets.

            Returns:
                emg (EXERCISE SET, SUBJECT, TRIAL, CHANNEL, TIME): EMG data for the dataset with target_normalization (if applicable).
            """

            emg = []
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                for exercise in args.exercises:
                    if (args.target_normalize > 0):
                        mins, maxes = utils.getExtrema(args.leftout_subject, args.target_normalize, exercise, args)
                        emg_async = pool.map_async(utils.getEMG, [(i+1, exercise, mins, maxes, args.leftout_subject, args) for i in range(utils.num_subjects)])

                    else:
                        emg_async = pool.map_async(utils.getEMG, [(i+1, exercise, args) for i in range(utils.num_subjects)])

                    emg.append(emg_async.get()) # (EXERCISE SET, SUBJECT, TRIAL, CHANNEL, TIME)

            return self.process_ninapro(emg)

        def load_EMG_other_datasets(): 
            """Loads the EMG data for other, non Ninapro datasets.

            Returns:
                emg (SUBJECT, TRIAL, CHANNEL, TIME STEP): EMG data for the dataset with target_normalization (if applicable). 
            """
            emg = []
            if (args.target_normalize > 0):
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                    mins, maxes = utils.getExtrema(args.leftout_subject, args.target_normalize)
                    if args.leave_one_session_out:
                        emg = []
                        labels = []
                        for i in range(1, utils.num_sessions+1):
                            emg_async = pool.map_async(utils.getEMG_separateSessions, [(j+1, i, mins, maxes, args.leftout_subject) for j in range(utils.num_subjects)])
                            emg.extend(emg_async.get())

                    else:
                        emg_async = pool.map_async(utils.getEMG, [(i+1, mins, maxes, args.leftout_subject + 1) for i in range(utils.num_subjects)])

                        emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
            else: 
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                    if args.leave_one_session_out:
            
                        emg = []
                        labels = []
                        for i in range(1, utils.num_sessions+1):
                            emg_async = pool.map_async(utils.getEMG_separateSessions, [(j+1, i) for j in range(utils.num_subjects)])
                            emg.extend(emg_async.get())
                
                    else: # Not leave one session out
                        emg_async = pool.map_async(utils.getEMG, [(i+1) for i in range(utils.num_subjects)])
                        emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
            return emg

        if exercises:
            # Ninapro datasets have to be processed.
            self.data = load_EMG_ninapro()
        else:

            self.data = load_EMG_other_datasets()

    # Helper for loading EMG data for Ninapro
    def concat_across_exercises(self, subject_data):
        """Concatenates EMG data across exercises for a given subject.
        Helper function for process_ninapro.
        """
        concatenated_trials = np.concatenate(subject_data, axis=0)
        return concatenated_trials

    def scaler_normalize_emg(self):
        """Sets the global_low_value, global_high_value, and scaler for X (EMG) data. 

        Returns:

            if args.leave_n_subjects_out_randomly:
                leaveOutIndices: leave out indices
                train_indices, validation_indices: None
            if args.held_out_test:
                leaveOutIndices: None
                train_indices, validation_indices: train and validation indices
        """

        def compute_emg_in():
            """Prepares emg_in by concatenating EMG data and reshaping if neccessary. emg_in is a temporary variable used to compute the scaler.

            If args.held_out_test, returns train and validation indices
            If args.leave_n_subjects_out_randomly, returns leaveOutIndices

            """

            leaveOutIndices = []

            # train and validation indices only for args.held_out_test
            train_indices = None
            validation_indices = None
            
            if args.leave_n_subjects_out_randomly:
                leave_out_count = args.leave_n_subjects_out_randomly
                print(f"Leaving out {leave_out_count} subjects randomly")
                # subject indices to leave out randomly
                leaveOutIndices = np.random.choice(range(utils.num_subjects), leave_out_count, replace=False)
                print(f"Leaving out subjects {np.sort(leaveOutIndices)}")
                emg_in = np.concatenate([np.array(i.view(len(i), self.length*self.width)) for i in emg if i not in leaveOutIndices], axis=0, dtype=np.float32)
                
            else:
                if (args.held_out_test): # can probably be deprecated and deleted
                    if args.turn_on_kfold:
                        
                        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
                    
                        emg_in = np.concatenate([np.array(i.reshape(-1, self.length*self.width)) for i in emg], axis=0, dtype=np.float32)
                        labels_in = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)
                        
                        labels_for_folds = np.argmax(labels_in, axis=1)
                        
                        fold_count = 1
                        for train_index, test_index in skf.split(emg_in, labels_for_folds):
                            if fold_count == args.fold_index:
                                train_indices = train_index
                                validation_indices = test_index
                                break
                            fold_count += 1

                    else:
                        # Reshape and concatenate EMG data
                        # Flatten each subject's data from (TRIAL, CHANNEL, TIME) to (TRIAL, CHANNEL*TIME)
                        # Then concatenate along the subject dimension (axis=0)
                        emg_in = np.concatenate([np.array(i.reshape(-1, self.length*self.width)) for i in emg], axis=0, dtype=np.float32)
                        labels_in = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)

                        indices = np.arange(emg_in.shape[0])
                        train_indices, validation_indices = model_selection.train_test_split(indices, test_size=0.2, stratify=labels_in)

                elif (not args.turn_off_scaler_normalization and not (args.target_normalize > 0)): # Running LOSO standardization
                    emg_in = np.concatenate([np.array(i.view(len(i), self.length*self.width)) for i in emg[:(self.leaveOut-1)]] + [np.array(i.view(len(i), self.length*self.width)) for i in emg[self.leaveOut:]], axis=0, dtype=np.float32)
                else:
                    assert False, "Should not reach here. Need to catch none case earlier in scalar_normalize_emg()"
                        
            # if args.held_out_test, returns train and validation indices 
            # if args.leave_n_subjects_out_randomly, returns leaveOutIndices
            # if LOSO, uses self.leaveOut

            return emg_in, train_indices, validation_indices, leaveOutIndices
    
        def compute_scaler(emg_in, train_indices=None):
            """Compues global low, global high, and scaler for EMG data.

            Args:
                emg_in: incoming EMG data to scaler normalize
                train_indices: list of training indices. Needed when args.held_out_test. Defaults to None.

            Returns:
                global_low_value, global_high_value, scaler: global low value, global high value, and scaler for EMG data.
            """
            if args.held_out_test:
                selected_emg = emg_in[train_indices]
            else:
                selected_emg = emg_in

            global_low_value = selected_emg.mean() - sigma_coefficient*selected_emg.std()
            global_high_value = selected_emg.mean() + sigma_coefficient*selected_emg.std()

            # Normalize by electrode
            emg_in_by_electrode = selected_emg.reshape(-1, self.length, self.width)

            # Assuming emg is your initial data of shape (SAMPLES, 16, 50)
            # Reshape data to (SAMPLES*50, 16)
            emg_reshaped = emg_in_by_electrode.reshape(-1, utils.numElectrodes)

            # Initialize and fit the scaler on the reshaped data
            # This will compute the mean and std dev for each electrode across all samples and features
            scaler = preprocessing.StandardScaler()
            scaler.fit(emg_reshaped)
            
            # Repeat means and std_devs for each time point using np.repeat
            scaler.mean_ = np.repeat(scaler.mean_, self.width)
            scaler.scale_ = np.repeat(scaler.scale_, self.width)
            scaler.var_ = np.repeat(scaler.var_, self.width)
            scaler.n_features_in_ = self.width*utils.numElectrodes

            del emg_in
            del emg_in_by_electrode
            del emg_reshaped

            return global_low_value, global_high_value, scaler
    

        # Define conditions for scaler normalization
        train_indices, validation_indices, leaveOutIndices = None, None, None
        global_low_value, global_high_value, scaler = None, None, None

        if args.leave_n_subjects_out_randomly != 0 and (not args.turn_off_scaler_normalization and not (args.target_normalize > 0)):
            emg_in, _, _, leaveOutIndices = compute_emg_in()
            global_low_value, global_high_value, scaler = compute_scaler(emg_in)
        else:
            if not args.held_out_test:
                emg_in, train_indices, validation_indices, _ = compute_emg_in()
                global_low_value, global_high_value, scaler = compute_scaler(emg_in, train_indices)

            elif (not args.turn_off_scaler_normalization and not (args.target_normalize > 0)): # Running LOSO standardization
                emg_in, _, _, _ = compute_emg_in()
                global_low_value, global_high_value, scaler = compute_scaler(emg_in)

        # These three are only needed to get images
        self.global_high_value = global_high_value
        self.global_low_value = global_low_value
        self.scaler = scaler

        return train_indices, validation_indices, leaveOutIndices
    
    def load_images(self, base_foldername_zarr):
        """Updates self.data to be the loaded images for EMG data. Returns flexwear_unlabeled_data if args.load_unlabeled_data_flexwearhd.
        
        If dataset exists, loads images. Otherwise, creates imaeges and saves in directory. 

        Returns:
            flexwear_unlabeled_data: unlabeled data if args.load_unlabeled_data_flexwearhd
        """
        assert utils is not None, "utils is not defined. Please run initialize() first."

        flexwear_unlabeled_data = None

        emg = self.data # should already be defined as emg using load_data
        image_data = []
        for x in tqdm(range(len(emg)), desc="Number of Subjects "):
            if args.held_out_test:
                subject_folder = f'subject{x}/'
            elif args.leave_one_session_out:
                subject_folder = f'session{x}/'
            else:
                subject_folder = f'LOSO_subject{x}/'
            foldername_zarr = base_foldername_zarr + subject_folder
            
            subject_or_session = "session" if args.leave_one_session_out else "subject"
            print(f"Attempting to load dataset for {subject_or_session}", x, "from", foldername_zarr)

            print("Looking in folder: ", foldername_zarr)
            # Check if the folder (dataset) exists, load if yes, else create and save
            if os.path.exists(foldername_zarr):
                # Load the dataset
                dataset = zarr.open(foldername_zarr, mode='r')
                print(f"Loaded dataset for {subject_or_session} {x} from {foldername_zarr}")
                if args.load_few_images:
                    image_data += [dataset[:10]]
                else: 
                    image_data += [dataset[:]]
            else:
                print(f"Could not find dataset for {subject_or_session} {x} at {foldername_zarr}")
                # Get images and create the dataset
                if (args.target_normalize > 0):
                    self.scaler = None
                images = utils.getImages(emg[x], self.scaler, self.length, self.width, 
                                        turn_on_rms=args.turn_on_rms, rms_windows=args.rms_input_windowsize, 
                                        turn_on_magnitude=args.turn_on_magnitude, global_min=self.global_low_value, global_max=self.global_high_value, 
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
                image_data += [images]
                
        if args.load_unlabeled_data_flexwearhd:
            unlabeled_images = utils.getImages(unlabeled_online_data, self.scaler, self.length, self.width,
                                                        turn_on_rms=args.turn_on_rms, rms_windows=args.rms_input_windowsize,
                                                        turn_on_magnitude=args.turn_on_magnitude, global_min=self.global_low_value, global_max=self.global_high_value,
                                                        turn_on_spectrogram=args.turn_on_spectrogram, turn_on_cwt=args.turn_on_cwt,
                                                        turn_on_hht=args.turn_on_hht)
            unlabeled_images = np.array(unlabeled_images, dtype=np.float16)
            flexwear_unlabeled_data = unlabeled_images
            self.flexwear_unlabeled_data = flexwear_unlabeled_data
            del unlabeled_images, unlabeled_online_data

        self.data = image_data
        

        return flexwear_unlabeled_data
        
    # Helper for leave_one_session_out

    def append_flexwear_unlabeled_to_finetune_unlabeled_list(self, flexwear_unlabeled_data):
        assert args.load_unlabeled_data_flexwearhd, "Cannot append unlabeled data if load_unlabeled_data_flexwearhd is turned off."
        self.finetune_unlabeled_list.append(flexwear_unlabeled_data)
        



class Y_Data(Data):
    
    def __init__(self):
        super().__init__("Y")

    def load_data(self, exercises):
        """ Sets self.data to force data if args.force_regression, otherwise sets self.data to labels. """


        def load_labels_ninapro():
            labels = []
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                for exercise in args.exercises:
                    labels_async = pool.map_async(utils.getLabels, list(zip([(i+1) for i in range(utils.num_subjects)], exercise*np.ones(utils.num_subjects).astype(int), [args]*utils.num_subjects)))
                    labels.append(labels_async.get())

            return self.process_ninapro(labels)

        def load_labels_other_datasets():
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                if args.leave_one_session_out:
                    labels = []
                    for i in range(1, utils.num_sessions+1):
                        labels_async = pool.map_async(utils.getLabels_separateSessions, [(j+1, i) for j in range(utils.num_subjects)])
                        labels.extend(labels_async.get())
                else:
                    labels_async = pool.map_async(utils.getLabels, [(i+1) for i in range(utils.num_subjects)])
                    labels = labels_async.get()
            
            return labels

        def load_labels():
            if exercises:
                return load_labels_ninapro()
            else:
                return load_labels_other_datasets()

        def load_forces():
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                for exercise in args.exercises:
                    assert(exercise == 3), "Regression only implemented for exercise 3"
                    forces_async = pool.map_async(utils.getForces, list(zip([(i+1) for i in range(utils.num_subjects)], exercise*np.ones(utils.num_subjects).astype(int))))
                    forces.append(forces_async.get())

            return self.process_ninapro(forces)

        if args.force_regression:
            self.data = load_forces()
        else:
            self.data = load_labels()

    def concat_across_exercises(self, subject_data):
        """Concatenates forces/labels across exercises for a given subject. 
        
        Helper function for process_ninapro. If args.force_regression, concatenates forces. Otherwise, processes labels from one hot encoding and concatenates. 
        """
        if args.force_regression:
            concatenated_forces = np.concatenate(subject_data, axis=0)
            return concatenated_forces
        else:

            # Convert from one hot encoding to labels
            # Assuming labels are stored separately and need to be concatenated end-to-end

            labels_set = []
            index_to_start_at = 0
            for i in range(len(subject_data)):
                subject_labels_to_concatenate = [x + index_to_start_at if x != 0 else 0 for x in np.argmax(subject_data[i], axis=1)]
                if args.dataset == "ninapro-db5":
                    index_to_start_at = max(subject_labels_to_concatenate)
                labels_set.append(subject_labels_to_concatenate)

            if args.partial_dataset_ninapro:
                desired_gesture_labels = utils.partial_gesture_indices

            concatenated_labels = np.concatenate(labels_set, axis=0) # (TRIAL)

            if args.partial_dataset_ninapro:
                indices_for_partial_dataset = np.array([indices for indices, label in enumerate(concatenated_labels) if label in desired_gesture_labels])
                concatenated_labels = concatenated_labels[indices_for_partial_dataset]
                concatenated_trials = concatenated_trials[indices_for_partial_dataset]
                if args.force_regression:
                    concatenated_forces = concatenated_forces[indices_for_partial_dataset]
                # convert labels to indices
                label_to_index = {label: index for index, label in enumerate(desired_gesture_labels)}
                concatenated_labels = [label_to_index[label] for label in concatenated_labels]
            
            numGestures = len(np.unique(concatenated_labels))

            # Convert to one hot encoding
            concatenated_labels = np.eye(np.max(concatenated_labels) + 1)[concatenated_labels] # (TRIAL, GESTURE)

            return concatenated_labels


    def append_flexwear_unlabeled_to_finetune_unlabeled_list(self, flexwear_unlabeled_data):
        assert args.load_unlabeled_data_flexwearhd, "Cannot append unlabeled data if load_unlabeled_data_flexwearhd is turned off."
        self.finetune_unlabeled_list.append(np.zeros(flexwear_unlabeled_data.shape[0]))

class Label_Data(Data):

    def __init__(self):
            super().__init__("Label")

    def load_data(self, exercises):
        """Sets self.data to labels. """

        def get_labels_ninapro():
            labels = []
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                for exercise in args.exercises:
                    labels_async = pool.map_async(utils.getLabels, list(zip([(i+1) for i in range(utils.num_subjects)], exercise*np.ones(utils.num_subjects).astype(int), [args]*utils.num_subjects)))
                    labels.append(labels_async.get())

            return self.process_ninapro(labels)

            
        def get_labels_other_datasets():
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                if args.leave_one_session_out:
                    labels = []
                    for i in range(1, utils.num_sessions+1):
                        labels_async = pool.map_async(utils.getLabels_separateSessions, [(j+1, i) for j in range(utils.num_subjects)])
                        labels.extend(labels_async.get())
                else:
                    labels_async = pool.map_async(utils.getLabels, [(i+1) for i in range(utils.num_subjects)])
                    labels = labels_async.get()
            
            return labels

        if exercises:
            self.data = get_labels_ninapro()
        else: 
            self.data = get_labels_other_datasets()
        
    def concat_across_exercises(self, subject_data):

        """Concatenates label data across exercises for a given subject.
        Helper function for process_ninapro.
        """

        # Convert from one hot encoding to labels
        # Assuming labels are stored separately and need to be concatenated end-to-end

        labels_set = []
        index_to_start_at = 0
        for i in range(len(subject_data)):
            subject_labels_to_concatenate = [x + index_to_start_at if x != 0 else 0 for x in np.argmax(subject_data[i], axis=1)]
            if args.dataset == "ninapro-db5":
                index_to_start_at = max(subject_labels_to_concatenate)
            labels_set.append(subject_labels_to_concatenate)

        if args.partial_dataset_ninapro:
            desired_gesture_labels = utils.partial_gesture_indices

        concatenated_labels = np.concatenate(labels_set, axis=0) # (TRIAL)

        if args.partial_dataset_ninapro:
            indices_for_partial_dataset = np.array([indices for indices, label in enumerate(concatenated_labels) if label in desired_gesture_labels])
            concatenated_labels = concatenated_labels[indices_for_partial_dataset]
            concatenated_trials = concatenated_trials[indices_for_partial_dataset]
            if args.force_regression:
                concatenated_forces = concatenated_forces[indices_for_partial_dataset]
            # convert labels to indices
            label_to_index = {label: index for index, label in enumerate(desired_gesture_labels)}
            concatenated_labels = [label_to_index[label] for label in concatenated_labels]
        
        numGestures = len(np.unique(concatenated_labels))

        # Convert to one hot encoding
        concatenated_labels = np.eye(np.max(concatenated_labels) + 1)[concatenated_labels] # (TRIAL, GESTURE)

        return concatenated_labels

    def append_flexwear_unlabeled_to_finetune_unlabeled_list(self, flexwear_unlabeled_data):
        assert args.load_unlabeled_data_flexwearhd, "Cannot append unlabeled data if load_unlabeled_data_flexwearhd is turned off."
        self.finetune_unlabeled_list.append(np.zeros(flexwear_unlabeled_data.shape[0]))


# Helper routines for main (can likely be moved into their own class)
def parse_args(): 
    """Argument parser for configuring different trials. 

    Returns:
        ArgumentParser: argument parser 
    """


    def list_of_ints(arg):
        """Define a custom argument type for a list of integers"""
        return list(map(int, arg.split(',')))

    ## Argument parser with optional argumenets

    # Create the parser
    parser = argparse.ArgumentParser(description="Include arguments for running different trials")
    parser.add_argument("--force_regression", type=utils.str2bool, help="Regression between EMG and force data", default=False)
    parser.add_argument('--dataset', help='dataset to test. Set to MCS_EMG by default', default="MCS_EMG")
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

    args = parser.parse_args()
    return args

def initialize():
    """ Conducts safety checks on the args, downloads needed datasets, and imports the correct utils file. """

    args = parse_args()
    exercises = False
    args.dataset = args.dataset.lower()

    # SAFETY CHECKS 
    if args.load_unlabeled_data_flexwearhd:
        assert args.dataset == "flexwear-hd", "Can only load unlabeled online data from FlexWear-HD dataset"
        print("Loading unlabeled online data from FlexWear-HD dataset")
        unlabeled_online_data = utils.getOnlineUnlabeledData(args.leftout_subject)

    # TODO: check tha proportion of unlabeled data always means domwain adaptation
    if args.proportion_unlabeled_data_from_training_subjects > 0.0:
        assert args.turn_on_unlabeled_domain_adaptation, "Cannot use unlabeled data from training subjects without turning on unlabeled domain adaptation"

    if args.proportion_unlabeled_data_from_leftout_subject > 0.0:
        assert args.turn_on_unlabeled_domain_adaptation, "Cannot use unlabeled data from leftout subject without turning on unlabeled domain adaptation"

    if args.proportion_unlabeled_data_from_leftout_subject > 0.0:
        assert args.turn_on_unlabeled_domain_adaptation, "Cannot use unlabeled data from leftout subject without turning on unlabeled domain adaptation"

    if utils.num_subjects == 1:
            assert not args.pretrain_and_finetune, "Cannot pretrain and finetune with only one subject"

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
        print(f"The dataset being tested is uciEMG")
        project_name = 'emg_benchmarking_uci'
        args.dataset = "uciemg"

    elif (args.dataset in {"ninapro-db2", "ninapro_db2"}):
        if (not os.path.exists("./NinaproDB2")):
            print("NinaproDB2 dataset does not exist yet. Downloading now...")
            subprocess.run(['python', './get_datasets.py', '--NinaproDB2'])
        import utils_NinaproDB2 as utils
        print(f"The dataset being tested is ninapro-db2")
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
        print(f"The dataset being tested is ninapro-db5")
        project_name = 'emg_benchmarking_ninapro-db5'
        exercises = True
        if args.leave_one_session_out:
            raise ValueError("leave-one-session-out not implemented for ninapro-db5; only one session exists")
        args.dataset = 'ninapro-db5'

    elif (args.dataset in {"ninapro-db3", "ninapro_db3"}):
        import utils_NinaproDB3 as utils

        assert args.exercises == [1] or args.partial_dataset_ninapro or (args.exercises == [3] and args.force_regression), "Exercise C cannot be used for classification due to missing data."
        print(f"The dataset being tested is ninapro-db3")
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
        print(f"The dataset being tested is myoarmbanddataset")
        project_name = 'emg_benchmarking_myoarmbanddataset'
        if args.leave_one_session_out:
            raise ValueError("leave-one-session-out not implemented for myoarmbanddataset; only one session exists")
        args.dataset = 'myoarmbanddataset'

    elif (args.dataset.lower() == "hyser"):
        if (not os.path.exists("./hyser")):
            print("Hyser dataset does not exist yet. Downloading now...")
            subprocess.run(['python', './get_datasets.py', '--Hyser'])
        import utils_Hyser as utils
        print(f"The dataset being tested is hyser")
        project_name = 'emg_benchmarking_hyser'
        args.dataset = 'hyser'

    elif (args.dataset.lower() == "capgmyo"):
        if (not os.path.exists("./CapgMyo_B")):
            print("CapgMyo_B dataset does not exist yet. Downloading now...")
            subprocess.run(['python', './get_datasets.py', '--CapgMyo_B'])
        import utils_CapgMyo as utils
        print(f"The dataset being tested is CapgMyo")
        project_name = 'emg_benchmarking_capgmyo'
        if args.leave_one_session_out:
            utils.num_subjects = 10
        args.dataset = 'capgmyo'

    elif (args.dataset.lower() == "flexwear-hd"):
        if (not os.path.exists("./FlexWear-HD")):
            print("FlexWear-HD dataset does not exist yet. Downloading now...")
            subprocess.run(['python', './get_datasets.py', '--FlexWearHD_Dataset'])
        import utils_FlexWearHD as utils
        print(f"The dataset being tested is FlexWear-HD Dataset")
        project_name = 'emg_benchmarking_flexwear-hd_dataset'
        # if args.leave_one_session_out:
            # raise ValueError("leave-one-session-out not implemented for FlexWear-HDDataset; only one session exists")
        args.dataset = 'flexwear-hd'

    elif (args.dataset.lower() == "sci"):
        import utils_SCI as utils
        print(f"The dataset being tested is SCI")
        project_name = 'emg_benchmarking_sci'
        args.dataset = 'sci'
        assert not args.transfer_learning, "Transfer learning not implemented for SCI dataset"
        assert not args.leave_one_subject_out, "Leave one subject out not implemented for SCI dataset"

    elif (args.dataset.lower() == "mcs"):
        if (not os.path.exists("./MCS_EMG")):
            print("MCS dataset does not exist yet. Downloading now...")
            subprocess.run(['python', './get_datasets.py', '--MCS_EMG'])

        print(f"The dataset being tested is MCS_EMG")
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
        
    if args.turn_off_scaler_normalization:
        assert args.target_normalize == 0.0, "Cannot turn off scaler normalization and turn on target normalize at the same time"
        

    def print_params(args):
        for param, value in vars(args).items():
            if getattr(args, param):
                print(f"The value of --{param} is {value}")

    print_params(args)
            

    # Add date and time to filename
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    print("------------------------------------------------------------------------------------------------------------------------")
    print("Starting run at", formatted_datetime)
    print("------------------------------------------------------------------------------------------------------------------------")

    return args, exercises, project_name, formatted_datetime, utils

def set_exercise():
    """ Set the exercises for the partial dataset for Ninapro datasets. 
    """

    if args.partial_dataset_ninapro:
        if args.dataset == "ninapro-db2":
            args.exercises = [1]
        elif args.dataset == "ninapro-db5":
            args.exercises = [2]
        elif args.dataset == "ninapro-db3":
            args.exercises = [1]

def create_foldername_zarr():
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
        if args.target_normalize > 0:
            base_foldername_zarr += 'target_normalize_' + str(args.target_normalize) + '/'  

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

    return base_foldername_zarr


def main():

    global args, exercises, project_name, formatted_datetime, utils
    import utils_MCS_EMG as utils # default to work for argparse
    args, exercises, project_name, formatted_datetime, utils = initialize()

    # this allows you to access them individually or just do calls that affect all three of then
    X = X_Data()
    Y = Y_Data()
    Label = Label_Data()

    set_exercise()

    All_Data = Combined(X, Y, Label)
    All_Data.load_data()

    # Print data information
    X.length = X.data[0].shape[1]
    X.width = X.data[0].shape[2]
    
    print("Number of Samples (across all participants): ", sum([e.shape[0] for e in All_Data.X.data]))
    print("Number of Electrode Channels: ", X.length)
    print("Number of Timesteps per Trial:", X.width)

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

    if args.leave_n_subjects_out_randomly:
        leaveOut = args.leave_n_subjects_out_randomly 
     
    X.scaler_normalize_emg()

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

    print("Width of EMG data: ", All_Data.X.width)
    print("Length of EMG data: ", All_Data.X.length)

    base_foldername_zarr = create_foldername_zarr() # needs access to self.leaveOut
    X.load_images(base_foldername_zarr)


if __name__ == "__main__":
    main()



# need to add a process function