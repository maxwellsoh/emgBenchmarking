from .Data_Split_Strategy import Data_Split_Strategy
import numpy as np
from .cross_validation_utilities import train_test_split as tts # custom train test split to split stratified without shuffling
import torch
from sklearn import preprocessing, model_selection

class Leave_One_Subject_Out(Data_Split_Strategy):
        
    def __init__(self, X_data, Y_data, label_data, env):
        super().__init__(X_data, Y_data, label_data, env)

        # Set seeds for reproducibility
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def split(self):

        self.train_from_non_left_out_subj()
        self.validation_from_leave_out_subj()
        self.convert_datasets()

        if self.args.transfer_learning:
            self.adjust_sets_for_leave_out_subj()

        super().test_from_validation()
        super().print_set_shapes()
        super().all_sets_to_tensor()
 

    def append_to_train_unlabeled_list(self, X_new_data, Y_new_data, label_new_data):
        self.X.append_to_train_unlabeled_list(X_new_data)
        self.Y.append_to_train_unlabeled_list(Y_new_data)
        self.label.append_to_train_unlabeled_list(label_new_data)

    def append_to_train_list(self, X_new_data, Y_new_data, label_new_data):
        self.X.append_to_train_list(X_new_data)
        self.Y.append_to_train_list(Y_new_data)
        self.label.append_to_train_list(label_new_data)
    
    def convert_datasets(self):
        super().concatenate_sessions(set_to_assign="train", set_to_concat="train_list")
        super().convert_to_16_tensors(set_to_convert="train")
        
        if self.args.proportion_unlabeled_data_from_training_subjects>0:
            super().concatenate_sessions(set_to_assign="train_unlabeled", set_to_concat="train_unlabeled_list")
            super().convert_to_16_tensors(set_to_convert="train_unlabeled")

        super().convert_to_16_tensors(set_to_convert="validation")

    def concatenate_to_train(self, X_new_data, Y_new_data, label_new_data):
        self.X.concatenate_to_train(X_new_data)
        self.Y.concatenate_to_train(Y_new_data)
        self.label.concatenate_to_train(label_new_data) 

    def concatenate_to_train_unlabeled(self, X_new_data, Y_new_data, label_new_data):
        self.X.concatenate_to_train_unlabeled(X_new_data)
        self.Y.concatenate_to_train_unlabeled(Y_new_data)
        self.label.concatenate_to_train_unlabeled(label_new_data)

    def train_unlabeled_from_self_tensor(self):
        self.X.set_to_self_tensor("train_unlabeled")
        self.Y.set_to_self_tensor("train_unlabeled")
        self.label.set_to_self_tensor("train_unlabeled")
   

    def train_from_non_left_out_subj(self):

        for i in range(len(self.X.data)):
            if i == self.leaveOut-1:
                continue

            X_train_temp = np.array(self.X.data[i])
            Y_train_temp = np.array(self.Y.data[i])
            label_train_temp = np.array(self.label.data[i])

            if self.args.reduce_training_data_size:
                
                reduced_size_per_subject = self.args.reduced_training_data_size // (self.utils.num_subjects - 1)
                proportion_to_keep = reduced_size_per_subject / X_train_temp.shape[0]
                X_train_temp, _, \
                Y_train_temp, _, \
                label_train_temp, _ \
                = model_selection.train_test_split(
                    X_train_temp, 
                    Y_train_temp, 
                    train_size=proportion_to_keep, 
                    stratify=label_train_temp, 
                    random_state=self.args.seed, 
                    shuffle=(not self.args.train_test_split_for_time_series)
                )

            if self.args.proportion_data_from_training_subjects < 1.0:
                X_train_temp, _, \
                Y_train_temp, _, \
                label_train_temp, _ \
                = tts.train_test_split(
                    X_train_temp, 
                    Y_train_temp, 
                    train_size=self.args.proportion_data_from_training_subjects, 
                    stratify=label_train_temp, 
                    random_state=self.args.seed, 
                    shuffle=(not self.args.train_test_split_for_time_series),
                    force_regression=self.args.force_regression
                )
 
            if self.args.proportion_unlabeled_data_from_training_subjects>0:
                X_train_labeled, X_train_unlabeled, \
                Y_train_labeled, Y_train_unlabeled, \
                label_train_labeled, label_train_unlabeled = tts.train_test_split(
                    X_train_temp, 
                    Y_train_temp, 
                    train_size=1-self.args.proportion_unlabeled_data_from_training_subjects, 
                    stratify=label_train_temp, 
                    random_state=self.args.seed, 
                    shuffle=(not self.args.train_test_split_for_time_series), 
                    force_regression=self.args.force_regression
                )

                self.append_to_train_list(X_train_labeled, Y_train_labeled, label_train_labeled)

                self.append_to_train_unlabeled_list(X_train_unlabeled, Y_train_unlabeled, label_train_unlabeled)

            else:
                self.append_to_train_list(X_train_temp, Y_train_temp, label_train_temp)

    def validation_from_leave_out_subj(self):
        self.X.validation_from_leave_out_subj()
        self.Y.validation_from_leave_out_subj()
        self.label.validation_from_leave_out_subj()

    def train_finetuning_from(self, X_new_data, Y_new_data, label_new_data):
        self.X.train_finetuning_from(X_new_data)
        self.Y.train_finetuning_from(Y_new_data)
        self.label.train_finetuning_from(label_new_data)

    def train_finetuning_unlabeled_from(self, X_new_data, Y_new_data, label_new_data):
        self.X.train_finetuning_unlabeled_from(X_new_data)
        self.Y.train_finetuning_unlabeled_from(Y_new_data)
        self.label.train_finetuning_unlabeled_from(label_new_data)

    def adjust_sets_for_leave_out_subj(self): 
        """
        If doing transfer learning, splits left out subject's data into train_partial_leftout_subject and validation_partial_leftout_subject sets.
        """

        assert self.args.transfer_learning, "Transfer learning must be turned on to split left out subject's data."

        proportion_to_keep_of_leftout_subject_for_training = self.args.proportion_transfer_learning_from_leftout_subject
        
        proportion_unlabeled_of_proportion_to_keep_of_leftout = self.args.proportion_unlabeled_data_from_leftout_subject

        proportion_unlabeled_of_training_subjects = self.args.proportion_unlabeled_data_from_training_subjects
        

        # Split leftout validation into train and validation
        if proportion_to_keep_of_leftout_subject_for_training>0.0:
            X_train_partial_leftout_subject, X_validation_partial_leftout_subject, \
            Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject, \
            label_train_partial_leftout_subject, label_validation_partial_leftout_subject = \
                tts.train_test_split(
                    self.X.validation, 
                    self.Y.validation, 
                    train_size=proportion_to_keep_of_leftout_subject_for_training, 
                    stratify=self.label.validation, 
                    random_state=self.args.seed, 
                    shuffle=(not self.args.train_test_split_for_time_series), 
                    force_regression=self.args.force_regression    
                )

        # Otherwise validate with all of left out subject's data
        else:
            X_validation_partial_leftout_subject = self.X.validation
            Y_validation_partial_leftout_subject = self.Y.validation
            label_validation_partial_leftout_subject = self.label.validation

            X_train_partial_leftout_subject = torch.tensor([])
            Y_train_partial_leftout_subject = torch.tensor([])
            label_train_partial_leftout_subject = torch.tensor([])

        # If unlabeled domain adaptation, split the training data into labeled and unlabeled
        if self.args.turn_on_unlabeled_domain_adaptation and proportion_unlabeled_of_proportion_to_keep_of_leftout>0:

            X_train_labeled_partial_leftout_subject, X_train_unlabeled_partial_leftout_subject, \
            Y_train_labeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject, \
            label_train_labeled_partial_leftout_subject, label_train_unlabeled_partial_leftout_subject = \
                tts.train_test_split(
                    X_train_partial_leftout_subject, 
                    Y_train_partial_leftout_subject, 
                    train_size=1-proportion_unlabeled_of_proportion_to_keep_of_leftout, 
                    stratify=label_train_partial_leftout_subject, 
                    random_state=self.args.seed, 
                    shuffle=(not self.args.train_test_split_for_time_series), 
                    force_regression=self.args.force_regression
                )

        # Flexwear unlabeled domain adaptatoin
        if self.args.load_unlabeled_data_flexwearhd:
            if proportion_unlabeled_of_proportion_to_keep_of_leftout>0:
                X_train_unlabeled_partial_leftout_subject = np.concatenate([X_train_unlabeled_partial_leftout_subject, self.X.flexwear_unlabeled_data], axis=0)
                Y_train_unlabeled_partial_leftout_subject = np.concatenate([Y_train_unlabeled_partial_leftout_subject, np.zeros((self.X.flexwear_unlabeled_data.shape[0], self.utils.numGestures))], axis=0)
                label_train_unlabeled_partial_leftout_subject = Y_train_unlabeled_partial_leftout_subject

            else:
                X_train_unlabeled_partial_leftout_subject = self.X.flexwear_unlabeled_data
                Y_train_unlabeled_partial_leftout_subject = np.zeros((self.X.flexwear_unlabeled_data.shape[0], self.utils.numGestures))
                label_train_unlabeled_partial_leftout_subject = Y_train_unlabeled_partial_leftout_subject

        # Add the partial from leftout subject to train/finetune
        if not self.args.turn_on_unlabeled_domain_adaptation:
            # Append the partial validation data to the training data
            if proportion_to_keep_of_leftout_subject_for_training>0:
                
                if not self.args.pretrain_and_finetune:
                    self.concatenate_to_train(X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject)
            
                else:
                    self.train_finetuning_from(X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject)
         
        else: # unlabeled domain adaptation
            if proportion_unlabeled_of_training_subjects>0:
                self.train_from_self_tensor()
                self.train_unlabeled_from_self_tensor()


            if proportion_unlabeled_of_proportion_to_keep_of_leftout>0 or self.args.load_unlabeled_data_flexwearhd:
                if proportion_unlabeled_of_proportion_to_keep_of_leftout==0:
                    X_train_labeled_partial_leftout_subject = X_train_partial_leftout_subject
                    Y_train_labeled_partial_leftout_subject = Y_train_partial_leftout_subject
                    label_train_labeled_partial_leftout_subject = label_train_partial_leftout_subject

                if self.args.pretrain_and_finetune:
                    self.train_finetuning_from(X_train_labeled_partial_leftout_subject, Y_train_labeled_partial_leftout_subject, label_train_labeled_partial_leftout_subject)

                    self.train_finetuning_unlabeled_from(X_train_unlabeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject, label_train_unlabeled_partial_leftout_subject)

                else: 
                    self.concatenate_to_train(X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject)

                    self.train_from_self_tensor()

                    self.concatenate_to_train_unlabeled(X_train_unlabeled_partial_leftout_subject, Y_train_unlabeled_partial_leftout_subject, label_train_unlabeled_partial_leftout_subject)
                    self.train_unlabeled_from_self_tensor()
                  
            else:
                if proportion_to_keep_of_leftout_subject_for_training>0:
                    if not self.args.pretrain_and_finetune:
                        self.concatenate_to_train(X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject)
                        self.train_from_self_tensor()
                    
                    else: 
                        self.train_finetuning_from(X_train_partial_leftout_subject, Y_train_partial_leftout_subject, label_train_partial_leftout_subject)

        # Update the validation data
        self.train_from_self_tensor()
        self.validation_from(X_validation_partial_leftout_subject, Y_validation_partial_leftout_subject, label_validation_partial_leftout_subject)
        
        del X_train_partial_leftout_subject, X_validation_partial_leftout_subject, Y_train_partial_leftout_subject, Y_validation_partial_leftout_subject, label_train_partial_leftout_subject, label_validation_partial_leftout_subject
        
    