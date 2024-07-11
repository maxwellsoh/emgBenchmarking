from .Data_Split_Strategy import Data_Split_Strategy
from .cross_validation_utilities import train_test_split as tts # custom train test split to split stratified without shuffling
class Leave_One_Session_Out(Data_Split_Strategy):
    """ 
    Splits data based on leaving one session of self.leaveOut subject out. 
    """

    def __init__(self, X_data, Y_data, label_data, env):
        super().__init__(X_data, Y_data, label_data, env)

    def concatenate_pretrain_and_finetune(self):
        """
        Concatenate sets across the 0th axis (sessions) for pretrain, finetune, and if enabled, pretrain_unlabeled and finetune_unlabeled. 
        """

        super().concatenate_sessions(set_to_assign="pretrain", set_to_concat="pretrain")
        super().concatenate_sessions(set_to_assign="finetune", set_to_concat="finetune")

        if self.args.proportion_unlabeled_data_from_training_subjects and self.args.turn_on_unlabeled_domain_adaptation:
            super().concatenate_sessions(set_to_assign="pretrain_unlabeled", set_to_concat="pretrain_unlabeled_list")

        if (self.args.proportion_unlabeled_data_from_leftout_subject > 0 or self.args.load_unlabeled_data_flexwearhd) and self.args.turn_on_unlabeled_domain_adaptation:
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

        self.args:
            X_train_temp: EMG data for the current subject and session
            Y_train_temp: Label/Force data for the current subject and session
            label_train_temp: Label data for the current subject and session

        Sets:
            self.pretrain: Non left out subject's labeled data
            self.pretrain_unlabeled_list: Non left out subject's unlabeled data (if proportion_unlabeled_data_from_training_subjects > 0)
        """

        if self.args.proportion_data_from_training_subjects<1.0:
            X_train_temp, _, \
            Y_train_temp, _, \
            label_train_temp, _ \
            = tts.train_test_split(
                X_train_temp, 
                Y_train_temp, 
                train_size=self.args.proportion_data_from_training_subjects, 
                stratify=label_train_temp, 
                random_state=self.args.seed, 
                shuffle=(not self.args.train_test_split_for_time_series)
            )
            
        if self.args.proportion_unlabeled_data_from_training_subjects>0 and self.args.turn_on_unlabeled_domain_adaptation:
            X_pretrain_labeled, X_pretrain_unlabeled, \
            Y_pretrain_labeled, Y_pretrain_unlabeled, \
            label_pretrain_labeled, label_pretrain_unlabeled \
            = tts.train_test_split(
                X_train_temp, 
                Y_train_temp, 
                train_size=1-self.args.proportion_unlabeled_data_from_training_subjects, 
                stratify=label_train_temp, 
                random_state=self.args.seed, 
                shuffle=(not self.args.train_test_split_for_time_series)
            )

            self.append_to_pretrain(X_pretrain_labeled, Y_pretrain_labeled, label_pretrain_labeled)

            self.append_to_pretrain_unlabeled_list(X_pretrain_unlabeled, Y_pretrain_unlabeled, label_pretrain_unlabeled)

        else:
            self.append_to_pretrain(X_train_temp, Y_train_temp, label_train_temp)

    def finetune_from_leftout_first_n(self, X_train_temp, Y_train_temp, label_train_temp):

        """ 
        Creates finetune and finetune_unlabeled sets from left out subject's first n sessions data.

        If proportion_unlabeld_data_from_leftout_subject > 0, splits the first n sessions data into labeled and unlabeled data.

        self.args:
            X_train_temp: EMG data for the current subject and session
            Y_train_temp: Label/Force data for the current subject and session
            label_train_temp: Label data for the current subject and session

        Sets:
            self.finetune: Left out subject's first n sessions labeled data
            self.finetune_unlabeled_list: Left out subject's first n sessions unlabeled data (if proportion_unlabeled_data_from_leftout_subject > 0)    
        
        """

        if self.args.proportion_unlabeled_data_from_leftout_subject>0 and self.args.turn_on_unlabeled_domain_adaptation:

            X_finetune_labeled, X_finetune_unlabeled, \
            Y_finetune_labeled, Y_finetune_unlabeled, \
            label_finetune_labeled, label_finetune_unlabeled \
            = tts.train_test_split(
                X_train_temp, 
                Y_train_temp, 
                train_size=1-self.args.proportion_unlabeled_data_from_leftout_subject, 
                stratify=label_train_temp, 
                random_state=self.args.seed, 
                shuffle=(not self.args.train_test_split_for_time_series)
            )

            self.append_to_finetune(X_finetune_labeled, Y_finetune_labeled, label_finetune_labeled)
            self.append_to_finetune_unlabeled_list(X_finetune_unlabeled, Y_finetune_unlabeled, label_finetune_unlabeled)

        else:
            self.append_to_finetune(X_train_temp, Y_train_temp, label_train_temp)

    def validation_from_leftout_last(self, left_out_subject_last_session_index):   
        self.X.validation_from_leftout_last(left_out_subject_last_session_index)
        self.Y.validation_from_leftout_last(left_out_subject_last_session_index)
        self.label.validation_from_leftout_last(left_out_subject_last_session_index)

    def train_from_train_and_finetuning(self):
        """ If not self.args.pretrain_and_finetune, combine train and finetune sets into train. """

        self.X.train_from_train_and_finetuning()
        self.Y.train_from_train_and_finetuning()
        self.label.train_from_train_and_finetuning()

    def train_from_train_finetuning(self):
        """ 
        If not self.args.unlabeled_domain_adaptation, train from train_finetuning. """

        self.X.train_from_train_finetuning()
        self.Y.train_from_train_finetuning()
        self.label.train_from_train_finetuning()

    def create_pretrain_and_finetune(self, left_out_subject_last_session_index, left_out_subject_first_n_sessions_indices):
        """
        Creates pretrain, pretrain_unlabeled, finetune, finetune_unlabeled, and validation sets.

        self.args:
            left_out_subject_last_session_index: index of leftout subject's last session
            left_out_subject_first_n_sessions_indices: indices of leftout subject's first n sessions

        Sets:
            pretrain: non left out subject's labeled
            pretrain_unlabeled: non left out subject's unlabeled (if proportion_unlabeled_data_from_training_subjects > 0)
            finetune: first n sessions of leftout subject labeled
            finetune_unlabeled: first n sessions of leftout subject unlabeled (if proportion_unlabeled_data_from_leftout_subject > 0)
            validation: last session of leftout subject
        """
        for i in range(self.utils.num_sessions*self.utils.num_subjects):

            X_train_temp = self.X.data[i]
            Y_train_temp = self.Y.data[i]
            label_train_temp = self.label.data[i]

            if i != left_out_subject_last_session_index and i not in left_out_subject_first_n_sessions_indices:
                self.pretrain_from_non_leftout(X_train_temp, Y_train_temp, label_train_temp)

            elif i in left_out_subject_first_n_sessions_indices:
                self.finetune_from_leftout_first_n(X_train_temp, Y_train_temp, label_train_temp)

        if self.args.load_unlabeled_data_flexwearhd:
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
        if self.args.proportion_unlabeled_data_from_training_subjects:
            super().convert_to_16_tensors("train_unlabeled")
        if self.args.proportion_unlabeled_data_from_leftout_subject or self.args.load_unlabeled_data_flexwearhd:
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

    def split(self): 
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
        left_out_subject_last_session_index = (self.utils.num_sessions-1) * self.utils.num_subjects + self.leaveOut-1
        left_out_subject_first_n_sessions_indices = [i for i in range(self.utils.num_sessions * self.utils.num_subjects) if i % self.utils.num_subjects == (self.leaveOut-1) and i != left_out_subject_last_session_index]

        print("left_out_subject_last_session_index:", left_out_subject_last_session_index)
        print("left_out_subject_first_n_sessions_indices:", left_out_subject_first_n_sessions_indices)

        self.create_pretrain_and_finetune(left_out_subject_last_session_index, left_out_subject_first_n_sessions_indices)
        self.concatenate_pretrain_and_finetune()
        self.reassign_sets()
        self.convert_datasets_to_tensors()
        
        # Undo any excess splitting
        if self.args.turn_on_unlabeled_domain_adaptation and not (self.args.proportion_unlabeled_data_from_leftout_subject > 0):
            self.finetuning_if_no_proportion()
        elif self.utils.num_subjects == 1:
            self.train_from_train_finetuning()

        if not self.args.pretrain_and_finetune:
            self.train_from_train_and_finetuning()

        self.del_data
        self.test_from_validation()
        super().print_set_shapes()
        super().all_sets_to_tensor()
