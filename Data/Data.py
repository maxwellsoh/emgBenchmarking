"""
Data.py
- Contains Data class, which is a superclass for all data classes.
- Data is responsible for loading and processing data for the given dataset.
"""

import torch
import numpy as np

class Data():
    """
    Superclass for all data classes. Intializes shared arguments using env. Helper functions support data loading and processing. 
    """

    def __init__(self, field, env):


        self.args = env.args
        self.utils = env.utils
        self.leaveOut = env.leaveOut
        self.env = env
        self.exercises = env.exercises

        # Set seeds for reproducibility
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.field = field
        self.data = None

        self.scaler = None # if not self.args.turn_off_scaler_normalization

        # Leave_N_Subjects_Out_Randomly Helper Variables
        leaveOutIndices = None 

        # Leave_One_Session_Out Helper Variables
        self.pretrain = []
        if self.args.proportion_unlabeled_data_from_training_subjects:
            self.pretrain_unlabeled_list = []
        self.finetune = []
        if self.args.proportion_unlabeled_data_from_leftout_subject:
            self.finetune_unlabeled_list= []

        # Leave_One_Subject_Out Helper Variables
        self.train_list = []
        if self.args.proportion_unlabeled_data_from_training_subjects > 0:
            self.train_unlabeled_list = []

        self.train = None
        self.train_unlabeled = None
        self.train_finetuning = None
        self.train_finetuning_unlabeled = None
        self.validation = None

    def load_images(self):
        raise NotImplementedError("Images are loaded by X_Data subclass.")
        
    def load_data(self):
        raise NotImplementedError("Loading data is particular to each set. Subclass must implement this method.")
    
    # Process Ninapro Helper
    def append_to_trials(self, exercise_set, subject):
        raise NotImplementedError("Subclass must implement this method")
    
    def append_to_new_data(self, concatenated_data):
        self.new_data.append(concatenated_data)

    def set_new_data(self):
        self.data = [torch.from_numpy(data_np) for data_np in self.new_data]
    
    def set_values(self, attr, value):
        setattr(self, attr, value)

    # NOTE: Helpers only ever set or update values that are defined in init. (i.e. split sets). Should limit use of temporary variables to within helper function.

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
        self.train = np.concatenate([np.array(self.data[i]) for i in range(self.utils.num_subjects) if i not in self.leaveOutIndices], axis=0, dtype=np.float16)

    def validation_from_leave_out_indices(self):
        self.validation = np.concatenate([np.array(self.data[i]) for i in range(self.utils.num_subjects) if i in self.leaveOutIndices], axis=0, dtype=np.float16)

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

        concatenated_data = np.concatenate(getattr(self, set_to_concat),  axis=0, dtype=np.float16)
        setattr(self, set_to_assign, concatenated_data)

    def reassign_sets(self):

        self.train = self.pretrain
        if self.args.proportion_unlabeled_data_from_training_subjects:
            self.train_unlabeled = self.pretrain_unlabeled

        self.train_finetuning = self.finetune
        if self.args.proportion_unlabeled_data_from_leftout_subject or self.args.load_unlabeled_data_flexwearhd:
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
        for set_name in ["train", "train_unlabeled", "train_finetuning", "train_finetuning_unlabeled", "test", "validation"]:
            dataset = getattr(self, set_name, None)
            if dataset is not None:
                print(f"Size of {self.field}_{set_name}: {dataset.shape}")

    def finetuning_if_no_proportion(self):
        self.train_finetuning = self.train_finetuning
        self.train_finetuning_unlabeled = None

    # Leave_One_Subject_Out Helper Functions (LOSO)
    def append_to_train_unlabeled_list(self, new_data):
        self.train_unlabeled_list.append(new_data)
    
    def validation_from_leave_out_subj(self):
        self.validation = np.array(self.data[self.leaveOut-1])

    def set_to_self_tensor(self, value):
        setattr(self, value, torch.tensor(getattr(self, value)))

    def train_finetuning_from(self, new_data):
        self.train_finetuning = torch.tensor(new_data)

    def train_finetuning_unlabeled_from(self, new_data):
        self.train_finetuning_unlabeled = torch.tensor(new_data)
    
    def validation_from(self, new_data):
        self.validation = torch.tensor(new_data)

    def train_from(self, new_data):
        self.train = torch.tensor(new_data)

    def append_to_train_list(self, new_data):
        self.train_list.append(new_data)

    def concatenate_to_train(self, new_data):
        self.train = np.concatenate((self.train, new_data), axis=0)

    def concatenate_to_train_unlabeled(self, new_data):
        self.train_unlabeled = np.concatenate((self.train_unlabeled, new_data), axis=0)

    # Single_Subject Helper
    def train_from_one_subject(self):
        self.train = self.data[0]

    def all_sets_to_tensor(self):
        for set_name in ["train", "validation", "test"]:

            if isinstance(getattr(self, set_name), np.ndarray):

                new_torch = torch.from_numpy(getattr(self, set_name)).to(torch.float16)
                setattr(self, set_name, new_torch)