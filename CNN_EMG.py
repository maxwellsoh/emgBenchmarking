import torch
import numpy as np
import random 
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from Hook_Manager import Hook_Manager

# Imports for Setup_Run
from Setup.Parse_Arguments import Parse_Arguments
from Setup.Parse_Config import Parse_Config

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
from Model.IRM import IRM_Trainer

class Run_Setup():
    """
    Sets up the run by reading in arguments, setting the dataset source, conducting safety checks, printing values and setting up env.

    Returns:
        env: Setup object which contains necessary information for the run 
    """
   

    def __init__(self, config_args=None):
        self.config_args = config_args

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

        if self.config_args:
            run = Parse_Config(self.config_args)
        else:
            run = Parse_Arguments() 
    
        run.set_args()
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
        all_data.load_data()
        all_data.scaler_normalize_emg()
        
        self.X.load_images()
        self.X.print_data_information()

        return self.X, self.Y, self.label
    

class Data_Splitter():

    def __init__(self, env):

        self.args = env.args
        self.utils = env.utils
        self.env = env

    def split_data(self, X_data, Y_data, label_data):
        if self.args.leave_one_session_out:
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

        if self.args.turn_on_unlabeled_domain_adaptation:
            model_trainer = Unlabeled_Domain_Adaptation_Trainer(X, Y, label, self.env)
        else:
            if self.args.model == "MLP":
                model_trainer = MLP_Trainer(X, Y, label, self.env)
            elif self.args.model in ["SVC", "RF"]:
                model_trainer = SVC_RF_Trainer(X, Y, label, self.env)
            else:
                if self.args.domain_generalization == "IRM":
                    model_trainer = IRM_Trainer(X, Y, label, self.env)
                else:
                    model_trainer = CNN_Trainer(X, Y, label, self.env)

        model_trainer.setup_model()
        model_trainer.model_loop()


def main(config_args=None):

    hooks = Hook_Manager()
    run_setup = Run_Setup(config_args)
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

def use_config(config_args):
    """
    Called by run_CNN_EMG if a config file is passed.
    """

    main(config_args)

if __name__ == "__main__":
    main()