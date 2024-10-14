
import os
import subprocess
import datetime
import argparse

class Setup():
    """
    Sets up the run by reading in arguments, setting the dataset source, conducting safety checks, printing values and setting up env.

    Returns:
        env: Setup object which contains necessary information for the run 
    """

    
    def __init__(self):
        self.args = None 
        self.exercises = None
        self.project_name = None
        self.formatted_datetime = None
        from .Utils import utils_MCS_EMG as utils # default for argparse
        self.utils = utils

    class Env():
        """
        Object to store run information shared across all Data and Model classes. 
        """

        def __init__(self):
            self.args = None
            from .Utils import utils_MCS_EMG as utils
            self.utils = utils
            self.exercises = None
            self.project_name = None
            self.formatted_datetime = None
            self.leaveOut = None
            self.seed = None


    def create_argparse(self): 
        """
        
        Create argument parser for configuring different trials. 
        Important that argparse is declared here as run_CNN_EMG still uses it when using a config to set the default values. 

        Returns:
            ArgumentParser: argument parser 
        """
        
        from .Utils import utils_MCS_EMG as utils

        def list_of_ints(arg):
            """Define a custom argument type for a list of integers"""
            return list(map(int, arg.split(',')))

        ## Argument parser with optional argumenets

        # Create the parser
        parser = argparse.ArgumentParser(description="Include arguments for running different trials")

        # Arguments for run_CNN_EMG/using config files
        parser.add_argument('--config', type=str, help="Path to the config file.")
        parser.add_argument('--table', type=str, help="Specify which table to replicate. (Ex: 1, 2, 3, 3_intersession)")

        
        parser.add_argument("--include_transitions", type=utils.str2bool, help="Whether or not to include transitions windows and label them as the final gesture. Set to False by default.", default=False)
        parser.add_argument("--transition_classifier", type=utils.str2bool, help="Whether or not to classify whether a window is a transition. Set to False by default.", default=False)
        parser.add_argument("--multiprocessing", type=utils.str2bool, help="Whether or not to use multiprocessing when acquiring data. Set to True by default.", default=True)
        parser.add_argument("--force_regression", type=utils.str2bool, help="Regression between EMG and force data", default=False)

        # Add argument for doing domain generalization algorithm
        parser.add_argument('--domain_generalization', type=str, help='domain generalization algorithm to use (e.g. \'IRM\',\'CORAL\'.', default=False)

        parser.add_argument('--dataset', help='dataset to test. Set to MCS_EMG by default', default="MCS_EMG")
        # Add argument for doing leave-one-subject-out
        parser.add_argument('--leave_one_subject_out', type=utils.str2bool, help='whether or not to do leave one subject out. Set to False by default.', default=False)
        # Add argument for leftout subject (indexed from 1)
        parser.add_argument('--leftout_subject', type=int, help='number of subject that is left out for cross validation, starting from subject 1', default=0)
        # Add parser for seed
        parser.add_argument('--seed', type=int, help='seed for reproducibility. Set to 0 by default.', default=0)
        # Add number of epochs to train for
        parser.add_argument('--epochs', type=int, help='number of epochs to train for. Set to 25 by default.', default=25)
        # Add argument for whether or not to use RMS
        parser.add_argument('--turn_on_rms', type=utils.str2bool, help='whether or not to use RMS. Set to False by default.', default=False)
        # Add argument for RMS input window size (resulting feature dimension to classifier)
        parser.add_argument('--rms_input_windowsize', type=int, help='RMS input window size. Set to 1000 by default.', default=1000)
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
        # Add argument for using phase spectrogram transform
        parser.add_argument('--turn_on_phase_spectrogram', type=utils.str2bool, help='whether or not to use phase spectrogram transform. Set to False by default.', default=False)
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
        # Add argument for reducing training data size while remaining stratified in terms of gestures and amount of data from each subject
        parser.add_argument('--reduce_training_data_size', type=utils.str2bool, help='whether or not to reduce training data size while remaining stratified in terms of gestures and amount of data from each subject. Set to False by default.', default=False)
        # Add argument for size of reduced training data
        parser.add_argument('--reduced_training_data_size', type=int, help='size of reduced training data. Set to 56000 by default.', default=56000)
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
        parser.add_argument('--target_normalize_subject', type=int, help='number of subject that is left out for target normalization, starting from subject 1', default=0)

        args = parser.parse_args()
        self.args = args

        return self.args
           
    
    def setup_for_dataset(self):

        def get_dataset(dataset):
            """
            Downloads dataset in the root rather than in Setup directory.
            """
            setup_dir = os.path.dirname(__file__)
            script_path = os.path.abspath(os.path.join(setup_dir, f"Get_Datasets/{dataset}.sh"))
            subprocess.run(['sh', script_path])

        def process_dataset(dataset, params=''):
            """
            Downloads dataset in the root rather than in Setup directory.
            """
            setup_dir = os.path.dirname(__file__)
            script_path = os.path.abspath(os.path.join(setup_dir, f"Get_Datasets/{dataset}.py"))
            subprocess.run(['python', script_path, params])

        """ Conducts safety checks on the self.args, downloads needed datasets, and imports the correct self.utils file. """
        
        self.exercises = False
        self.args.dataset = self.args.dataset.lower()

        if self.args.transition_classifier:
            self.args.include_transitions = True

        if self.args.include_transitions:
            transition_datasets = {"ninapro-db2", "ninapro_db2", "ninapro-db5", "ninapro_db5", "ninapro-db3", "ninapro_db3", "uciemg", "uci", "mcs"}
            assert self.args.dataset in transition_datasets, f"Transitions from rest to gesture are only preserved in {transition_datasets} datasets."

        if self.args.model == "MLP" or self.args.model == "SVC" or self.args.model == "RF":
            print("Warning: not using pytorch, many arguments will be ignored")
            if self.args.turn_on_unlabeled_domain_adaptation:
                raise NotImplementedError("Cannot use unlabeled domain adaptation with MLP, SVC, or RF")
            if self.args.pretrain_and_finetune:
                raise NotImplementedError("Cannot use pretrain and finetune with MLP, SVC, or RF")

        if (self.args.dataset in {"uciemg", "uci"}):
            if (not os.path.exists("./uciEMG")):
                print("uciEMG dataset does not exist yet. Downloading now...")
                get_dataset("get_UCI")
            from .Utils import utils_UCI as utils
            self.project_name = 'emg_benchmarking_uci'
            self.args.dataset = "uciemg"

            if self.args.include_transitions: 
                utils.include_transitions = True

        elif (self.args.dataset in {"ninapro-db2", "ninapro_db2"}):
            if (not os.path.exists("./NinaproDB2")):
                print("NinaproDB2 dataset does not exist yet. Downloading now...")
                get_dataset("get_NinaproDB2")
            from .Utils import utils_NinaproDB2 as utils
            self.project_name = 'emg_benchmarking_ninapro-db2'
            self.exercises = True
            if self.args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for ninapro-db2; only one session exists")
            if self.args.force_regression:
                assert self.args.exercises == [3], "Regression only implemented for exercise 3"
            self.args.dataset = 'ninapro-db2'

        elif (self.args.dataset in {"ninapro-db5", "ninapro_db5"}):
            if (not os.path.exists("./NinaproDB5")):
                print("NinaproDB5 dataset does not exist yet. Downloading now...")
                get_dataset("get_NinaproDB5")
                process_dataset("process_NinaproDB5")

            if (not os.path.exists("./DatasetsProcessed_hdf5/NinaproDB5/")):
                print("NinaproDB5 dataset not yet processed. Processing now")
                process_dataset("process_NinaproDB5")

            from .Utils import utils_NinaproDB5 as utils
            self.project_name = 'emg_benchmarking_ninapro-db5'
            self.exercises = True
            if self.args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for ninapro-db5; only one session exists")
            self.args.dataset = 'ninapro-db5'

        elif (self.args.dataset in {"ninapro-db3", "ninapro_db3"}):

            if (not os.path.exists("./NinaproDB3")):
                print("NinaproDB3 dataset does not exist yet. Downloading now...")
                get_dataset("get_NinaproDB3")

            from .Utils import utils_NinaproDB3 as utils
            assert self.args.exercises == [1] or self.args.partial_dataset_ninapro or (self.args.exercises == [3] and self.args.force_regression), "Exercise C cannot be used for classification due to missing data."
            self.project_name = 'emg_benchmarking_ninapro-db3'
            self.exercises = True
            if self.args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for ninapro-db3; only one session exists")
            
            if self.args.force_regression:
                print("NOTE: Subject 10 is missing gesture data for exercise 3 and is not used for regression.")
            
            assert not(self.args.force_regression and self.args.leftout_subject == 10), "Subject 10 is missing gesture data for exercise 3 and cannot be used. Please choose another subject."

            if self.args.force_regression and self.args.leftout_subject == 11: 
                # subject 10 is missing force data and is deleted internally 
                self.args.leftout_subject = 10

            self.args.dataset = 'ninapro-db3'

        elif (self.args.dataset.lower() == "myoarmbanddataset"):
            if (not os.path.exists("./myoarmbanddataset")):
                print("MyoArmbandDataset does not exist yet. Downloading now...")
                get_dataset("get_MyoArmbandDataset")
            from .Utils import utils_MyoArmbandDataset as utils
            self.project_name = 'emg_benchmarking_myoarmbanddataset'
            if self.args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for myoarmbanddataset; only one session exists")
            self.args.dataset = 'myoarmbanddataset'

        elif (self.args.dataset.lower() == "hyser"):
            if (not os.path.exists("./hyser")):
                print("Hyser dataset does not exist yet. Downloading now...")
                get_dataset("get_Hyser")
            from .Utils import utils_Hyser as utils
            self.project_name = 'emg_benchmarking_hyser'
            self.args.dataset = 'hyser'

        elif (self.args.dataset.lower() == "capgmyo"):
            if (not os.path.exists("./CapgMyo_B")):
                print("CapgMyo_B dataset does not exist yet. Downloading now...")
                get_dataset("get_CapgMyo_B")
            from .Utils import utils_CapgMyo as utils
            self.project_name = 'emg_benchmarking_capgmyo'
            if self.args.leave_one_session_out:
                utils.num_subjects = 10
            self.args.dataset = 'capgmyo'

        elif (self.args.dataset.lower() == "flexwear-hd"):
            if (not os.path.exists("./FlexWear-HD")):
                print("FlexWear-HD dataset does not exist yet. Downloading now...")
                get_dataset("get_FlexWearHD")
            from .Utils import utils_FlexWearHD as utils
            self.project_name = 'emg_benchmarking_flexwear-hd_dataset'
            # if self.args.leave_one_session_out:
                # raise ValueError("leave-one-session-out not implemented for FlexWear-HDDataset; only one session exists")
            self.args.dataset = 'flexwear-hd'

        elif (self.args.dataset.lower() == "sci"):
            from .Utils import utils_SCI as utils
            self.project_name = 'emg_benchmarking_sci'
            self.args.dataset = 'sci'
            assert not self.args.transfer_learning, "Transfer learning not implemented for SCI dataset"
            assert not self.args.leave_one_subject_out, "Leave one subject out not implemented for SCI dataset"

        elif (self.args.dataset.lower() == "mcs"):

            from .Utils import utils_MCS_EMG as utils
            if (not os.path.exists("./MCS_EMG")):
                print("MCS dataset does not exist yet. Downloading now...")
                get_dataset("get_MCS_EMG")

            if self.args.transition_classifier:
                if (not os.path.exists("./DatasetsProcessed_hdf5/MCS_EMG_transition_classifer/")):
                    print("MCS dataset not yet processed. Processing now")
                    process_dataset("process_MCS", "--transition_classifier=True")

                # TODO: Pass in arg parse to utils and use that instead
                utils.include_transitions = True
                utils.transition_classifier = True

            elif self.args.include_transitions: 
                if (not os.path.exists("./DatasetsProcessed_hdf5/MCS_EMG_include_transitions/")):
                    print("MCS dataset not yet processed for include transitions. Processing now")
                    process_dataset("process_MCS", "--include_transitions=True")

                utils.include_transitions = True
                utils.transition_classifier = self.args.transition_classifier

            else: 
                if (not os.path.exists("./DatasetsProcessed_hdf5/MCS_EMG/")):
                    print("MCS dataset not yet processed. Processing now")
                    process_dataset("process_MCS", "--include_transitions=False")      
                utils.include_transitions = False      
           
            self.project_name = 'emg_benchmarking_mcs'
            if self.args.full_dataset_mcs:
                print(f"Using the full dataset for MCS EMG")
                utils.gesture_labels = utils.gesture_labels_full
                utils.numGestures = len(utils.gesture_labels)
            else: 
                print(f"Using the partial dataset for MCS EMG")
                utils.gesture_labels = utils.gesture_labels_partial
                utils.numGestures = len(utils.gesture_labels)
            if self.args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for MCS_EMG; only one session exists")
            self.args.dataset = 'mcs'
            
        else: 
            print(self.args.dataset)
            if os.path.exists(f"DatasetsProcessed_hdf5/{self.args.dataset}/"):
                from .Utils import utils_generic as utils
                utils.initialize(self.args.dataset)
                self.project_name = f'emg_benchmarking_{self.args.dataset}'
            else:
                raise ValueError("Dataset not recognized. Please choose from 'uciemg', 'ninapro-db2', 'ninapro-db5', 'myoarmbanddataset', 'hyser'," + "'capgmyo', 'flexwear-hd', 'sci', or 'mcs'")
            
        # Safety Checks
        if self.args.turn_off_scaler_normalization:
            assert self.args.target_normalize == 0.0, "Cannot turn off scaler normalization and turn on target normalize at the same time"

        if utils.num_subjects == 1:
                assert not self.args.pretrain_and_finetune, "Cannot pretrain and finetune with only one subject"

        if self.args.force_regression:
            assert self.args.dataset in {"ninapro-db2", "ninapro-db3"}, "Regression only implemented for Ninapro DB2 and DB3 dataset." 
            assert not self.args.partial_dataset_ninapro, "Cannot use partial dataset for regression. Set exercises=3." 

        if (self.args.target_normalize > 0) and self.args.target_normalize_subject == 0:
            self.args.target_normalize_subject = self.args.leftout_subject
            print("Target normalize subject defaulting to leftout subject.")

        if self.args.transition_classifier:
            assert self.args.leave_one_subject_out, "Binary classifier only implemented for LOSO."

        if self.args.domain_generalization in {"IRM", "CORAL"}:
            assert not self.args.turn_on_unlabeled_domain_adaptation, "Domain generalization cannot be used with unlabeled domain adaptation currently."
            assert self.args.leave_one_subject_out, "Domain generalization can only be used with leave-one-subject-out currently."

            assert self.args.model in {"MLP", "resnet18"}, "Domain generalization can only be used with MLP or resnet18 currently."


        # Set Final Values
        # Add date and time to filename
        current_datetime = datetime.datetime.now()
        self.formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        self.utils = utils
        self.utils.args = self.args

        print("------------------------------------------------------------------------------------------------------------------------")
        print("Starting run at", self.formatted_datetime)
        print("------------------------------------------------------------------------------------------------------------------------")

        

        

    def print_params(self):
        for param, value in vars(self.args).items():
            if getattr(self.args, param):
                if param == "target_normalize_subject":
                    if self.args.target_normalize != 0.0:
                        print(f"The value of --{param} is {value}")
                else:
                    print(f"The value of --{param} is {value}")

    def set_exercise(self):
        """ Set the self.exercises for the partial dataset for Ninapro datasets. 
        """

        if self.args.partial_dataset_ninapro:
            if self.args.dataset == "ninapro-db2":
                self.args.exercises = [1]
            elif self.args.dataset == "ninapro-db5":
                self.args.exercises = [2]
            elif self.args.dataset == "ninapro-db3":
                self.args.exercises = [1]


    def set_env(self):
        env = self.Env()
        env.args = self.args
        env.exercises = self.exercises
        env.project_name = self.project_name
        env.formatted_datetime = self.formatted_datetime
        env.utils = self.utils
        env.leaveOut = int(self.args.leftout_subject)
        # TODO: fix this 
        if hasattr(self.utils, 'numGestures'):
            env.num_gestures = self.utils.numGestures
        else:
            env.num_gestures = None 
        env.seed = self.args.seed

        return env
