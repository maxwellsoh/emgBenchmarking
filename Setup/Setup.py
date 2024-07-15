
import os
import subprocess
import datetime

class Run_Setup():
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
        self.utils = None



    def read_input(self):
        raise NotImplementedError("Subclasses must implement read_input()")

    
    def setup_for_dataset(self):
        """ Conducts safety checks on the args, downloads needed datasets, and imports the correct utils file. """

        
        exercises = False
        self.args.dataset = self.args.dataset.lower()

        if self.args.model == "MLP" or self.args.model == "SVC" or self.args.model == "RF":
            print("Warning: not using pytorch, many arguments will be ignored")
            if self.args.turn_on_unlabeled_domain_adaptation:
                raise NotImplementedError("Cannot use unlabeled domain adaptation with MLP, SVC, or RF")
            if self.args.pretrain_and_finetune:
                raise NotImplementedError("Cannot use pretrain and finetune with MLP, SVC, or RF")

        if self.args.force_regression:
            assert self.args.dataset in {"ninapro-db3", "ninapro_db3"}, "Regression only implemented for Ninapro DB2 and DB3 dataset."

        if (self.args.dataset in {"uciemg", "uci"}):
            if (not os.path.exists("./uciEMG")):
                print("uciEMG dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--UCI'])
            import utils_UCI as utils
            project_name = 'emg_benchmarking_uci'
            self.args.dataset = "uciemg"

        elif (self.args.dataset in {"ninapro-db2", "ninapro_db2"}):
            if (not os.path.exists("./NinaproDB2")):
                print("NinaproDB2 dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--NinaproDB2'])
            import utils_NinaproDB2 as utils
            project_name = 'emg_benchmarking_ninapro-db2'
            exercises = True
            if self.args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for ninapro-db2; only one session exists")
            if self.args.force_regression:
                assert self.args.exercises == [3], "Regression only implemented for exercise 3"
            self.args.dataset = 'ninapro-db2'

        elif (self.args.dataset in { "ninapro-db5", "ninapro_db5"}):
            if (not os.path.exists("./NinaproDB5")):
                print("NinaproDB5 dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--NinaproDB5'])
                subprocess.run(['python', './process_NinaproDB5.py'])
            import utils_NinaproDB5 as utils
            project_name = 'emg_benchmarking_ninapro-db5'
            exercises = True
            if self.args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for ninapro-db5; only one session exists")
            self.args.dataset = 'ninapro-db5'

        elif (self.args.dataset in {"ninapro-db3", "ninapro_db3"}):
            import utils_NinaproDB3 as utils

            assert self.args.exercises == [1] or self.args.partial_dataset_ninapro or (self.args.exercises == [3] and self.args.force_regression), "Exercise C cannot be used for classification due to missing data."
            project_name = 'emg_benchmarking_ninapro-db3'
            exercises = True
            if self.args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for ninapro-db3; only one session exists")
            
            if self.args.force_regression:
                print("NOTE: Subject 10 is missing gesture data for exercise 3 and is not used for regression.")
            
            assert not(self.args.force_regression and self.args.leftout_subject == 10), "Subject 10 is missing gesture data for exercise 3 and cannot be used. Please choose another subject."

            if self.args.force_regression and self.args.leftout_subject == 11: 
                self.args.leftout_subject = 10
                # subject 10 is missing force data and is deleted internally 

            self.args.dataset = 'ninapro-db3'

        elif (self.args.dataset.lower() == "myoarmbanddataset"):
            if (not os.path.exists("./myoarmbanddataset")):
                print("myoarmbanddataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--MyoArmbandDataset'])
            import utils_MyoArmbandDataset as utils
            project_name = 'emg_benchmarking_myoarmbanddataset'
            if self.args.leave_one_session_out:
                raise ValueError("leave-one-session-out not implemented for myoarmbanddataset; only one session exists")
            self.args.dataset = 'myoarmbanddataset'

        elif (self.args.dataset.lower() == "hyser"):
            if (not os.path.exists("./hyser")):
                print("Hyser dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--Hyser'])
            import utils_Hyser as utils
            project_name = 'emg_benchmarking_hyser'
            self.args.dataset = 'hyser'

        elif (self.args.dataset.lower() == "capgmyo"):
            if (not os.path.exists("./CapgMyo_B")):
                print("CapgMyo_B dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--CapgMyo_B'])
            import utils_CapgMyo as utils
            project_name = 'emg_benchmarking_capgmyo'
            if self.args.leave_one_session_out:
                utils.num_subjects = 10
            self.args.dataset = 'capgmyo'

        elif (self.args.dataset.lower() == "flexwear-hd"):
            if (not os.path.exists("./FlexWear-HD")):
                print("FlexWear-HD dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--FlexWearHD_Dataset'])
            import utils_FlexWearHD as utils
            project_name = 'emg_benchmarking_flexwear-hd_dataset'
            # if self.args.leave_one_session_out:
                # raise ValueError("leave-one-session-out not implemented for FlexWear-HDDataset; only one session exists")
            self.args.dataset = 'flexwear-hd'

        elif (self.args.dataset.lower() == "sci"):
            import utils_SCI as utils
            project_name = 'emg_benchmarking_sci'
            self.args.dataset = 'sci'
            assert not self.args.transfer_learning, "Transfer learning not implemented for SCI dataset"
            assert not self.args.leave_one_subject_out, "Leave one subject out not implemented for SCI dataset"

        elif (self.args.dataset.lower() == "mcs"):
            if (not os.path.exists("./MCS_EMG")):
                print("MCS dataset does not exist yet. Downloading now...")
                subprocess.run(['python', './get_datasets.py', '--MCS_EMG'])

            project_name = 'emg_benchmarking_mcs'
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
            raise ValueError("Dataset not recognized. Please choose from 'uciemg', 'ninapro-db2', 'ninapro-db5', 'myoarmbanddataset', 'hyser'," +
                            "'capgmyo', 'flexwear-hd', 'sci', or 'mcs'")
            
        # Safety Check 
        if self.args.turn_off_scaler_normalization:
            assert self.args.target_normalize == 0.0, "Cannot turn off scaler normalization and turn on target normalize at the same time"

        if utils.num_subjects == 1:
                assert not self.args.pretrain_and_finetune, "Cannot pretrain and finetune with only one subject"

    

        # Add date and time to filename
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        print("------------------------------------------------------------------------------------------------------------------------")
        print("Starting run at", formatted_datetime)
        print("------------------------------------------------------------------------------------------------------------------------")

        self.exercises = exercises
        self.project_name = project_name
        self.formatted_datetime = formatted_datetime
        self.utils = utils


    def print_params(self):
        for param, value in vars(self.args).items():
            if getattr(self.args, param):
                print(f"The value of --{param} is {value}")

                if param == "target_normalize_subject":
                    print("Target normalize is defaulting to leftout subject.")

    def set_exercise(self):
        """ 
        Set the exercises for the partial dataset for Ninapro datasets. 
        """

        if self.args.partial_dataset_ninapro:
            if self.args.dataset == "ninapro-db2":
                self.args.exercises = [1]
            elif self.args.dataset == "ninapro-db5":
                self.args.exercises = [2]
            elif self.args.dataset == "ninapro-db3":
                self.args.exercises = [1]
    
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


    def setup_env(self):
        env = self.Setup()
        env.args = self.args
        env.exercises = self.exercises
        env.project_name = self.project_name
        env.formatted_datetime = self.formatted_datetime
        env.utils = self.utils
        env.leaveOut = int(self.args.leftout_subject)
        env.num_gestures = self.utils.numGestures
        env.seed = self.args.seed

        return env

    