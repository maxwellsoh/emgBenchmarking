"""
X_Data.py
- This file contains the class definition for the X_Data class, which is a subclass of the Data class.
- The X_Data class is used to load and process EMG data for the given dataset.
"""
import torch
import numpy as np
from .Data import Data
import multiprocessing

from tqdm import tqdm
import os
import zarr

class X_Data(Data):

    def __init__(self, env):
        super().__init__("X", env)

        # Set seeds for reproducibility
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # EMG specific values
        self.width = None
        self.length = None

        self.global_low_value = None
        self.global_high_value = None
        self.scaler = None
        self.train_indices = None
        self.validation_indices = None

    # Load EMG Data
    def load_data(self):
        """ Sets self.data to EMG data. (emg) """

        def load_EMG_ninapro():
            """Gets the EMG data for Ninapro datasets.

            Returns:
                emg (EXERCISE SET, SUBJECT, TRIAL, CHANNEL, TIME): EMG data for the dataset with target_normalization (if applicable).
            """

            emg = []
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                for exercise in self.args.exercises:
                    if (self.args.target_normalize > 0):
                        mins, maxes = self.utils.getExtrema(self.args.target_normalize_subject, self.args.target_normalize, exercise, self.args)
                        emg_async = pool.map_async(self.utils.getEMG, [(i+1, exercise, mins, maxes, self.args.target_normalize_subject, self.args) for i in range(self.utils.num_subjects)])

                    else:
                        emg_async = pool.map_async(self.utils.getEMG, [(i+1, exercise, self.args) for i in range(self.utils.num_subjects)])

                    emg.append(emg_async.get()) # (EXERCISE SET, SUBJECT, TRIAL, CHANNEL, TIME)

            return emg

        def load_EMG_other_datasets(): 
            """Loads the EMG data for other, non Ninapro datasets.

            Returns:
                emg (SUBJECT, TRIAL, CHANNEL, TIME STEP): EMG data for the dataset with target_normalization (if applicable). 
            """
            emg = []
            if (self.args.target_normalize > 0):
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                    mins, maxes = self.utils.getExtrema(self.args.target_normalize_subject, self.args.target_normalize)
                    if self.args.leave_one_session_out:
                        emg = []
                        labels = []
                        for i in range(1, self.utils.num_sessions+1):
                            emg_async = pool.map_async(self.utils.getEMG_separateSessions, [(j+1, i, mins, maxes, self.args.target_normalize_subject) for j in range(self.utils.num_subjects)])
                            emg.extend(emg_async.get())

                    else:
                        emg_async = pool.map_async(self.utils.getEMG, [(i+1, mins, maxes, self.args.leftout_subject + 1) for i in range(self.utils.num_subjects)])

                        emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
            else: 
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                    if self.args.leave_one_session_out:
            
                        emg = []
                        labels = []
                        for i in range(1, self.utils.num_sessions+1):
                            emg_async = pool.map_async(self.utils.getEMG_separateSessions, [(j+1, i) for j in range(self.utils.num_subjects)])
                            emg.extend(emg_async.get())
                
                    else: # Not leave one session out
                        emg_async = pool.map_async(self.utils.getEMG, [(i+1) for i in range(self.utils.num_subjects)])
                        emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
            return emg

        if self.exercises:
            # Ninapro datasets have to be processed.
            self.data = load_EMG_ninapro()
        else:

            self.data = load_EMG_other_datasets()


    def print_data_information(self):

        print("Number of Samples (across all participants): ", sum([e.shape[0] for e in self.data]))
        print("Number of Electrode Channels (length of EMG): ", self.length)
        print("Number of Timesteps per Trial (width of EMG):", self.width)


    # Process Ninapro Helper
    def append_to_trials(self, exercise_set, subject):
        """Appends EMG data for a given subject across all exercises to self.X.subject_trials. Helper function for process_ninapro.
        """
        self.subject_trials.append(self.data[exercise_set][subject])

    # Helper for loading EMG data for Ninapro
    def concat_across_exercises(self, indices_for_partial_dataset=None):
        """Concatenates EMG data across exercises for a given subject.
        Helper function for process_ninapro.
        """
        self.concatenated_trials = np.concatenate(self.subject_trials, axis=0)

        if self.args.partial_dataset_ninapro:
            self.concatenated_trials = self.concatenated_trials[indices_for_partial_dataset]

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


    
    def load_images(self):
        """Updates self.data to be the loaded images for EMG data. Returns flexwear_unlabeled_data if self.args.load_unlabeled_data_flexwearhd.
        
        If dataset exists, loads images. Otherwise, creates imaeges and saves in directory. 

        Returns:
            flexwear_unlabeled_data: unlabeled data if self.args.load_unlabeled_data_flexwearhd
        """
        assert self.utils is not None, "self.utils is not defined. Please run initialize() first."

        base_foldername_zarr = self.create_foldername_zarr()
        self.length = self.data[0].shape[1]
        self.width = self.data[0].shape[2]
        flexwear_unlabeled_data = None

        emg = self.data # should already be defined as emg using load_data
        image_data = []
        for x in tqdm(range(len(emg)), desc="Number of Subjects "):
            if self.args.held_out_test:
                subject_folder = f'subject{x}/'
            elif self.args.leave_one_session_out:
                subject_folder = f'session{x}/'
            else:
                subject_folder = f'LOSO_subject{x}/'
            foldername_zarr = base_foldername_zarr + subject_folder
            
            subject_or_session = "session" if self.args.leave_one_session_out else "subject"
            print(f"Attempting to load dataset for {subject_or_session}", x, "from", foldername_zarr)

            print("Looking in folder: ", foldername_zarr)
            # Check if the folder (dataset) exists, load if yes, else create and save
            if os.path.exists(foldername_zarr):
                # Load the dataset
                dataset = zarr.open(foldername_zarr, mode='r')
                print(f"Loaded dataset for {subject_or_session} {x} from {foldername_zarr}")
                if self.args.load_few_images:
                    image_data += [dataset[:10]]
                else: 
                    image_data += [dataset[:]]
            else:
                print(f"Could not find dataset for {subject_or_session} {x} at {foldername_zarr}")
                # Get images and create the dataset
                if (self.args.target_normalize > 0):
                    self.scaler = None
                images = self.utils.getImages(
                    emg[x], 
                    self.scaler, 
                    self.length, 
                    self.width,
                    turn_on_rms=self.args.turn_on_rms, 
                    rms_windows=self.args.rms_input_windowsize,
                    turn_on_magnitude=self.args.turn_on_magnitude, 
                    global_min=self.global_low_value, 
                    global_max=self.global_high_value,
                    turn_on_spectrogram=self.args.turn_on_spectrogram, 
                    turn_on_cwt=self.args.turn_on_cwt,
                    turn_on_hht=self.args.turn_on_hht
                )
                images = np.array(images, dtype=np.float16)
                
                # Save the dataset
                if self.args.save_images:
                    os.makedirs(foldername_zarr, exist_ok=True)
                    dataset = zarr.open(foldername_zarr, mode='w', shape=images.shape, dtype=images.dtype, chunks=True)
                    dataset[:] = images
                    print(f"Saved dataset for subject {x} at {foldername_zarr}")
                else:
                    print(f"Did not save dataset for subject {x} at {foldername_zarr} because save_images is set to False")
                image_data += [images]
                
        if self.args.load_unlabeled_data_flexwearhd:
            unlabeled_images = self.utils.getImages(
                unlabeled_online_data, 
                self.scaler, 
                self.length, 
                self.width,
                turn_on_rms=self.args.turn_on_rms, 
                rms_windows=self.args.rms_input_windowsize,
                turn_on_magnitude=self.args.turn_on_magnitude, 
                global_min=self.global_low_value, 
                global_max=self.global_high_value,
                turn_on_spectrogram=self.args.turn_on_spectrogram, 
                turn_on_cwt=self.args.turn_on_cwt,
                turn_on_hht=self.args.turn_on_hht
            )
            unlabeled_images = np.array(unlabeled_images, dtype=np.float16)
            flexwear_unlabeled_data = unlabeled_images
            self.flexwear_unlabeled_data = flexwear_unlabeled_data
            del unlabeled_images, unlabeled_online_data

        self.data = image_data
        

        return flexwear_unlabeled_data
        
    # Helper for leave_one_session_out

    def append_flexwear_unlabeled_to_finetune_unlabeled_list(self, flexwear_unlabeled_data):
        assert self.args.load_unlabeled_data_flexwearhd, "Cannot append unlabeled data if load_unlabeled_data_flexwearhd is turned off."
        self.finetune_unlabeled_list.append(flexwear_unlabeled_data)
   