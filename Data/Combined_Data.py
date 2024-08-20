"""
Combined_Data.py
- Contains Combined_Data class, which is a wrapper class that repeats a given functions for all the data sets (X, Y, Label).
"""
from .X_Data import X_Data
from .Y_Data import Y_Data
from .Label_Data import Label_Data

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing, model_selection
import numpy as np

import multiprocessing

class Combined_Data():
    """Wrapper class that repeats a given functions for all the data. 
    """
    def __init__(self, x_obj, y_obj, label_obj, env):

        self.args = env.args
        self.utils = env.utils
        self.leaveOut = env.leaveOut
        self.exercises = env.exercises
        self.env = env

        self.X = x_obj
        self.Y = y_obj
        self.label = label_obj

        # Set seeds for reproducibility
        np.random.seed(self.args.seed)
    

    def process_ninapro(self):
        """ Appends exercise sets together and adds dimensions to data if necessary for Ninapro dataset values. 

        Returns:
            Torch with all data concatenated for each subject for each exercise.
        """

        # Remove Subject 10 who is missing most of exercise 3 data in DB3 

        if self.args.force_regression and self.args.dataset == "ninapro-db3":
            MISSING_SUBJECT = 10
            self.utils.num_subjects -= 1  
            del self.X.data[0][MISSING_SUBJECT-1] 
            del self.Y.data[0][MISSING_SUBJECT-1]
            del self.label.data[0][MISSING_SUBJECT-1]

        self.X.new_data = []
        self.Y.new_data = []
        self.label.new_data = []



        for subject in range(self.utils.num_subjects):
            
            self.X.subject_trials = []
            self.Y.subject_trials = []
            self.label.subject_trials = []
            
            # Store data for this subject across all exercises
            for exercise_set in range(len(self.X.data)):
                self.append_to_trials(exercise_set, subject)
            
            numGestures = self.concat_across_exercises()
            self.env.num_gestures  = numGestures # different gesture count for Ninapro
            self.append_to_new_data(self.X.concatenated_trials, self.Y.concatenated_trials, self.label.concatenated_trials)

            print(f"Subject {subject}: has {numGestures} gestures and self.label.shape {self.label.concatenated_trials.shape}") 

            # turn self.label.concatenated_trails from one hot encoding to labels
            gestures_to_append = np.argmax(self.label.concatenated_trials, axis=1)
            unique, counts = np.unique(gestures_to_append, return_counts=True)
            for i in unique:
                print(f"Gesture {i}: {counts[i]} ")

        self.set_new_data()

        del self.X.new_data, self.Y.new_data, self.label.new_data
        del self.X.subject_trials, self.Y.subject_trials, self.label.subject_trials

    def load_ninapro(self):
        emg = []
        labels = []

        if self.args.force_regression:
            forces = []

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
            for exercise in self.args.exercises:
                if (self.args.target_normalize > 0):
                    mins, maxes = self.utils.getExtrema(self.args.target_normalize_subject, self.args.target_normalize, exercise, self.args)
                    emg_async = pool.map_async(self.utils.getEMG, [(i+1, exercise, mins, maxes, self.args.target_normalize_subject, self.args) for i in range(self.utils.num_subjects)])

                else:
                    emg_async = pool.map_async(self.utils.getEMG, list(zip([(i+1) for i in range(self.utils.num_subjects)], exercise*np.ones(self.utils.num_subjects).astype(int), [self.args]*self.utils.num_subjects)))

                emg.append(emg_async.get()) # (EXERCISE SET, SUBJECT, TRIAL, CHANNEL, TIME)
                
                labels_async = pool.map_async(self.utils.getLabels, list(zip([(i+1) for i in range(self.utils.num_subjects)], exercise*np.ones(self.utils.num_subjects).astype(int), [self.args]*self.utils.num_subjects)))

                labels.append(labels_async.get())

                if self.args.force_regression:
                    assert(exercise == 3), "Regression only implemented for exercise 3"
                    forces_async = pool.map_async(self.utils.getForces, list(zip([(i+1) for i in range(self.utils.num_subjects)], exercise*np.ones(self.utils.num_subjects).astype(int))))
                    forces.append(forces_async.get())
                    
                assert len(emg[-1]) == len(labels[-1]), "Number of trials for EMG and labels do not match"
                if self.args.force_regression:
                    assert len(emg[-1]) == len(forces[-1]), "Number of trials for EMG and forces do not match"

        self.X.data = emg
        if self.args.force_regression:
            self.Y.data = forces
        else:
            self.Y.data = labels
        self.label.data = labels

    def load_other_datasets(self):

        emg = []
        labels = []
       
        if (self.args.target_normalize > 0):
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                if self.args.leave_one_session_out:
                    total_number_of_sessions = 2
                    mins, maxes = self.utils.getExtrema(self.args.target_normalize_subject, self.args.target_normalize, lastSessionOnly=False)
                    emg = []
                    labels = []
                    for i in range(1, total_number_of_sessions+1):
                        emg_async = pool.map_async(self.utils.getEMG_separateSessions, [(j+1, i, mins, maxes, self.args.target_normalize_subject) for j in range(self.utils.num_subjects)])
                        emg.extend(emg_async.get())
                        
                        labels_async = pool.map_async(self.utils.getLabels_separateSessions, [(j+1, i) for j in range(self.utils.num_subjects)])
                        labels.extend(labels_async.get())
                else:
                    mins, maxes = self.utils.getExtrema(self.args.target_normalize_subject, self.args.target_normalize)
                    
                    emg_async = pool.map_async(self.utils.getEMG, [(i+1, mins, maxes, self.args.target_normalize_subject) for i in range(self.utils.num_subjects)])

                    emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
                    
                    labels_async = pool.map_async(self.utils.getLabels, [(i+1) for i in range(self.utils.num_subjects)])
                    labels = labels_async.get()
        else: # Not target_normalize
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                if self.args.leave_one_session_out: # based on 2 sessions for each subject
                    total_number_of_sessions = 2
                    emg = []
                    labels = []
                    for i in range(1, total_number_of_sessions+1):
                        emg_async = pool.map_async(self.utils.getEMG_separateSessions, [(j+1, i) for j in range(self.utils.num_subjects)])
                        emg.extend(emg_async.get())
                        
                        labels_async = pool.map_async(self.utils.getLabels_separateSessions, [(j+1, i) for j in range(self.utils.num_subjects)])
                        labels.extend(labels_async.get())
                    
                else: # Not leave one session out
                    dataset_identifiers = self.utils.num_subjects
                        
                    emg_async = pool.map_async(self.utils.getEMG, [(i+1) for i in range(dataset_identifiers)])
                    emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
                    
                    labels_async = pool.map_async(self.utils.getLabels, [(i+1) for i in range(dataset_identifiers)])
                    labels = labels_async.get()

        self.X.data = emg
        self.Y.data = labels
        self.label.data = labels



    def load_data(self):

        if self.exercises:
            self.load_ninapro()
            self.process_ninapro()
        else:
            self.load_other_datasets()

            

        assert len(self.X.data[-1]) == len(self.Y.data[-1]), "Number of trials for X and Y do not match."
        assert len(self.Y.data[-1]) == len(self.label.data[-1]), "Number of trials for Y and Labels do not match."

    def set_values(self, attr, value):
        self.X.set_values(attr, value)
        self.Y.set_values(attr, value)
        self.label.set_values(attr, value)

    def scaler_normalize_emg(self):
        """
        Sets the global_low_value, global_high_value, and scaler for X (EMG) data. Also shares the indices/split across all data sets (which is needed since randomly generated). 

        """

        # These can be tuned to change the normalization
        # This is the coefficient for the standard deviation
        # used for the magnitude images. In practice, these
        # should be fairly small so that images have more
        # contrast
        if self.args.turn_on_rms:
            # This tends to be small because in pracitice
            # the RMS is usually much smaller than the raw EMG
            # NOTE: Should check why this is the case
            sigma_coefficient = 0.1
        else:
            # This tends to be larger because the raw EMG
            # is usually much larger than the RMS
            sigma_coefficient = 0.5

        def compute_emg_in():
            """Prepares emg_in by concatenating EMG data and reshaping if neccessary. emg_in is a temporary variable used to compute the scaler.
            """
            emg = self.X.data
            leaveOutIndices = []
         
            # Running LOSO standardization
            emg_in = np.concatenate([np.array(i.view(len(i), self.X.length*self.X.width)) for i in emg[:(self.leaveOut-1)]] + [np.array(i.view(len(i), self.X.length*self.X.width)) for i in emg[self.leaveOut:]], axis=0, dtype=np.float32)
        
                        
            return emg_in, leaveOutIndices
    
        def compute_scaler(emg_in):
            """Compues global low, global high, and scaler for EMG data.

            Args:
                emg_in: incoming EMG data to scaler normalize

            Returns:
                global_low_value, global_high_value, scaler: global low value, global high value, and scaler for EMG data.
            """
  

            global_low_value = emg_in.mean() - sigma_coefficient*emg_in.std()
            global_high_value = emg_in.mean() + sigma_coefficient*emg_in.std()

            # Normalize by electrode
            emg_in_by_electrode = emg_in.reshape(-1, self.X.length, self.X.width)

            # Assuming emg is your initial data of shape (SAMPLES, 16, 50)
            # Reshape data to (SAMPLES*50, 16)
            emg_reshaped = emg_in_by_electrode.reshape(-1, self.utils.numElectrodes)

            # Initialize and fit the scaler on the reshaped data
            # This will compute the mean and std dev for each electrode across all samples and features
            scaler = preprocessing.StandardScaler()
            scaler.fit(emg_reshaped)
            
            # Repeat means and std_devs for each time point using np.repeat
            scaler.mean_ = np.repeat(scaler.mean_, self.X.width)
            scaler.scale_ = np.repeat(scaler.scale_, self.X.width)
            scaler.var_ = np.repeat(scaler.var_, self.X.width)
            scaler.n_features_in_ = self.X.width*self.utils.numElectrodes

            del emg_in
            del emg_in_by_electrode
            del emg_reshaped

            return global_low_value, global_high_value, scaler
    
        global_low_value, global_high_value, scaler = None, None, None

        if (not self.args.turn_off_scaler_normalization and not (self.args.target_normalize > 0)):
            emg_in, leaveOutIndices = compute_emg_in()
            global_low_value, global_high_value, scaler = compute_scaler(emg_in)
            self.set_values(attr="leaveOutIndices", value=leaveOutIndices)

        # Values needed to compute image
        self.set_values(attr="global_high_value", value=global_high_value)
        self.set_values(attr="global_low_value", value=global_low_value)
        self.set_values(attr="scaler", value=scaler)

    def append_to_trials(self, exercise_set, subject):
        self.X.append_to_trials(exercise_set, subject)
        self.Y.append_to_trials(exercise_set, subject)
        self.label.append_to_trials(exercise_set, subject)

    def concat_across_exercises(self):

        # Labels always needs to go first to provide indices for partial dataset
        numGestures, indices_for_partial_dataset = self.label.concat_across_exercises()
        self.X.concat_across_exercises(indices_for_partial_dataset)
        self.Y.concat_across_exercises(indices_for_partial_dataset)

        return numGestures
    
    def append_to_new_data(self, X_concatenated_trials, Y_concatenated_trials, label_concatenated_trials):
        self.X.append_to_new_data(X_concatenated_trials)
        self.Y.append_to_new_data(Y_concatenated_trials)
        self.label.append_to_new_data(label_concatenated_trials)

    def set_new_data(self):
        self.X.set_new_data()
        self.Y.set_new_data()
        self.label.set_new_data()

    def load_images(self, base_foldername_zarr):
        self.X.load_images(base_foldername_zarr)
        
   