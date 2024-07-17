"""
Label_Data.py
- Contains Label_Data class, which is a subclass of Data.
- Label_Data is responsible for loading and processing labels of the given dataset.
"""
import torch
import numpy as np
from .Data import Data
import multiprocessing

class Label_Data(Data):

    def __init__(self, env):
            super().__init__("Label",env)

            # Set seeds for reproducibility
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed(self.args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    # Process Ninapro Helper
    def append_to_trials(self, exercise_set, subject):
        """Appends EMG data for a given subject across all exercises to self.X.subject_trials. Helper function for process_ninapro.
        """
        self.subject_trials.append(self.data[exercise_set][subject])

    def concat_across_exercises(self):

        """Concatenates label data across exercises for a given subject.
        Helper function for process_ninapro.
        """

        # Convert from one hot encoding to labels
        # Assuming labels are stored separately and need to be concatenated end-to-end


        labels_set = []
        index_to_start_at = 0
        for i in range(len(self.subject_trials)):
            subject_labels_to_concatenate = [x + index_to_start_at if x != 0 else 0 for x in np.argmax(self.subject_trials[i], axis=1)]
            if self.args.dataset == "ninapro-db5":
                index_to_start_at = max(subject_labels_to_concatenate)
            labels_set.append(subject_labels_to_concatenate)

        if self.args.partial_dataset_ninapro:
            desired_gesture_labels = self.utils.partial_gesture_indices

        concatenated_labels = np.concatenate(labels_set, axis=0) # (TRIAL)

        indices_for_partial_dataset = None
        if self.args.partial_dataset_ninapro:
            indices_for_partial_dataset = np.array([indices for indices, label in enumerate(concatenated_labels) if label in desired_gesture_labels])
            
            concatenated_labels = concatenated_labels[indices_for_partial_dataset]
        
            # convert labels to indices
            label_to_index = {label: index for index, label in enumerate(desired_gesture_labels)}
            concatenated_labels = [label_to_index[label] for label in concatenated_labels]
        

        # Convert to one hot encoding
        concatenated_labels = np.eye(np.max(concatenated_labels) + 1)[concatenated_labels] # (TRIAL, GESTURE)

        self.concatenated_trials = concatenated_labels

        return indices_for_partial_dataset
        

    def append_flexwear_unlabeled_to_finetune_unlabeled_list(self, flexwear_unlabeled_data):
        assert self.args.load_unlabeled_data_flexwearhd, "Cannot append unlabeled data if load_unlabeled_data_flexwearhd is turned off."
        self.finetune_unlabeled_list.append(np.zeros(flexwear_unlabeled_data.shape[0]))
