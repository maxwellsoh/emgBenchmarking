import torch
import numpy as np
from .Data import Data
import multiprocessing

class Y_Data(Data):
    
    def __init__(self, env):
        super().__init__("Y", env)


    def load_data(self, exercises):
        """ Sets self.data to force data if self.args.force_regression, otherwise sets self.data to labels. """


        def load_labels_ninapro():
            
           
            if self.args.multiprocessing:
                    print("Multiprocessing occasionally has issues. If timing out, try running with multiprocesing=False")
            
                    labels = []
                    with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                        for exercise in self.args.exercises:
                            labels_async = pool.map_async(self.utils.getLabels, list(zip([(i+1) for i in range(self.utils.num_subjects)], exercise*np.ones(self.utils.num_subjects).astype(int), [self.args]*self.utils.num_subjects)))
                            labels.append(labels_async.get()) 

            else: 
                labels = []
                for exercise in self.args.exercises:
                    exercise_labels = []
                    for i in range(self.utils.num_subjects):
                        label = self.utils.getLabels((i+1, exercise, self.args))
                        exercise_labels.append(label)
                    labels.append(exercise_labels)

            return labels

        def load_labels_other_datasets():
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                if self.args.leave_one_session_out:
                    labels = []
                    for i in range(1, self.utils.num_sessions+1):
                        labels_async = pool.map_async(self.utils.getLabels_separateSessions, [(j+1, i) for j in range(self.utils.num_subjects)])
                        labels.extend(labels_async.get())
                else:
                    labels_async = pool.map_async(self.utils.getLabels, [(i+1) for i in range(self.utils.num_subjects)])
                    labels = labels_async.get()
            
            return labels

        def load_labels():
            if exercises:
                return load_labels_ninapro()
            else:
                return load_labels_other_datasets()

        def load_forces():
            
            forces = []
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//8) as pool:
                for exercise in self.args.exercises:
                    assert(exercise == 3), "Regression only implemented for exercise 3"
                    forces_async = pool.map_async(self.utils.getForces, list(zip([(i+1) for i in range(self.utils.num_subjects)], exercise*np.ones(self.utils.num_subjects).astype(int))))
                    forces.append(forces_async.get())

            return forces

        if self.args.force_regression:
            self.data = load_forces()
        else:
            self.data = load_labels()


    # Process Ninapro Helper
    def append_to_trials(self, exercise_set, subject):
        """Appends EMG data for a given subject across all exercises to self.X.subject_trials. Helper function for process_ninapro.
        """

        if self.args.force_regression:
            self.subject_trials.append(self.data[exercise_set][subject][:,:,0])
        else:
            self.subject_trials.append(self.data[exercise_set][subject])

    def concat_across_exercises(self, indices_for_partial_dataset=None):
        """Concatenates forces/labels across exercises for a given subject. 
        
        Helper function for process_ninapro. If self.args.force_regression, concatenates forces. Otherwise, processes labels from one hot encoding and concatenates. 
        """
        if self.args.force_regression:
            self.concatenated_trials = np.concatenate(self.subject_trials, axis=0)
        else:
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

            if self.args.partial_dataset_ninapro:
                indices_for_partial_dataset = np.array([indices for indices, label in enumerate(concatenated_labels) if label in desired_gesture_labels])
                concatenated_labels = concatenated_labels[indices_for_partial_dataset]
                # convert labels to indices
                label_to_index = {label: index for index, label in enumerate(desired_gesture_labels)}
                concatenated_labels = [label_to_index[label] for label in concatenated_labels]

            # Convert to one hot encoding
            concatenated_labels = np.eye(np.max(concatenated_labels) + 1)[concatenated_labels] # (TRIAL, GESTURE)

            self.concatenated_trials = concatenated_labels


    def append_flexwear_unlabeled_to_finetune_unlabeled_list(self, flexwear_unlabeled_data):
        assert self.args.load_unlabeled_data_flexwearhd, "Cannot append unlabeled data if load_unlabeled_data_flexwearhd is turned off."
        self.finetune_unlabeled_list.append(np.zeros(flexwear_unlabeled_data.shape[0]))
