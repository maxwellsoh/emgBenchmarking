from .Data_Split_Strategy import Data_Split_Strategy
import numpy as np
from sklearn import preprocessing, model_selection

class Held_Out_Test(Data_Split_Strategy):
    """
    Splits such that train uses train_indices, validation uses validation_indices, and test uses test_indices.
    """

    def __init__(self, X_data, Y_data, label_data, env):
        super().__init__(X_data, Y_data, label_data, env)

        # Set seeds for reproducibility
        np.random.seed(self.args.seed)
    

    def combine_data(self):
        """
        Combines data across the 0th axis.
        """

        X_combined_data = np.concatenate([np.array(i) for i in self.X.data], axis=0, dtype=np.float16)
        Y_combined_data = np.concatenate([np.array(i) for i in self.Y.data], axis=0, dtype=np.float16)
        label_combined_data = np.concatenate([np.array(i) for i in self.label.data], axis=0, dtype=np.float16)

        return X_combined_data, Y_combined_data, label_combined_data

    def train_from_train_indices(self, X_combined_data, Y_combined_data, label_combined_data):
        """
        Create train set by taking train indices from data.
        """

        self.X.train_from_train_indices(X_combined_data)
        self.Y.train_from_train_indices(Y_combined_data)
        self.label.train_from_train_indices(label_combined_data)
    
    def validation_from_validation_indices(self, X_combined_data, Y_combined_data, label_combined_data):
        """
        Create validation set by taking validation indices from data.
        """

        self.X.validation_from_validation_indices(X_combined_data)
        self.Y.validation_from_validation_indices(Y_combined_data)
        self.label.validation_from_validation_indices(label_combined_data)

    def convert_datasets_to_tensors(self):
        """
        Covert train, validation, and test to tensors.
        """
        super().convert_to_16_tensors("train")
        super().convert_to_16_tensors("validation")
        super().convert_to_16_tensors("test")

    def del_combined_data(self):
        """
        Delete combined_data variable.
        """
        self.X.del_combined_data()
        self.Y.del_combined_data()
        self.label.del_combined_data()
    
    def split(self):
        """Train using train_indices. Split validation indices into validation and test using labels to stratify.""" 

        X_combined_data, Y_combined_data, label_combined_data = self.combine_data()
        self.train_from_train_indices(X_combined_data, Y_combined_data, label_combined_data)
        self.validation_from_validation_indices(X_combined_data, Y_combined_data, label_combined_data)
       
        # Split validation into validation and test 50/50
        self.X.validation, self.X.test, \
        self.Y.validation, self.Y.test, \
        self.label.validation, self.label.test \
        = model_selection.train_test_split(
            self.X.validation, 
            self.Y.validation, 
            self.label.validation, 
            test_size=0.5, 
            stratify=self.label.validation
        )

        self.convert_datasets_to_tensors()

        del X_combined_data
        del Y_combined_data
        del label_combined_data
      
        self.del_data()
        super().test_from_validation() # NOTE: are we intentionally splitting twice? 
        super().print_set_shapes()
        super().all_sets_to_tensor()
