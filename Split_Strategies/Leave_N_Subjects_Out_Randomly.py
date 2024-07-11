from .Data_Split_Strategy import Data_Split_Strategy
class Leave_N_Subjects_Out_Randomly(Data_Split_Strategy):
    """
    Splits such that validate uses left out subjects (leaveOutIndices) and train uses the rest.
    """

    def __init__(self, X_data, Y_data, label_data, env):
        super().__init__(X_data, Y_data, label_data, env)

    
    def validation_from_leave_out_indices(self):
        """
        Create validation sets by taking leaveOutIndices from data. 
        """
        self.X.validation_from_leave_out_indices()
        self.Y.validation_from_leave_out_indices()
        self.label.validation_from_leave_out_indices()

    def train_from_non_leave_out_indices(self):
        self.X.train_from_non_leave_out_indices()
        self.Y.train_from_non_leave_out_indices()
        self.label.train_from_non_leave_out_indices()

    def convert_datasets_to_tensors(self):
        """
        Convert validation and training sets to tensors.
        """
        super().convert_to_16_tensors("validation")
        super().convert_to_16_tensors("train")

    def split(self):
        """
        Split data into training and validation sets.

        This method creates validation sets from leave out subjects (leaveOutIndices) and training sets from the rest of the subjects, converts the data to tensors, and deletes the original data. 
        """

        self.validation_from_leave_out_indices()
        self.train_from_non_leave_out_indices()
        self.convert_datasets_to_tensors()
        self.del_data()
        super().test_from_validation()
        super().print_set_shapes()
        super().all_sets_to_tensor()
     