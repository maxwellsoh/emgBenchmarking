from .cross_validation_utilities import train_test_split as tts # custom train test split to split stratified without shuffling
class Data_Split_Strategy():
    """
    Base strategy. Serves as a wrapper to hold X, Y, and label objects. 
    """

    def __init__(self, X_data, Y_data, label_data, env):

        self.args =  env.args
        self.utils = env.utils
        self.leaveOut = env.leaveOut
        
        self.X = X_data
        self.Y = Y_data
        self.label = label_data

    def split(self):
        raise NotImplementedError("Subclasses must implement split()")

    def convert_to_16_tensors(self, set_to_convert):
        self.X.convert_to_16_tensors(set_to_convert)
        self.Y.convert_to_16_tensors(set_to_convert)
        self.label.convert_to_16_tensors(set_to_convert)

    def concatenate_sessions(self, set_to_assign, set_to_concat):
        """
        Useful for when set_to_assign and set_to_concat are both instance variables. 
        """
        self.X.concatenate_sessions(set_to_assign, set_to_concat)
        self.Y.concatenate_sessions(set_to_assign, set_to_concat)
        self.label.concatenate_sessions(set_to_assign, set_to_concat)

    def del_data(self):
        self.X.del_data()
        self.Y.del_data()
        self.label.del_data()

    def print_set_shapes(self):
        self.X.print_set_shapes()
        self.Y.print_set_shapes()
        self.label.print_set_shapes()

    def validation_from(self, X_new_data, Y_new_data, label_new_data):
        self.X.validation_from(X_new_data)
        self.Y.validation_from(Y_new_data)
        self.label.validation_from(label_new_data)

    def train_from(self, X_new_data, Y_new_data, label_new_data):
        self.X.train_from(X_new_data)
        self.Y.train_from(Y_new_data)
        self.label.train_from(label_new_data)

    def train_from_self_tensor(self):
        self.X.set_to_self_tensor("train")
        self.Y.set_to_self_tensor("train")
        self.label.set_to_self_tensor("train")

    def validation_from_self_tensor(self):
        self.X.set_to_self_tensor("validation")
        self.Y.set_to_self_tensor("validation")
        self.label.set_to_self_tensor("validation")

    def test_from_validation(self):
        
        self.X.test, self.X.validation, \
        self.Y.test, self.Y.validation, \
        self.label.test, self.label.validation \
        = tts.train_test_split(
            self.X.validation, 
            self.Y.validation, 
            test_size=0.5, 
            stratify=self.label.validation,
            random_state=self.args.seed,
            shuffle=(not self.args.train_test_split_for_time_series)
        )

    def all_sets_to_tensor(self):
        self.X.all_sets_to_tensor()
        self.Y.all_sets_to_tensor()
        self.label.all_sets_to_tensor()
   