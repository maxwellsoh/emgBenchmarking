from .Data_Split_Strategy import Data_Split_Strategy
from .cross_validation_utilities import train_test_split as tts # custom train test split to split stratified without shuffling

class Single_Subject(Data_Split_Strategy):

    def __init__(self, X_data, Y_data, label_data, env):
        super().__init__(X_data, Y_data, label_data, env)
        

    def train_from_one_subject(self):
        self.X.train_from_one_subject()
        self.Y.train_from_one_subject()
        self.label.train_from_one_subject()

    def split(self):
        
        assert not self.args.pretrain_and_finetune, "Cannot pretrain and finetune with a single subject."

        self.train_from_one_subject()

        self.X.train, self.Y.validation, \
        self.Y.train, self.Y.validation, \
        self.label.train, self.label.validation \
        = tts.train_test_split(
            self.X.train, 
            self.Y.train, 
            test_size=1-self.args.proportion_transfer_learning_from_leftout_subject, 
            stratify=self.label.train, 
            shuffle=(not self.args.train_test_split_for_time_series), 
            force_regression=self.args.force_regression
            )

        self.train_from_self_tensor()
        self.validation_from_self_tensor()

        super().test_from_validation()
        super().print_set_shapes()
        super().all_sets_to_tensor()
  