
from .Setup import Setup

class Parse_Config(Setup):
    def __init__(self, config_args):
        super().__init__()
        self.config_args = config_args

    def set_args(self):
        self.args = self.config_args