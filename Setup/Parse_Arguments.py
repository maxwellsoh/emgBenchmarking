
import argparse
from .Setup import Setup

class Parse_Arguments(Setup):
    
    def __init__(self):
        super().__init__()

    def set_args(self):
        super().create_argparse()