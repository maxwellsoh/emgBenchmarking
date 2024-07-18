#!/usr/bin/env python
import argparse
import yaml
import CNN_EMG
import copy
from Setup.Setup import Setup # used to get

def list_of_ints(arg):
        """Define a custom argument type for a list of integers"""
        return list(map(int, arg.split(',')))

def load_config(args, config_path, return_table_args=False): 
    """Load the config file and override the args values with the config values. 

    If a key is used in table_fields value is not updated in config but is stored in table_args. This is useful for paramaters that vary over different lines in the same table. 

    Args:
        args (argparser): Default argument values
        config_path (str): Path to config file to load
        return_extra (bool, optional): Whether or not to return the extra fields that config has. Useful for table configs. Defaults to False.

    Returns:
        argparser: argparse object with updated values. values that are assumed to be the default line to line
        table_args: argparse object with extra fields that config has (ex: start index, dataset)
    """

    # Dict of fields that get updated in each line for each table. 
    table_fields = {"1": ['starting_index', 'ending_index', 'current_dataset', 'number_windows'], "2": ['starting_index', 'ending_index', 'current_dataset', 'preprocessing'], "3": ['starting_index', 'ending_index', 'current_dataset', 'preprocessing', 'best_model'], "3_intersession": ['starting_index', 'ending_index', 'current_dataset', 'preprocessing', 'best_model']}

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if "learning_rate" in config: 
        assert type(config["learning_rate"]) == float, "ERROR: Learning rate in config file must be a float. Adding a decimal point to the number should fix this. (Ex: 5.0e-4 instead of 5e-4)"

    # Override args values with config values
    args_dict = vars(args)
    table_args = {}

    if 'best_model' in config:
        config['model'] = config['best_model']

    for key in config:
        # If a table variable, update only the table_args
        if return_table_args and (key in table_fields[args.table]):
            table_args[key] = config[key]
        # If just a general variable that is the same across all lines, update "global" args
        else:
            args_dict[key] = config[key]

    args = argparse.Namespace(**args_dict)

    if return_table_args:
        table_args = argparse.Namespace(**table_args)
        return args, table_args
    return args

def run_command(args):
    """
    Pass args to CNN_EMG.py's use_config function to run the program using config influenced arg values. 

    Args:
        args (argparser): Arguments (updated with config values) to pass.
    """

    delattr(args, "config")
    delattr(args, "table")
    CNN_EMG.use_config(args) # special call to main that passes config args

def replicate_table(args, table_args, table_name):
    """Replicate a given table from the paper.

    Each line in the interation is stored in a separate config file. The line config file is loaded and the parameters that change for each iteration are updated. CNN_EMG.py is then run with the updated args.

    Args:
        args (argparse): Default arguments with table{i}.yaml values overridden
        table_args (argparse): Fields for the table
        table_name (str): Table number to replicate
    """

    starting_index = table_args.starting_index
    ending_index = table_args.ending_index

    if table_name == "4":
        NUM_LINES = 1
    else:
        NUM_LINES = 4
    
    for subj in range(starting_index, ending_index+1):
        args_copy = copy.deepcopy(args)

        for line in range(1, NUM_LINES+1):
            # Load in the config for the given line
            line_args = load_config(args_copy, f'config/table{table_name}_line{line}.yaml') # defaults + config values + line values

            # Any value that is variable (ex $best_model) per line should be updated here to the value in table_args config
            line_args.dataset = table_args.current_dataset
            line_args.leftout_subject = subj
            if table_name == "1" and line == "2": 
                line_args.number_windows = table_args.number_windows
            if table_name == "3" or table_name == "3_intersession":
                line_args.model = table_args.best_model
            
            run_command(line_args)
    
def main():
    """
    Gets the default parse_args from CNN_EMG. If config exists, overrides its fields with the config values. If table exists, uses config to get the values of the parameters that change at each iteration. Otherwise, uses command line arguments to run CNN_EMG.

    Important to get the parse args in order to keep default values. 
    """

    # Doing this ensures the default arg values are kept consistent between a manual CNN_EMG run and a config run_CNN_EMG run. 
    setup = Setup()
    args = setup.create_argparse() # keeps the arg parsers consistent between if manually set in CNN_EMG and if set here with a config

    if args.table:
        args, table_args = load_config(args, f'config/table{args.table}.yaml', return_table_args=True)
        replicate_table(args, table_args, args.table)
    
    elif args.config: 
        args = load_config(args, args.config)
        run_command(args)

    else:
        run_command(args)

if __name__ == "__main__":
    main()