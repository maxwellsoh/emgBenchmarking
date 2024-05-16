import subprocess
import argparse
import utils_OzdemirEMG as utils # needed for parser
import yaml

def list_of_ints(arg):
    """
    Define a custom argument type for a list of integers.
    """
    return list(map(int, arg.split(',')))

# def list_of_strings(arg):
#     """
#     Define a custom argument type for a list of strings.
#     """
#     return list(map(str, arg.split(',')))

def initialize_parser():
    """
    Sets up the arg parser for CNN_EMG
    """
    # Create the parser
    parser = argparse.ArgumentParser(description="Specifiy the configurations for a given run.")

    group = parser.add_mutually_exclusive_group()

    # (1) preset runs (need a preset flag)
    group.add_argument("--preset", type=str, required=False, help="Choose a preset run that runs multiple times.")

    # (2) custom run (yaml config file, fields, max)
    group.add_argument("--custom", action="store_true")
    parser.add_argument("--config", type=str, help="Specify the config .yaml file.")
    parser.add_argument("--fields", help="List of fields whose value should iterate each run.", nargs="+", default=[])
    parser.add_argument("--max", help="Max range of iterations.", default=2, type=int)

    # (3) argparse: take in manually given arguments
    # this one has to be manually called using ./CNN_EMG --argument

    # Parse the arguments
    args = parser.parse_args()
    return args

def update_yaml(yaml_file, update_fields, new_value):
    """
    Updates the fields listed in update_fields to be new_value in a given yaml_file. Currently updates all the fields to the same value.

    Args:
        yaml_file (str): yaml file to update
        update_field (str): list of fields to update
        new_value: value to update to
    """
    with open(yaml_file, "r") as file: 
        lines = file.readlines()

    new_yaml = []

    for line in lines:
        for field in update_fields: 
            # print("field:", field)       
            if line.startswith(field):
                line = f"{field}: {new_value}\n"
        new_yaml.append(line)

    with open(yaml_file, "w") as file:
        file.writelines(new_yaml)

def run_config(yaml_file, fields, max):
    """
    Runs a given configuration (specified by yaml_file) max times. In each iteration, the value of fields is updated to the value of the current iteration. (Replaces for i in {1..13}; do --field${i} behavior)

    Args:
        yaml_file: path to relevant config file
        fields (str list): list of yaml fields to update at each iteration
        max (int): max number of iterations
    """
    for i in range(1, max):
        update_yaml(yaml_file, fields, i)
        subprocess.run(["python", "./CNN_EMG.py", "--config", yaml_file])
        
def main():
    """
    Takes in arguments for specific, preset runs. 
    To add a new preset run, (1) add an argument (2) run_config 
    """

    args = initialize_parser()

    # (1) preset run 
    if args.preset:
        if args.preset == "one":
            run_config(yaml_file="./config/one_config.yaml", fields=["leftout_subject"], max=13)

        if args.preset == "two":
            #TODO: modify to account for this one
            print("need to work on this one....")
        
        if args.preset == "three":
            run_config(yaml_file="./config/three_config.yaml", fields=["leftout_subject"], max=13)

        if args.preset == "four":
            run_config(yaml_file="./config/four_config.yaml", fields=["leftout_subject"], max=13)
            
    # (2) custom run (yaml config file, fields, max)
    if args.custom:
        if args.config and args.fields and args.max:
            print("CUSTOM: args.fields", args.fields)
            run_config(args.config, args.fields, args.max)
        else:
            print("--config, --fields, and --max must all be specified.")
            
    # (3) arg parse (pass in specific arguments to read)
    # need to call CNN_EMG for that 
    

    # TODO: name the different runs
    # TODO: check that the for i in range is going the full amount

if __name__ == "__main__":
    main()


