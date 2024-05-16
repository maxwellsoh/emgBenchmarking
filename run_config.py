import subprocess
import argparse


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

    # TODO: name the different runs
    # TODO: generalize to create a new config set up.

    parser = argparse.ArgumentParser(description="Specify which preset config you'd like to run.")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--one", action="store_true")
    parser.add_argument("--two", action="store_true")
    parser.add_argument("--three", action="store_true")
    parser.add_argument("--four", action="store_true")

    args = parser.parse_args()

    # TODO or just take in the config, max, and fields manually 
    # TODO: check that the for i in range is going the full amount

    if not any(vars(args).values()):
        print("You must specify a run.")

    if args.test:
        run_config(yaml_file="./config/test_config.yaml", fields=[], max=2)
    
    if args.one:
        run_config(yaml_file="./config/one_config.yaml", fields=["leftout_subject"], max=13)

    if args.two:
        #TODO: modify to account for this one
        print("need to work on this one....")

    if args.three:
        run_config(yaml_file="./config/three_config.yaml", fields=["leftout_subject"], max=13)

    if args.four:
        run_config(yaml_file="./config/four_config.yaml", fields=["leftout_subject"], max=13)


if __name__ == "__main__":
   
    main()


