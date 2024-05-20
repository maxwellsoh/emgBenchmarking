"""
File: get_config.py
Description: Parses either command line arguments (argparse) or a given yaml file (yaml) to configure a given run.  
"""
import argparse
import yaml
import utils_OzdemirEMG as utils # needed for parser

def list_of_ints(arg):
    """
    Define a custom argument type for a list of integers.
    """
    return list(map(int, arg.split(',')))

def override_with_config(args):
    """
    Overrides the default and manually passed in arguments with config file value passed in --config
    """

    print("passed in args.config:", args.config)
    with open(args.config, "r") as file: 
        data = yaml.safe_load(file)

    delattr(args, "config")
    arg_dict = args.__dict__
    
    for key in data:
        arg_dict[key] = data[key]

def initialize_parser():
    """
    Sets up the arg parser for CNN_EMG
    """
    # Create the parser
    parser = argparse.ArgumentParser(description="Include arguments for running different trials")

    # Add argument for taking in config file
    parser.add_argument("--config", type=str, required=False)
    # Add argument for dataset
    parser.add_argument('--dataset', help='dataset to test. Set to OzdemirEMG by default', default="OzdemirEMG")
    # Add argument for doing leave-one-subject-out
    parser.add_argument('--leave_one_subject_out', type=utils.str2bool, help='whether or not to do leave one subject out. Set to False by default.', default=False)
    # Add argument for leftout subject
    parser.add_argument('--leftout_subject', type=int, help='number of subject that is left out for cross validation, starting from subject 1', default=0)
    # Add parser for seed
    parser.add_argument('--seed', type=int, help='seed for reproducibility. Set to 0 by default.', default=0)
    # Add number of epochs to train for
    parser.add_argument('--epochs', type=int, help='number of epochs to train for. Set to 25 by default.', default=25)
    # Add whether or not to use k folds (leftout_subject must be 0)
    parser.add_argument('--turn_on_kfold', type=utils.str2bool, help='whether or not to use k folds cross validation. Set to False by default.', default=False)
    # Add argument for stratified k folds cross validation
    parser.add_argument('--kfold', type=int, help='number of folds for stratified k-folds cross-validation. Set to 5 by default.', default=5)
    # Add argument for checking the index of the fold
    parser.add_argument('--fold_index', type=int, help='index of the fold to use for cross validation (should be from 1 to --kfold). Set to 1 by default.', default=1)
    # Add argument for whether or not to use cyclical learning rate
    parser.add_argument('--turn_on_cyclical_lr', type=utils.str2bool, help='whether or not to use cyclical learning rate. Set to False by default.', default=False)
    # Add argument for whether or not to use cosine annealing with warm restarts
    parser.add_argument('--turn_on_cosine_annealing', type=utils.str2bool, help='whether or not to use cosine annealing with warm restarts. Set to False by default.', default=False)
    # Add argument for whether or not to use RMS
    parser.add_argument('--turn_on_rms', type=utils.str2bool, help='whether or not to use RMS. Set to False by default.', default=False)
    # Add argument for RMS input window size (resulting feature dimension to classifier)
    parser.add_argument('--rms_input_windowsize', type=int, help='RMS input window size. Set to 1000 by default.', default=1000)
    # Add argument for whether or not to concatenate magnitude image
    parser.add_argument('--turn_on_magnitude', type=utils.str2bool, help='whether or not to concatenate magnitude image. Set to False by default.', default=False)
    # Add argument for model to use
    parser.add_argument('--model', type=str, help='model to use (e.g. \'convnext_tiny_custom\', \'convnext_tiny\', \'davit_tiny.msft_in1k\', \'efficientnet_b3.ns_jft_in1k\', \'vit_tiny_patch16_224\', \'efficientnet_b0\'). Set to resnet50 by default.', default='resnet50')
    # Add argument for exercises to include
    parser.add_argument('--exercises', type=list_of_ints, help='List the exercises of the 3 to load. The most popular for benchmarking seem to be 2 and 3. Can format as \'--exercises 1,2,3\'', default=[1, 2, 3])
    # Add argument for project suffix
    parser.add_argument('--project_name_suffix', type=str, help='suffix for project name. Set to empty string by default.', default='')
    # Add argument for full or partial dataset for Ozdemir EMG dataset
    parser.add_argument('--full_dataset_ozdemir', type=utils.str2bool, help='whether or not to use the full dataset for Ozdemir EMG Dataset. Set to False by default.', default=False)
    # Add argument for partial dataset for Ninapro DB2 and DB5
    parser.add_argument('--partial_dataset_ninapro', type=utils.str2bool, help='whether or not to use the partial dataset for Ninapro DB2 and DB5. Set to False by default.', default=False)
    # Add argument for using spectrogram transform
    parser.add_argument('--turn_on_spectrogram', type=utils.str2bool, help='whether or not to use spectrogram transform. Set to False by default.', default=False)
    # Add argument for using cwt
    parser.add_argument('--turn_on_cwt', type=utils.str2bool, help='whether or not to use cwt. Set to False by default.', default=False)
    # Add argument for using Hilbert Huang Transform
    parser.add_argument('--turn_on_hht', type=utils.str2bool, help='whether or not to use HHT. Set to False by default.', default=False)
    # Add argument for saving images
    parser.add_argument('--save_images', type=utils.str2bool, help='whether or not to save images. Set to False by default.', default=False)
    # Add argument to turn off scaler normalization
    parser.add_argument('--turn_off_scaler_normalization', type=utils.str2bool, help='whether or not to turn off scaler normalization. Set to False by default.', default=False)
    # Add argument to change learning rate
    parser.add_argument('--learning_rate', type=float, help='learning rate. Set to 1e-4 by default.', default=1e-4)
    # Add argument to specify which gpu to use (if any gpu exists)
    parser.add_argument('--gpu', type=int, help='which gpu to use. Set to 0 by default.', default=0)
    # Add argument for loading just a few images from dataset for debugging
    parser.add_argument('--load_few_images', type=utils.str2bool, help='whether or not to load just a few images from dataset for debugging. Set to False by default.', default=False)
    # Add argument for reducing training data size while remaining stratified in terms of gestures and amount of data from each subject
    parser.add_argument('--reduce_training_data_size', type=utils.str2bool, help='whether or not to reduce training data size while remaining stratified in terms of gestures and amount of data from each subject. Set to False by default.', default=False)
    # Add argument for size of reduced training data
    parser.add_argument('--reduced_training_data_size', type=int, help='size of reduced training data. Set to 56000 by default.', default=56000)
    # Add argument to leve n subjects out randomly
    parser.add_argument('--leave_n_subjects_out_randomly', type=int, help='number of subjects to leave out randomly. Set to 0 by default.', default=0)
    # use target domain for normalization
    parser.add_argument('--target_normalize', type=utils.str2bool, help='use a leftout window for normalization. Set to False by default.', default=False)
    # Test with transfer learning by using some data from the validation dataset
    parser.add_argument('--transfer_learning', type=utils.str2bool, help='use some data from the validation dataset for transfer learning. Set to False by default.', default=False)
    # Add argument for cross validation for time series
    parser.add_argument('--cross_validation_for_time_series', type=utils.str2bool, help='whether or not to use cross validation for time series. Set to False by default.', default=False)
    # Add argument for proportion of left-out-subject data to use for transfer learning
    parser.add_argument('--proportion_transfer_learning_from_leftout_subject', type=float, help='proportion of left-out-subject data to use for transfer learning. Set to 0.25 by default.', default=0.25)
    # Add argument for amount for reducing number of data to generate for transfer learning
    parser.add_argument('--reduce_data_for_transfer_learning', type=int, help='amount for reducing number of data to generate for transfer learning. Set to 1 by default.', default=1)
    # Add argument for whether to do leave-one-session-out
    parser.add_argument('--leave_one_session_out', type=utils.str2bool, help='whether or not to leave one session out. Set to False by default.', default=False)
    # Add argument for whether to do held_out test
    parser.add_argument('--held_out_test', type=utils.str2bool, help='whether or not to do held out test. Set to False by default.', default=False)
    # Add argument for whether to use only the subject left out for training in leave out session test
    parser.add_argument('--one_subject_for_training_set_for_session_test', type=utils.str2bool, help='whether or not to use only the subject left out for training in leave out session test. Set to False by default.', default=False)
    # Add argument for pretraining on all data from other subjects, and fine-tuning on some data from left out subject
    parser.add_argument('--pretrain_and_finetune', type=utils.str2bool, help='whether or not to pretrain on all data from other subjects, and fine-tune on some data from left out subject. Set to False by default.', default=False)
    # Add argument for finetuning epochs
    parser.add_argument('--finetuning_epochs', type=int, help='number of epochs to fine-tune for. Set to 25 by default.', default=25)
    # Add argument for whether or not to turn on unlabeled domain adaptation
    parser.add_argument('--turn_on_unlabeled_domain_adaptation', type=utils.str2bool, help='whether or not to turn on unlabeled domain adaptation methods. Set to False by default.', default=False)
    # Add argument to specify algorithm to use for unlabeled domain adaptation
    parser.add_argument('--unlabeled_algorithm', type=str, help='algorithm to use for unlabeled domain adaptation. Set to "fixmatch" by default.', default="fixmatch")
    # Add argument to specify proportion from left-out-subject to keep as unlabeled data
    parser.add_argument('--proportion_unlabeled_data_from_leftout_subject', type=float, help='proportion of data from left-out-subject to keep as unlabeled data. Set to 0.75 by default.', default=0.75)
    # Add argument to specify batch size
    parser.add_argument('--batch_size', type=int, help='batch size. Set to 64 by default.', default=64)
    # Add argument for whether to use unlabeled data for subjects used for training as well
    parser.add_argument('--proportion_unlabeled_data_from_training_subjects', type=float, help='proportion of data from training subjects to use as unlabeled data. Set to 0.0 by default.', default=0.0)
    # Add argument for cutting down amount of total data for training subjects
    parser.add_argument('--proportion_data_from_training_subjects', type=float, help='proportion of data from training subjects to use. Set to 1.0 by default.', default=1.0)

    # Parse the arguments
    args = parser.parse_args()
    return args

def get():
    """
    Parses either command line arguments (argparse) or a given yaml file (yaml) to configure a given run.  

    Returns:
        args: Final list of args and their values. 
    """

    args = initialize_parser()
    if args.config:
        override_with_config(args)
    return args

    

    


