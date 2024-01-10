import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from scipy.signal import butter,filtfilt,iirnotch
#from PyEMD import EMD
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import multiprocessing
from tqdm import tqdm
import argparse
import random 
import utils_NinaproDB5 as ut_NDB5
from sklearn.model_selection import StratifiedKFold
import os
import datetime
import matplotlib as mpl
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import timm
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


## Argument parser with optional argumenets

# Create the parser
parser = argparse.ArgumentParser(description="Include arguments for running different trials")

# Add argument for leftout subject
parser.add_argument('--leftout_subject', type=int, help='number of subject that is left out for cross validation. Set to 0 to run standard random held-out test. Set to 0 by default.', default=0)
# Add parser for seed
parser.add_argument('--seed', type=int, help='seed for reproducibility. Set to 0 by default.', default=0)
# Add number of epochs to train for
parser.add_argument('--epochs', type=int, help='number of epochs to train for. Set to 25 by default.', default=25)
# Add whether or not to use k folds (leftout_subject must be 0)
parser.add_argument('--turn_on_kfold', type=ut_NDB5.str2bool, help='whether or not to use k folds cross validation. Set to False by default.', default=False)
# Add argument for stratified k folds cross validation
parser.add_argument('--kfold', type=int, help='number of folds for stratified k-folds cross-validation. Set to 5 by default.', default=5)
# Add argument for checking the index of the fold
parser.add_argument('--fold_index', type=int, help='index of the fold to use for cross validation (should be from 1 to --kfold). Set to 1 by default.', default=1)
# Add argument for whether or not to use cyclical learning rate
parser.add_argument('--turn_on_cyclical_lr', type=ut_NDB5.str2bool, help='whether or not to use cyclical learning rate. Set to False by default.', default=False)
# Add argument for whether or not to use cosine annealing with warm restartfs
parser.add_argument('--turn_on_cosine_annealing', type=ut_NDB5.str2bool, help='whether or not to use cosine annealing with warm restarts. Set to False by default.', default=False)
# Add argument for whether or not to use RMS
parser.add_argument('--turn_on_rms', type=ut_NDB5.str2bool, help='whether or not to use RMS. Set to False by default.', default=False)
# Add argument for number of RMS windows
parser.add_argument('--num_rms_windows', type=int, help='number of RMS windows to use. Set to 10 by default.', default=10)
# Add argument for whether or not to concatenate magnitude image
parser.add_argument('--turn_on_magnitude', type=ut_NDB5.str2bool, help='whether or not to concatenate magnitude image. Set to False by default.', default=False)
# Add argument for model to use
parser.add_argument('--model', type=str, help='model to use (e.g. \'convnext_tiny_custom\', \'convnext_tiny\', \'davit_tiny.msft_in1k\', \'efficientnet_b3.ns_jft_in1k\', \'vit_tiny_path16_224\', \'efficientnet_b0\'). Set to resnet50 by default.', default='resnet50')

# Parse the arguments
args = parser.parse_args()

# Use the arguments
print(f"The value of --leftout_subject is {args.leftout_subject}")
print(f"The value of --seed is {args.seed}")
print(f"The value of --epochs is {args.epochs}")
print(f"The model to use is {args.model}")
if args.turn_on_kfold:
    if args.leftout_subject == 0:
        print(f"The value of --turn_on_kfold is {args.turn_on_kfold}")
    else: 
        print("Cannot turn on kfold if leftout_subject is not 0")
        exit()
    print(f"The value of --kfold is {args.kfold}")
    print(f"The value of --fold_index is {args.fold_index}")
    
if args.turn_on_cyclical_lr:
    print(f"The value of --turn_on_cyclical_lr is {args.turn_on_cyclical_lr}")
if args.turn_on_cosine_annealing:
    print(f"The value of --turn_on_cosine_annealing is {args.turn_on_cosine_annealing}")
if args.turn_on_cyclical_lr and args.turn_on_cosine_annealing:
    print("Cannot turn on both cyclical learning rate and cosine annealing")
    exit()
if args.turn_on_rms:
    print(f"The value of --turn_on_rms is {args.turn_on_rms}")
    print(f"The value of --num_rms_windows is {args.num_rms_windows}")
if args.turn_on_magnitude:
    print(f"The value of --turn_on_magnitude is {args.turn_on_magnitude}")
    
# Add date and time to filename
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

print("------------------------------------------------------------------------------------------------------------------------")
print("Starting run at", formatted_datetime)
print("------------------------------------------------------------------------------------------------------------------------")

# %%
# 0 for no LOSO; participants here are 1-13
leaveOut = int(args.leftout_subject)

# Set seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with  multiprocessing.Pool() as pool:
    emg_async = pool.map_async(ut_NDB5.getEMG, [(i+1) for i in range(10)])
    emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
    
    labels_async = pool.map_async(ut_NDB5.getLabels, [(i+1) for i in range(10)])
    labels = labels_async.get()

length = len(emg[0][0])
width = len(emg[0][0][0])
print("Number of Electrode Channels: ", length)
print("Number of Timesteps per Trial:", width)

# These can be tuned to change the normalization
# This is the coefficient for the standard deviation
# used for the magnitude images. In practice, these
# should be fairly small so that images have more
# contrast
if args.turn_on_rms:
    # This tends to be small because in pracitice
    # the RMS is usually much smaller than the raw EMG
    # NOTE: Should check why this is the case
    sigma_coefficient = 0.1
else:
    # This tends to be larger because the raw EMG
    # is usually much larger than the RMS
    sigma_coefficient = 0.5

if (leaveOut == 0):
    if args.turn_on_kfold:
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        
        emg_in = np.concatenate([np.array(i.reshape(-1, length*width)) for i in emg], axis=0, dtype=np.float32)
        labels_in = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)
        
        labels_for_folds = np.argmax(labels_in, axis=1)
        
        fold_count = 1
        for train_index, test_index in skf.split(emg_in, labels_for_folds):
            if fold_count == args.fold_index:
                train_indices = train_index
                validation_indices = test_index
                break
            fold_count += 1

        # Normalize by electrode
        emg_in_by_electrode = emg_in[train_indices].reshape(-1, length, width)
        # s = preprocessing.StandardScaler().fit(emg_in[train_indices])
        global_min = emg_in[train_indices].mean() - sigma_coefficient*emg_in[train_indices].std()
        global_max = emg_in[train_indices].mean() + sigma_coefficient*emg_in[train_indices].std()

        # Assuming emg is your initial data of shape (SAMPLES, 16, 50)
        # Reshape data to (SAMPLES*50, 16)
        emg_reshaped = emg_in_by_electrode.reshape(-1, ut_NDB5.numElectrodes)

        # Initialize and fit the scaler on the reshaped data
        # This will compute the mean and std dev for each electrode across all samples and features
        scaler = preprocessing.StandardScaler()
        scaler.fit(emg_reshaped)
        
        # Repeat means and std_devs for each time point using np.repeat
        scaler.mean_ = np.repeat(scaler.mean_, width)
        scaler.scale_ = np.repeat(scaler.scale_, width)
        scaler.var_ = np.repeat(scaler.var_, width)
        scaler.n_features_in_ = width*ut_NDB5.numElectrodes

        del emg_in
        del labels_in

        del emg_in_by_electrode
        del emg_reshaped

    else: 
        # Reshape and concatenate EMG data
        # Flatten each subject's data from (TRIAL, CHANNEL, TIME) to (TRIAL, CHANNEL*TIME)
        # Then concatenate along the subject dimension (axis=0)
        emg_in = np.concatenate([np.array(i.reshape(-1, length*width)) for i in emg], axis=0, dtype=np.float32)
        labels_in = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)
        indices = np.arange(emg_in.shape[0])
        train_indices, validation_indices = model_selection.train_test_split(indices, test_size=0.2, stratify=labels_in)
        train_emg_in = emg_in[train_indices]  # Select only the train indices
        # s = preprocessing.StandardScaler().fit(train_emg_in)

        # Normalize by electrode
        emg_in_by_electrode = train_emg_in.reshape(-1, length, width)
        global_min = emg_in[train_indices].mean() - sigma_coefficient*emg_in[train_indices].std()
        global_max = emg_in[train_indices].mean() + sigma_coefficient*emg_in[train_indices].std()

        # Assuming emg is your initial data of shape (SAMPLES, 16, 50)
        # Reshape data to (SAMPLES*50, 16)
        emg_reshaped = emg_in_by_electrode.reshape(-1, ut_NDB5.numElectrodes)

        # Initialize and fit the scaler on the reshaped data
        # This will compute the mean and std dev for each electrode across all samples and features
        scaler = preprocessing.StandardScaler()
        scaler.fit(emg_reshaped)
        
        # Repeat means and std_devs for each time point using np.repeat
        scaler.mean_ = np.repeat(scaler.mean_, width)
        scaler.scale_ = np.repeat(scaler.scale_, width)
        scaler.var_ = np.repeat(scaler.var_, width)
        scaler.n_features_in_ = width*ut_NDB5.numElectrodes

        del emg_in
        del labels_in

        del train_emg_in
        del indices

        del emg_in_by_electrode
        del emg_reshaped

else: # Running LOSO
    emg_in = np.concatenate([np.array(i.view(len(i), length*width)) for i in emg[:(leaveOut-1)]] + [np.array(i.view(len(i), length*width)) for i in emg[leaveOut:]], axis=0, dtype=np.float32)
    # s = preprocessing.StandardScaler().fit(emg_in)
    global_min = emg_in.mean() - sigma_coefficient*emg_in.std()
    global_max = emg_in.mean() + sigma_coefficient*emg_in.std()

    # Normalize by electrode
    emg_in_by_electrode = emg_in.reshape(-1, length, width)

    # Assuming emg is your initial data of shape (SAMPLES, 16, 50)
    # Reshape data to (SAMPLES*50, 16)
    emg_reshaped = emg_in_by_electrode.reshape(-1, ut_NDB5.numElectrodes)

    # Initialize and fit the scaler on the reshaped data
    # This will compute the mean and std dev for each electrode across all samples and features
    scaler = preprocessing.StandardScaler()
    scaler.fit(emg_reshaped)
    
    # Repeat means and std_devs for each time point using np.repeat
    scaler.mean_ = np.repeat(scaler.mean_, width)
    scaler.scale_ = np.repeat(scaler.scale_, width)
    scaler.var_ = np.repeat(scaler.var_, width)
    scaler.n_features_in_ = width*ut_NDB5.numElectrodes

    del emg_in
    del emg_in_by_electrode
    del emg_reshaped


data = []

# add tqdm to show progress bar
for x in tqdm(range(len(emg)), desc="Number of Subjects "):
    data += [ut_NDB5.getImages(emg[x], scaler, length, width, turn_on_rms=args.turn_on_rms, rms_windows=args.num_rms_windows, turn_on_magnitude=args.turn_on_magnitude, global_min=global_min, global_max=global_max)]

print("------------------------------------------------------------------------------------------------------------------------")
print("NOTE: The width 224 is natively used in Resnet50, height is currently integer  multiples of number of electrode channels")
print("------------------------------------------------------------------------------------------------------------------------")
if leaveOut == 0:
    combined_labels = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)
    combined_images = np.concatenate([np.array(i) for i in data], axis=0, dtype=np.float16)
    X_train = combined_images[train_indices]
    Y_train = combined_labels[train_indices]
    X_validation = combined_images[validation_indices]
    Y_validation = combined_labels[validation_indices]
    X_validation, X_test, Y_validation, Y_test = model_selection.train_test_split(X_validation, Y_validation, test_size=0.5, stratify=Y_validation)
    del combined_images
    del combined_labels
    del data
    del emg

    X_train = torch.from_numpy(X_train).to(torch.float32)
    Y_train = torch.from_numpy(Y_train).to(torch.float32)
    X_validation = torch.from_numpy(X_validation).to(torch.float32)
    Y_validation = torch.from_numpy(Y_validation).to(torch.float32)
    X_test = torch.from_numpy(X_test).to(torch.float32)
    Y_test = torch.from_numpy(Y_test).to(torch.float32)
    print("Size of X_train:     ", X_train.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_train:     ", Y_train.size()) # (SAMPLE, GESTURE)
    print("Size of X_validation:", X_validation.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_validation:", Y_validation.size()) # (SAMPLE, GESTURE)
    print("Size of X_test:      ", X_test.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_test:      ", Y_test.size()) # (SAMPLE, GESTURE)
else:
    X_validation = np.array(data.pop(leaveOut-1))
    Y_validation = np.array(labels.pop(leaveOut-1))
    X_train = np.concatenate([np.array(i) for i in data], axis=0, dtype=np.float32)
    Y_train = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float32)
    X_train = torch.from_numpy(X_train).to(torch.float32)
    Y_train = torch.from_numpy(Y_train).to(torch.float32)
    X_validation = torch.from_numpy(X_validation).to(torch.float32)
    Y_validation = torch.from_numpy(Y_validation).to(torch.float32)
    print("Size of X_train:", X_train.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_train:", Y_train.size()) # (SAMPLE, GESTURE)
    print("Size of X_validation:", X_validation.size()) # (SAMPLE, CHANNEL_RGB, HEIGHT, WIDTH)
    print("Size of Y_validation:", Y_validation.size()) # (SAMPLE, GESTURE)

model_name = args.model
if args.model == 'resnet50':
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-4])
    # #model = nn.Sequential(*list(model.children())[:-4])
    num_features = model[-1][-1].conv3.out_channels
    # #num_features = model.fc.in_features
    dropout = 0.5
    model.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
    model.add_module('flatten', nn.Flatten())
    '''
    model.add_module('fc1', nn.Linear(num_features, 1024))
    model.add_module('relu', nn.ReLU())
    model.add_module('dropout1', nn.Dropout(dropout))
    model.add_module('fc2', nn.Linear(1024, 1024))
    model.add_module('relu2', nn.ReLU())
    model.add_module('dropout2', nn.Dropout(dropout))
    model.add_module('fc3', nn.Linear(1024, ut_NDB5.numGestures))
    '''
    model.add_module('fc1', nn.Linear(num_features, 512))
    model.add_module('relu', nn.ReLU())
    model.add_module('dropout1', nn.Dropout(dropout))
    model.add_module('fc3', nn.Linear(512, ut_NDB5.numGestures))
    model.add_module('softmax', nn.Softmax(dim=1))
elif args.model == 'convnext_tiny_custom':
    # %% Referencing: https://medium.com/exemplifyml-ai/image-classification-with-resnet-convnext-using-pytorch-f051d0d7e098
    class LayerNorm2d(nn.LayerNorm):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 3, 1)
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x

    n_inputs = 768
    hidden_size = 128 # default is 2048
    n_outputs = ut_NDB5.numGestures

    # model = timm.create_model(model_name, pretrained=True, num_classes=10)
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    #model = nn.Sequential(*list(model.children())[:-4])
    #model = nn.Sequential(*list(model.children())[:-3])
    #num_features = model[-1][-1].conv3.out_channels
    #num_features = model.fc.in_features
    dropout = 0.1 # was 0.5

    sequential_layers = nn.Sequential(
        LayerNorm2d((n_inputs,), eps=1e-06, elementwise_affine=True),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(n_inputs, hidden_size, bias=True),
        nn.BatchNorm1d(hidden_size),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, n_outputs),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = sequential_layers

else: 
    # model_name = 'efficientnet_b0'  # or 'efficientnet_b1', ..., 'efficientnet_b7'
    # model_name = 'tf_efficientnet_b3.ns_jft_in1k'
    model = timm.create_model(model_name, pretrained=True, num_classes=ut_NDB5.numGestures)
    # # Load the Vision Transformer model
    # model_name = 'vit_base_patch16_224'  # This is just one example, many variations exist
    # model = timm.create_model(model_name, pretrained=True, num_classes=ut_NDB5.numGestures)

num = 0
for name, param in model.named_parameters():
    num += 1
    if (num > 0):
    #if (num > 72): # for -3
    #if (num > 33): # for -4
        param.requires_grad = True
    else:
        param.requires_grad = False

batch_size = 64
train_loader = DataLoader(list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=ut_NDB5.seed_worker, pin_memory=True)
val_loader = DataLoader(list(zip(X_validation, Y_validation)), batch_size=batch_size, num_workers=4, worker_init_fn=ut_NDB5.seed_worker, pin_memory=True)
if (leaveOut == 0):
    test_loader = DataLoader(list(zip(X_test, Y_test)), batch_size=batch_size, num_workers=4, worker_init_fn=ut_NDB5.seed_worker, pin_memory=True)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
learn = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learn)

num_epochs = args.epochs
if args.turn_on_cosine_annealing:
    number_cycles = 5
    annealing_multiplier = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=ut_NDB5.periodLengthForAnnealing(num_epochs, annealing_multiplier, number_cycles),
                                                                        T_mult=annealing_multiplier, eta_min=1e-5, last_epoch=-1)
elif args.turn_on_cyclical_lr:
    # Define the cyclical learning rate scheduler
    step_size = len(train_loader) * 6  # Number of iterations in half a cycle
    base_lr = 1e-4  # Minimum learning rate
    max_lr = 1e-3  # Maximum learning rate
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=step_size, mode='triangular2', cycle_momentum=False)

# Training loop
import gc
import datetime
gc.collect()
torch.cuda.empty_cache()

wandb_runname = 'CNN_seed-'+str(args.seed)
if args.turn_on_kfold:
    wandb_runname += '_k-fold-'+str(args.kfold)+'_fold-index-'+str(args.fold_index)
if args.turn_on_cyclical_lr:
    wandb_runname += '_cyclical-lr'
if args.turn_on_cosine_annealing: 
    wandb_runname += '_cosine-annealing'
if args.turn_on_rms:
    wandb_runname += '_rms-windows-'+str(args.num_rms_windows)
if args.turn_on_magnitude:  
    wandb_runname += '_magnitude'
if args.leftout_subject != 0:
    wandb_runname += '_LOSO-'+str(args.leftout_subject)
wandb_runname += '_' + model_name

project_name = 'emg_benchmarking_ninapro-db5'
if (leaveOut == 0):
    if args.turn_on_kfold:
        project_name += '_k-fold-'+str(args.kfold)
    else:
        project_name += '_heldout'
else:
    project_name += '_LOSO'

run = wandb.init(name=wandb_runname, project=project_name, entity='jehanyang')
wandb.config.lr = learn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)

wandb.watch(model)

testrun_foldername = f'test/{project_name}/{wandb_runname}/{formatted_datetime}/'
# Make folder if it doesn't exist
if not os.path.exists(testrun_foldername):
    os.makedirs(testrun_foldername)
model_filename = f'{testrun_foldername}model_{formatted_datetime}.pth'

if leaveOut == 0:
    # Plot and log images
    ut_NDB5.plot_average_images(X_test, np.argmax(Y_test.cpu().detach().numpy(), axis=1), ut_NDB5.gesture_labels, testrun_foldername, args, formatted_datetime, 'test')
    ut_NDB5.plot_first_fifteen_images(X_test, np.argmax(Y_test.cpu().detach().numpy(), axis=1), ut_NDB5.gesture_labels, testrun_foldername, args, formatted_datetime, 'test')

ut_NDB5.plot_average_images(X_validation, np.argmax(Y_validation.cpu().detach().numpy(), axis=1), ut_NDB5.gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')
ut_NDB5.plot_first_fifteen_images(X_validation, np.argmax(Y_validation.cpu().detach().numpy(), axis=1), ut_NDB5.gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')

ut_NDB5.plot_average_images(X_train, np.argmax(Y_train.cpu().detach().numpy(), axis=1), ut_NDB5.gesture_labels, testrun_foldername, args, formatted_datetime, 'train')
ut_NDB5.plot_first_fifteen_images(X_train, np.argmax(Y_train.cpu().detach().numpy(), axis=1), ut_NDB5.gesture_labels, testrun_foldername, args, formatted_datetime, 'train')

for epoch in tqdm(range(num_epochs), desc="Epoch"):
    model.train()
    train_acc = 0.0
    train_loss = 0.0
    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, Y_batch)
        train_loss += loss.item()

        train_acc += np.mean(np.argmax(output.cpu().detach().numpy(), 
                                       axis=1) == np.argmax(Y_batch.cpu().detach().numpy(), axis=1))

        loss.backward()
        optimizer.step()
        if args.turn_on_cyclical_lr:
            scheduler.step()
            
    if args.turn_on_cosine_annealing:
        scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            output = model(X_batch)
            val_loss += criterion(output, Y_batch).item()

            val_acc += np.mean(np.argmax(output.cpu().detach().numpy(), axis=1) == np.argmax(Y_batch.cpu().detach().numpy(), axis=1))

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")
    
    wandb.log({
        "Epoch": epoch,
        "Train Loss": train_loss,
        "Train Acc": train_acc,
        "Valid Loss": val_loss,
        "Valid Acc": val_acc, 
        "Learning Rate": optimizer.param_groups[0]['lr']})

torch.save(model.state_dict(), model_filename)
wandb.save(f'model/modelParameters_{formatted_datetime}.pth')

# Testing
if (leaveOut == 0):
    pred = []
    true = []

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            output = model(X_batch)
            test_loss += criterion(output, Y_batch).item()

            test_acc += np.mean(np.argmax(output.cpu().detach().numpy(), axis=1) == np.argmax(Y_batch.cpu().detach().numpy(), axis=1))

            output = np.argmax(output.cpu().detach().numpy(), axis=1)
            pred.extend(output)
            labels = np.argmax(Y_batch.cpu().detach().numpy(), axis=1)
            true.extend(labels)

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    wandb.log({
        "Test Loss": test_loss,
        "Test Acc": test_acc}) 
    
    
    # %% Confusion Matrix
    # Plot and log confusion matrix in wandb
    ut_NDB5.plot_confusion_matrix(true, pred, ut_NDB5.gesture_labels, testrun_foldername, args, formatted_datetime, 'test')

# Load validation in smaller batches for memory purposes
torch.cuda.empty_cache()  # Clear cache if needed

model.eval()
with torch.no_grad():
    validation_predictions = []
    for i, batch in tqdm(enumerate(torch.split(X_validation, split_size_or_sections=int(X_validation.shape[0]/10))), desc="Validation Batch Loading"):  # Or some other number that fits in memory
        batch = batch.to(device)
        outputs = model(batch)
        preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        validation_predictions.extend(preds)

ut_NDB5.plot_confusion_matrix(np.argmax(Y_validation.cpu().detach().numpy(), axis=1), np.array(validation_predictions), ut_NDB5.gesture_labels, testrun_foldername, args, formatted_datetime, 'validation')   

# Load training in smaller batches for memory purposes
torch.cuda.empty_cache()  # Clear cache if needed

model.eval()
with torch.no_grad():
    train_predictions = []
    for i, batch in tqdm(enumerate(torch.split(X_train, split_size_or_sections=int(X_train.shape[0]/40))), desc="Training Batch Loading"):  # Or some other number that fits in memory
        batch = batch.to(device)
        outputs = model(batch)
        preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        train_predictions.extend(preds)

ut_NDB5.plot_confusion_matrix(np.argmax(Y_train.cpu().detach().numpy(), axis=1), np.array(train_predictions), ut_NDB5.gesture_labels, testrun_foldername, args, formatted_datetime, 'train')
    
run.finish()
