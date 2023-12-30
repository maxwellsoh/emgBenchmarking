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
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing
from tqdm import tqdm
import argparse
import random 
import utils_NinaproDB5 as ut_NDB5
from sklearn.model_selection import StratifiedKFold

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

# Parse the arguments
args = parser.parse_args()

# Use the arguments
print(f"The value of --leftout_subject is {args.leftout_subject}")
print(f"The value of --seed is {args.seed}")
print(f"The value of --epochs is {args.epochs}")
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

if (leaveOut == 0):
    if args.turn_on_kfold:
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        
        emg_in = np.concatenate([np.array(i.reshape(-1, length*width)) for i in emg], axis=0, dtype=np.float16)
        labels_in = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)
        
        labels_for_folds = np.argmax(labels_in, axis=1)
        
        fold_count = 1
        for train_index, test_index in skf.split(emg_in, labels_for_folds):
            if fold_count == args.fold_index:
                train_indices = train_index
                validation_indices = test_index
                break
            fold_count += 1
        s = preprocessing.StandardScaler().fit(emg_in[train_indices])
        del emg_in
        del labels_in
    else: 
        # Reshape and concatenate EMG data
        # Flatten each subject's data from (TRIAL, CHANNEL, TIME) to (TRIAL, CHANNEL*TIME)
        # Then concatenate along the subject dimension (axis=0)
        emg_in = np.concatenate([np.array(i.reshape(-1, length*width)) for i in emg], axis=0, dtype=np.float16)
        labels_in = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)
        indices = np.arange(emg_in.shape[0])
        train_indices, validation_indices = model_selection.train_test_split(indices, test_size=0.2, stratify=labels_in)
        train_emg_in = emg_in[train_indices]  # Select only the train indices
        s = preprocessing.StandardScaler().fit(train_emg_in)
        del emg_in
        del train_emg_in
        del labels_in
        del indices
else: # Running LOSO
    emg_in = np.concatenate([np.array(i.view(len(i), length*width)) for i in emg[:(leaveOut-1)]] + [np.array(i.view(len(i), length*width)) for i in emg[leaveOut:]], axis=0, dtype=np.float16)
    s = preprocessing.StandardScaler().fit(emg_in)
    del emg_in

data = []

# add tqdm to show progress bar
for x in tqdm(range(len(emg)), desc="Subject"):
    data += [ut_NDB5.getImages(emg[x], s, length, width)]

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

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model = nn.Sequential(*list(model.children())[:-4])
#model = nn.Sequential(*list(model.children())[:-4])
num_features = model[-1][-1].conv3.out_channels
#num_features = model.fc.in_features
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
train_loader = DataLoader(list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(list(zip(X_validation, Y_validation)), batch_size=batch_size)
if (leaveOut == 0):
    test_loader = DataLoader(list(zip(X_test, Y_test)), batch_size=batch_size)

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
    step_size = len(train_loader) * 4  # Number of iterations in half a cycle
    base_lr = 1e-4  # Minimum learning rate
    max_lr = 1e-3  # Maximum learning rate
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=step_size, mode='triangular2', cycle_momentum=False)

# Training loop
import gc
gc.collect()
torch.cuda.empty_cache()

wandb_runname = 'CNN_seed-'+str(args.seed)
if args.turn_on_kfold:
    wandb_runname += '_kfold-'+str(args.kfold)+'_foldindex-'+str(args.fold_index)
if args.turn_on_cyclical_lr:
    wandb_runname += '_cyclicallr'
if args.turn_on_cosine_annealing:
    wandb_runname += '_cosineannealing'

if (leaveOut == 0):
    run = wandb.init(name=wandb_runname, project='emg_benchmarking_ninapro-db5_heldout', entity='jehanyang')
else:
    run = wandb.init(name=wandb_runname, project='emg_benchmarking_ninapro-db5_LOSO-' + str(args.leftout_subject), entity='jehanyang')
wandb.config.lr = learn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)

wandb.watch(model)

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
    
    # Gesture Labels
    gesture_labels = ['Rest', 'Thumb Up', 'Index Middle Extension', 'Ring Little Flexion', 'Thumb Opposition', 'Finger Abduction', 'Fist', 'Pointing Index', 'Finger Adduction', 
                      'Middle Axis Supination', 'Middle Axis Pronation', 'Little Axis Supination', 'Little Axis Pronation', 'Wrist Flexion', 'Wrist Extension', 'Radial Deviation', 
                      'Ulnar Deviation', 'Wrist Extension Fist']
    
    # %% Confusion Matrix
    
    # Calculate test confusion matrix
    cf_matrix = confusion_matrix(true, pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=gesture_labels,
                        columns=gesture_labels)
    plt.figure(figsize=(12, 7))
    
    # Plot test confusion matrix square
    sn.set(font_scale=0.8)
    sn.heatmap(df_cm, annot=True, fmt=".0%", square=True)
    confusionMatrix_filename = f'confusionMatrix_heldout_seed{args.seed}.png'
    plt.savefig(confusionMatrix_filename)
    np.save(f'confusionMatrix_test_seed{args.seed}.npy', cf_matrix)
    wandb.log({"Testing Confusion Matrix": wandb.Image(confusionMatrix_filename),
               "Raw Testing Confusion Matrix": f'confusionMatrix_test_seed{args.seed}.npy'})
    
    # Calculate validation confusion matrix
    cf_matrix = confusion_matrix(np.argmax(Y_validation.cpu().detach().numpy(), axis=1), np.argmax(model(X_validation.to(device)).cpu().detach().numpy(), axis=1))
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=gesture_labels,
                        columns=gesture_labels)
    plt.figure(figsize=(12, 7))
    
    # Save validation confusion matrix in wandb
    sn.set(font_scale=0.8)
    sn.heatmap(df_cm, annot=True, fmt=".0%", square=True)
    confusionMatrix_filename = f'confusionMatrix_validation_seed{args.seed}.png'
    plt.savefig(confusionMatrix_filename)
    np.save(f'confusionMatrix_validation_seed{args.seed}.npy', cf_matrix)
    wandb.log({"Validation Confusion Matrix": wandb.Image(confusionMatrix_filename), 
               "Raw Validation Confusion Matrix": f'confusionMatrix_validation_seed{args.seed}.npy'})
    
    # Calculate training confusion matrix
    cf_matrix = confusion_matrix(np.argmax(Y_train.cpu().detach().numpy(), axis=1), np.argmax(model(X_train.to(device)).cpu().detach().numpy(), axis=1))
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=gesture_labels,
                        columns=gesture_labels)
    plt.figure(figsize=(12, 7))
    
    # Save training confusion matrix in wandb
    sn.set(font_scale=0.8)
    sn.heatmap(df_cm, annot=True, fmt=".0%", square=True)
    confusionMatrix_filename = f'confusionMatrix_training_seed{args.seed}.png'
    plt.savefig(confusionMatrix_filename)
    np.save(f'confusionMatrix_training_seed{args.seed}.npy', cf_matrix)
    wandb.log({"Training Confusion Matrix": wandb.Image(confusionMatrix_filename),
               "Raw Training Confusion Matrix": f'confusionMatrix_training_seed{args.seed}.npy'})
    
run.finish()
