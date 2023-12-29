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

## Argument parser with optional argumenets

# Create the parser
parser = argparse.ArgumentParser(description="Include arguments for running different trials")

# Add an optional argument
parser.add_argument('--leftout_subject', type=int, help='number of subject that is left out for cross validation. Set to 0 to run standard random held-out test. Set to 0 by default.', default=0)
# Add parser for seed
parser.add_argument('--seed', type=int, help='seed for reproducibility. Set to 0 by default.', default=0)
# Add number of epochs to train for
parser.add_argument('--epochs', type=int, help='number of epochs to train for. Set to 25 by default.', default=25)

# Parse the arguments
args = parser.parse_args()

# Use the arguments
print(f"The value of --leftout_subject is {args.leftout_subject}")
print(f"The value of --seed is {args.seed}")
print(f"The value of --epochs is {args.epochs}")

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
    emg_in = np.concatenate([np.array(i.view(len(i), length*width)) for i in emg], axis=0, dtype=np.float16)
    s = preprocessing.StandardScaler().fit(emg_in)
    del emg_in
else:
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
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(combined_images, combined_labels, test_size=0.2, stratify=combined_labels)
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
learn = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learn)

# Define the cyclical learning rate scheduler
step_size = len(train_loader) * 2  # Number of iterations in half a cycle
base_lr = 0.0001  # Minimum learning rate
max_lr = 0.01  # Maximum learning rate
# number_annealing_cycles = 5
# annealing_multiplier = 2
# cawr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs/(pow(annealing_multiplier,number_annealing_cycles)-1), 
#                                                                       T_mult=annealing_multiplier, eta_min=0.01, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=step_size, mode='triangular2', cycle_momentum=False)

# Training loop
import gc
gc.collect()
torch.cuda.empty_cache()

if (leaveOut == 0):
    run = wandb.init(name='CNN_seed-'+str(args.seed)+"_cyclical-learningrate", project='emg_benchmarking_ninapro-db5_heldout', entity='jehanyang')
else:
    run = wandb.init(name='CNN_seed-'+str(args.seed)+"_cyclical-learningrate", project='emg_benchmarking_ninapro-db5_LOSO-' + str(args.leftout_subject), entity='jehanyang')
wandb.config.lr = learn

num_epochs = args.epochs
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
        "Valid Acc": val_acc})
    

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

    cf_matrix = confusion_matrix(true, pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=gesture_labels,
                        columns=gesture_labels)
    plt.figure(figsize=(12, 7))
    
    # Plot confusion matrix square
    sn.set(font_scale=0.8)
    sn.heatmap(df_cm, annot=True, fmt=".0%", square=True)
    confusionMatrix_filename = f'confusionMatrix_heldout_seed{args.seed}.png'
    plt.savefig(confusionMatrix_filename)

    # Save confusion matrix in Wandb
    wandb.log({"Confusion Matrix": wandb.Image(confusionMatrix_filename)})
    

run.finish()
