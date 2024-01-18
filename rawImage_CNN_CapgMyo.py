import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from scipy import io
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import argparse
import utils_NinaproDB5 as ut_NDB5
import random
import datetime
import multiprocessing
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import timm

# script not updated for LOSO
leaveOut = 0

sigma_coefficient = 0.5
numElectrodes = 128
cmap = mpl.colormaps['viridis']

## Argument parser with optional argumenets

# Create the parser
parser = argparse.ArgumentParser(description="Include arguments for running different trials")

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
# Add argument for model to use
parser.add_argument('--model', type=str, help='model to use (e.g. \'convnext_tiny_custom\', \'convnext_tiny\', \'davit_tiny.msft_in1k\', \'efficientnet_b3.ns_jft_in1k\', \'vit_tiny_path16_224\', \'efficientnet_b0\'). Set to resnet50 by default.', default='resnet50')

# Parse the arguments
args = parser.parse_args()

# Use the arguments
print(f"The value of --seed is {args.seed}")
print(f"The value of --epochs is {args.epochs}")
print(f"The model to use is {args.model}")
if args.turn_on_kfold:
    print(f"The value of --kfold is {args.kfold}")
    print(f"The value of --fold_index is {args.fold_index}")

# Add date and time to filename
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

print("------------------------------------------------------------------------------------------------------------------------")
print("Starting run at", formatted_datetime)
print("------------------------------------------------------------------------------------------------------------------------")

# Set seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# windowing
wLen = 250 #ms
def window (e):
    return e.unfold(dimension=0, size=wLen, step=50)

# mat to tensor
def getData (subject, gesture, trial):
    sub = str(subject)
    if subject < 10:
        sub = '0' + str(subject)
    name = '0' + sub + '-00' + str(gesture) + '-00' +str(trial)
    if trial == 10:
        name = '0' + sub + '-00' + str(gesture) + '-010'
    mat_data = io.loadmat('./CapgMyo_B/dbb-preprocessed-0' + sub + '/' + name + '.mat')
    mat_array = mat_data['data']
    tensor_data = torch.from_numpy(mat_array)
    if trial < 10:
        return torch.cat((window(tensor_data), getData(subject, gesture, trial + 1)), dim=0)
    elif gesture < 8:
        return torch.cat((window(tensor_data), getData(subject, gesture + 1, 1)), dim=0)
    else:
        return window(tensor_data)
        
def makeLabels (emg):
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(len(emg), 8))
    for x in range (8):
        for y in range (int(len(emg) / 8)):
            labels[int(x * (len(emg) / 8) + y)][x] = 1.0
    return labels

def optimized_makeOneMagnitudeImage(data, length, width, resize_length_factor, native_resnet_size, global_min, global_max):
    # Normalize with global min and max
    data = (data - global_min) / (global_max - global_min)
    data_converted = cmap(data)
    rgb_data = data_converted[:, :3]
    image_data = np.reshape(rgb_data, (numElectrodes, width, 3))
    image = np.transpose(image_data, (2, 0, 1))
    
    # Split image and resize
    imageL, imageR = np.split(image, 2, axis=2)
    resize = transforms.Resize([length * resize_length_factor, native_resnet_size // 2],
                               interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    imageL, imageR = map(lambda img: resize(torch.from_numpy(img)), (imageL, imageR))
    
    # Clamp between 0 and 1 using torch.clamp
    imageL, imageR = map(lambda img: torch.clamp(img, 0, 1), (imageL, imageR))
    
    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imageL, imageR = map(normalize, (imageL, imageR))
    
    return torch.cat([imageL, imageR], dim=2).numpy().astype(np.float32)

def optimized_makeOneImage(data, cmap, length, width, resize_length_factor, native_resnet_size):
    # Contrast normalize and convert data
    # NOTE: Should this be contrast normalized? Then only patterns of data will be visible, not absolute values
    data = (data - data.min()) / (data.max() - data.min())
    data_converted = cmap(data)
    rgb_data = data_converted[:, :3]
    image_data = np.reshape(rgb_data, (numElectrodes, width, 3))
    image = np.transpose(image_data, (2, 0, 1))
    
    # Split image and resize
    imageL, imageR = np.split(image, 2, axis=2)
    resize = transforms.Resize([length * resize_length_factor, native_resnet_size // 2],
                               interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    imageL, imageR = map(lambda img: resize(torch.from_numpy(img)), (imageL, imageR))
    
    # Get max and min values after interpolation
    max_val = max(imageL.max(), imageR.max())
    min_val = min(imageL.min(), imageR.min())
    
    # Contrast normalize again after interpolation
    imageL, imageR = map(lambda img: (img - min_val) / (max_val - min_val), (imageL, imageR))
    
    # Normalize with standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imageL, imageR = map(normalize, (imageL, imageR))
    
    return torch.cat([imageL, imageR], dim=2).numpy().astype(np.float32)

def calculate_rms(array_2d):
    # Calculate RMS for 2D array where each row is a window
    return np.sqrt(np.mean(array_2d**2))

def getImages(emg, standardScaler, length, width, turn_on_rms=False, rms_windows=10, turn_on_magnitude=False, global_min=None, global_max=None):

    emg = standardScaler.transform(np.array(emg.view(len(emg), length*width)))
    # Use RMS preprocessing
    if turn_on_rms:
        emg = emg.reshape(len(emg), length, width)
        # Reshape data for RMS calculation: (SAMPLES, 16, 5, 10)
        emg = emg.reshape(len(emg), length, rms_windows, width // rms_windows)
        
        # Apply RMS calculation along the last axis (axis=-1)
        emg_rms = np.apply_along_axis(calculate_rms, -1, emg)
        emg = emg_rms  # Resulting shape will be (SAMPLES, 16, 5)
        width = rms_windows
        emg = emg.reshape(len(emg), length*width)

    # Parameters that don't change can be set once
    resize_length_factor = 1
    if turn_on_magnitude:
        resize_length_factor = 1
    native_resnet_size = 224

    with multiprocessing.Pool(processes=5) as pool:
        args = [(emg[i], cmap, length, width, resize_length_factor, native_resnet_size) for i in range(len(emg))]
        images_async = pool.starmap_async(optimized_makeOneImage, args)
        images = images_async.get()

    if turn_on_magnitude:
        with multiprocessing.Pool(processes=5) as pool:
            args = [(emg[i], length, width, resize_length_factor, native_resnet_size, global_min, global_max) for i in range(len(emg))]
            images_async = pool.starmap_async(optimized_makeOneMagnitudeImage, args)
            images_magnitude = images_async.get()
        images = np.concatenate((images, images_magnitude), axis=2)
    
    return images

def getEMG (x):
    return torch.cat((getData(x-1,1,1), getData(x,1,1)), dim=0)

with  multiprocessing.Pool() as pool:
    emg_async = pool.map_async(getEMG, [i for i in range(2, 22, 2)])
    emg = emg_async.get() # (SUBJECT, TRIAL, CHANNEL, TIME)
    
    labels_async = pool.map_async(makeLabels, emg)
    labels = labels_async.get()

length = len(emg[0][0])
width = len(emg[0][0][0])
print("Number of Electrode Channels: ", length)
print("Number of Timesteps per Trial:", width)

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
    emg_reshaped = emg_in_by_electrode.reshape(-1, numElectrodes)

    # Initialize and fit the scaler on the reshaped data
    # This will compute the mean and std dev for each electrode across all samples and features
    scaler = preprocessing.StandardScaler()
    scaler.fit(emg_reshaped)
    
    # Repeat means and std_devs for each time point using np.repeat
    scaler.mean_ = np.repeat(scaler.mean_, width)
    scaler.scale_ = np.repeat(scaler.scale_, width)
    scaler.var_ = np.repeat(scaler.var_, width)
    scaler.n_features_in_ = width*numElectrodes

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
    emg_reshaped = emg_in_by_electrode.reshape(-1, numElectrodes)

    # Initialize and fit the scaler on the reshaped data
    # This will compute the mean and std dev for each electrode across all samples and features
    scaler = preprocessing.StandardScaler()
    scaler.fit(emg_reshaped)
    
    # Repeat means and std_devs for each time point using np.repeat
    scaler.mean_ = np.repeat(scaler.mean_, width)
    scaler.scale_ = np.repeat(scaler.scale_, width)
    scaler.var_ = np.repeat(scaler.var_, width)
    scaler.n_features_in_ = width*numElectrodes

    del emg_in
    del labels_in

    del train_emg_in
    del indices

    del emg_in_by_electrode
    del emg_reshaped

data = []

# add tqdm to show progress bar
for x in tqdm(range(len(emg)), desc="Number of Subjects "):
    data += [getImages(emg[x], scaler, length, width, turn_on_rms=False, rms_windows=10, turn_on_magnitude=False, global_min=global_min, global_max=global_max)]

print("------------------------------------------------------------------------------------------------------------------------")
print("NOTE: The width 224 is natively used in Resnet50, height is currently integer  multiples of number of electrode channels")
print("------------------------------------------------------------------------------------------------------------------------")
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

class Data(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

batch_size = 64
train_loader = DataLoader(list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=ut_NDB5.seed_worker, pin_memory=True)
val_loader = DataLoader(list(zip(X_validation, Y_validation)), batch_size=batch_size, num_workers=4, worker_init_fn=ut_NDB5.seed_worker, pin_memory=True)
test_loader = DataLoader(list(zip(X_test, Y_test)), batch_size=batch_size, num_workers=4, worker_init_fn=ut_NDB5.seed_worker, pin_memory=True)

model_name = args.model
if args.model == 'resnet50':
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-2])
    num_features = model[-1][-1].conv3.out_channels
    dropout = 0.5
    model.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
    model.add_module('flatten', nn.Flatten())
    model.add_module('fc1', nn.Linear(num_features, 512))
    model.add_module('relu', nn.ReLU())
    model.add_module('dropout1', nn.Dropout(dropout))
    model.add_module('fc2', nn.Linear(512, 512))
    model.add_module('relu2', nn.ReLU())
    model.add_module('dropout2', nn.Dropout(dropout))
    model.add_module('fc3', nn.Linear(512, 8))
    model.add_module('softmax', nn.Softmax(dim=1))

    num = 0
    for name, param in model.named_parameters():
        num += 1
        if (num > 72):
        #if (num > 33): #for -4
            param.requires_grad = True
        else:
            param.requires_grad = False
else:
    model = timm.create_model(model_name, pretrained=True, num_classes=8)
    for name, param in model.named_parameters():
        param.requires_grad = True

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
learn = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learn)

# Training loop
import gc
gc.collect()
torch.cuda.empty_cache()

num_epochs = args.epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

wandb_runname = 'CNN_seed-'+str(args.seed)
if args.turn_on_kfold:
    wandb_runname += '_k-fold-'+str(args.kfold)+'_fold-index-'+str(args.fold_index)
wandb_runname += '_' + model_name

project_name = 'emg_benchmarking_CapgMyo-b'
if args.turn_on_kfold:
    project_name += '_k-fold-'+str(args.kfold)
else:
    project_name += '_heldout'

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

for epoch in range(num_epochs):
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
'''
cf_matrix = confusion_matrix(true, pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = np.arange(1, 9, 1),
                     columns = np.arange(1, 9, 1))
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, fmt=".3f")
plt.savefig('output.png')
'''

run.finish()