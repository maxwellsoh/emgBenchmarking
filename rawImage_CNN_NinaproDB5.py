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

## Argument parser with optional argumenets

# Create the parser
parser = argparse.ArgumentParser(description="Include arguments for running different trials")

# Add an optional argument
parser.add_argument('--leftout_subject', type=int, help='number of subject that is left out for cross validation. Set to 0 to run standard random held-out test. Set to 0 by default.', default=0)
# Add parser for seed
parser.add_argument('--seed', type=int, help='seed for reproducibility. Set to 0 by default.', default=0)

# Parse the arguments
args = parser.parse_args()

# Use the arguments
print(f"The value of --leftout_subject is {args.leftout_subject}")
print(f"The value of --seed is {args.seed}")

# %%
# 0 for no LOSO; participants here are 1-13
leaveOut = int(args.leftout_subject)

wLen = 250 #ms
stepLen = 10 #50 ms

# Set seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def balance (restimulus):
    numZero = 0
    indices = []
    #print(restimulus.shape)
    for x in range (len(restimulus)):
        L = torch.chunk(restimulus[x], 2, dim=1)
        if torch.equal(L[0], L[1]):
            if L[0][0][0] == 0:
                if (numZero < 380):
                    #print("working")
                    indices += [x]
                numZero += 1
            else:
                indices += [x]
    return indices

def contract(R):
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(len(R), 18))
    for x in range(len(R)):
        labels[x][int(R[x][0][0])] = 1.0
    return labels

def filter(emg):
    # sixth-order Butterworth highpass filter
    b, a = butter(N=3, Wn=5, btype='highpass', analog=False, fs=200.0)
    emgButter = torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())

    #second-order notch filter at 50 Hz
    b, a = iirnotch(w0=50.0, Q=0.0001, fs=200.0)
    return torch.from_numpy(np.flip(filtfilt(b, a, emgButter),axis=0).copy())

def getRestim (n):
    restim = torch.tensor((pd.read_csv('./NinaproDB5/s' + str(n) + '/restimulusS' + str(n) + '_E2.csv')).to_numpy(), dtype=torch.float32)
    return restim.unfold(dimension=0, size=int(wLen / 5), step=stepLen)

def getEMG (n):
    restim = getRestim(n)
    emg = torch.tensor(((pd.read_csv('./NinaproDB5/s' + str(n) + '/emgS' + str(n) + '_E2.csv')).to_numpy()), dtype=torch.float32)
    return filter(emg.unfold(dimension=0, size=int(wLen / 5), step=10)[balance(restim)])

def getLabels (n):
    restim = getRestim(n)
    return contract(restim[balance(restim)])


with  multiprocessing.Pool() as pool:
    emg_async = pool.map_async(getEMG, [(i+1) for i in range(10)])
    emg = emg_async.get()
    
    labels_async = pool.map_async(getLabels, [(i+1) for i in range(10)])
    labels = labels_async.get()

length = len(emg[0][0])
width = len(emg[0][0][0])
print(length)
print(width)

if (leaveOut == 0):
    emg_in = np.concatenate([np.array(i.view(len(i), length*width)) for i in emg], axis=0, dtype=np.float16)
    s = preprocessing.StandardScaler().fit(emg_in)
    del emg_in
else:
    emg_in = np.concatenate([np.array(i.view(len(i), length*width)) for i in emg[:(leaveOut-1)]] + [np.array(i.view(len(i), length*width)) for i in emg[leaveOut:]], axis=0, dtype=np.float16)
    s = preprocessing.StandardScaler().fit(emg_in)
    del emg_in

cmap = mpl.colormaps['viridis']

def makeOneImage (data):
    data = data - min(data)
    data = data / max(data)
    data = torch.from_numpy(data).view(length, width).to(torch.float32)
    
    imageL = np.zeros((3, length, width//2))
    imageR = np.zeros((3, length, width//2))
    for p in range (length):
        for q in range (width//2):
            imageL[:, p, q] = (cmap(float(data[p][q])))[:3]
            imageR[:, p, q] = (cmap(float(data[p][q+width//2])))[:3]
    
    
    imageL = transforms.Resize([96, 112], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(torch.from_numpy(imageL))
    imageR = transforms.Resize([96, 112], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(torch.from_numpy(imageR))
    imageL = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(imageL)
    imageR = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(imageR)
    
    #imageL = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(torch.from_numpy(imageL))
    #imageR = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(torch.from_numpy(imageR))

    #allImages.append(torch.cat([imageL, imageR], dim=2).numpy().astype(np.float32))
    return torch.cat([imageL, imageR], dim=2).numpy().astype(np.float32)

def getImages (emg):
    emg = s.transform(np.array(emg.view(len(emg), length*width)))

    with  multiprocessing.Pool() as pool:
        images_async = pool.map_async(makeOneImage, [(emg[i]) for i in range(len(emg))])
        images = images_async.get()
    
    return images

data = []

# add tqdm to show progress bar
for x in tqdm(range(len(emg)), desc="Subject"):
    data += [getImages(emg[x])]


if leaveOut == 0:
    combined_labels = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)
    combined_images = np.concatenate([np.array(i) for i in data], axis=0, dtype=np.float16)
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(combined_images, combined_labels, test_size=0.2)
    X_validation, X_test, Y_validation, Y_test = model_selection.train_test_split(X_validation, Y_validation, test_size=0.5)

    X_train = torch.from_numpy(X_train).to(torch.float32)
    Y_train = torch.from_numpy(Y_train).to(torch.float32)
    X_validation = torch.from_numpy(X_validation).to(torch.float32)
    Y_validation = torch.from_numpy(Y_validation).to(torch.float32)
    X_test = torch.from_numpy(X_test).to(torch.float32)
    Y_test = torch.from_numpy(Y_test).to(torch.float32)
    print(X_train.size())
    print(Y_train.size())
    print(X_validation.size())
    print(Y_validation.size())
    print(X_test.size())
    print(Y_test.size())
else:
    X_validation = np.array(data.pop(leaveOut-1))
    Y_validation = np.array(labels.pop(leaveOut-1))
    X_train = np.concatenate([np.array(i) for i in data], axis=0, dtype=np.float16)
    Y_train = np.concatenate([np.array(i) for i in labels], axis=0, dtype=np.float16)
    X_train = torch.from_numpy(X_train).to(torch.float32)
    Y_train = torch.from_numpy(Y_train).to(torch.float32)
    X_validation = torch.from_numpy(X_validation).to(torch.float32)
    Y_validation = torch.from_numpy(Y_validation).to(torch.float32)
    print(X_train.size())
    print(Y_train.size())
    print(X_validation.size())
    print(Y_validation.size())


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
model.add_module('fc3', nn.Linear(1024, 18))
'''
model.add_module('fc1', nn.Linear(num_features, 512))
model.add_module('relu', nn.ReLU())
model.add_module('dropout1', nn.Dropout(dropout))
model.add_module('fc3', nn.Linear(512, 18))
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

class Data(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

batch_size = 64
train_loader = DataLoader(list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(list(zip(X_validation, Y_validation)), batch_size=batch_size)
if (leaveOut == 0):
    test_loader = DataLoader(list(zip(X_test, Y_test)), batch_size=batch_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
learn = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learn)

# Training loop
import gc
gc.collect()
torch.cuda.empty_cache()

if (leaveOut == 0):
    run = wandb.init(name='CNN_seed-'+str(args.seed), project='emg_benchmarking_ninapro-db5_heldout', entity='jehanyang')
else:
    run = wandb.init(name='CNN_seed-'+str(args.seed), project='emg_benchmarking_ninapro-db5_LOSO-' + str(args.leftout_subject), entity='jehanyang')
wandb.config.lr = learn

num_epochs = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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
    
run.finish()
'''

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

    cf_matrix = confusion_matrix(true, pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = np.arange(1, 19, 1),
                        columns = np.arange(1, 19, 1))
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, fmt=".3f")
    plt.savefig('output.png')
'''
