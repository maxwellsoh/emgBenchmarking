import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from scipy.signal import butter,filtfilt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import h5py

wLen = 250 #ms
stepLen = 50 #ms
freq = 4000 #Hz

def filter (emg):
    b, a = butter(N=1, Wn=120.0, btype='highpass', analog=False, fs=freq)
    return torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=-1).copy())

numFeatures = 2
def extractFeatures (emg):
    #slope sign change
    threshold = 0.01  # volts (should range between 0.00005 (50 microV) and 0.1 (100 mV))
    differences = emg[:, :, 1:] - emg[:, :, :-1]
    sign_changes = torch.logical_and(differences[:, :, :-1] * differences[:, :, 1:] < 0, torch.abs(differences[:, :, 1:]) > threshold)
    SSC = torch.sum(sign_changes, dim=2)
    
    # sum of absolute values
    SAV = torch.sum(torch.abs(emg), dim=2)

    # standard deviation of fourier transform
    emgFFT = torch.abs(torch.fft.fft(emg, dim=2))
    STD = torch.std(emgFFT, dim=2)

    # combine active features
    features = torch.cat((STD, SAV), dim=1)
    #features = torch.cat((STD, SAV, SSC), dim=1)

    # z transform
    s = preprocessing.StandardScaler().fit(features)
    return s.transform(features)

def getData(file, gesture):
    data = filter(torch.from_numpy(np.array(file[gesture])).unfold(dimension=-1, size=int(wLen/1000*freq), step=int(stepLen/1000*freq)))
    if (len(data) == 10):
        return torch.cat((data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]), axis=1).permute([1, 0, 2])
    elif (len(data) == 9):
        return torch.cat((data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]), axis=1).permute([1, 0, 2])
    else:
        return torch.cat((data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]), axis=1).permute([1, 0, 2])

def getEMG(n):
    if (n<10):
        f = h5py.File('./Jehan_Dataset/p00' + str(n) +'/data_allchannels_initial.h5', 'r')
    else:
        f = h5py.File('./Jehan_Dataset/p0' + str(n) +'/data_allchannels_initial.h5', 'r')
    #print(f.keys())
    return extractFeatures(torch.cat((getData(f, 'abduct_p1'), getData(f, 'adduct_p1'), getData(f, 'extend_p1'), getData(f, 'grip_p1'), getData(f, 'pronate_p1'),
                      getData(f, 'rest_p1'), getData(f, 'supinate_p1'), getData(f, 'tripod_p1'), getData(f, 'wextend_p1'), getData(f, 'wflex_p1')), axis=0))

emg_8 = getEMG(8)
emg_9 = getEMG(9)
emg_11 = getEMG(11)
emg_12 = getEMG(12)
emg_13 = getEMG(13)
emg_15 = getEMG(15)
emg_16 = getEMG(16)
emg_17 = getEMG(17)
emg_18 = getEMG(18)
emg_19 = getEMG(19)
emg_20 = getEMG(20)
emg_21 = getEMG(21)
emg_22 = getEMG(22)
label_8 = np.load("./Jehan_Dataset/labels_8.npy")
label_9 = np.load("./Jehan_Dataset/labels.npy")
label_11 = np.load("./Jehan_Dataset/labels.npy")
label_12 = np.load("./Jehan_Dataset/labels.npy")
label_13 = np.load("./Jehan_Dataset/labels_13.npy")
label_15 = np.load("./Jehan_Dataset/labels.npy")
label_16 = np.load("./Jehan_Dataset/labels.npy")
label_17 = np.load("./Jehan_Dataset/labels.npy")
label_18 = np.load("./Jehan_Dataset/labels_18.npy")
label_19 = np.load("./Jehan_Dataset/labels.npy")
label_20 = np.load("./Jehan_Dataset/labels_20.npy")
label_21 = np.load("./Jehan_Dataset/labels_21.npy")
label_22 = np.load("./Jehan_Dataset/labels_22.npy")

leaveOut = True

if leaveOut:
    outEMG = emg_22
    outLabels = label_22
    data = np.concatenate((emg_8, emg_9, emg_11, emg_12, emg_13, emg_15, emg_16, emg_17, emg_18, emg_19, emg_20, emg_21), axis=0, dtype=np.float32)
    combined_labels = np.concatenate((label_8, label_9, label_11, label_12, label_13, label_15, label_16, label_17, label_18, label_19,
                                      label_20, label_21), axis=0, dtype=np.float32)
else:
    data = np.concatenate((emg_8, emg_9, emg_11, emg_12, emg_13, emg_15, emg_16, emg_17, emg_18, emg_19, emg_20, emg_21, emg_22), axis=0, dtype=np.float32)
    combined_labels = np.concatenate((label_8, label_9, label_11, label_12, label_13, label_15, label_16, label_17, label_18, label_19, label_20,
                                      label_21, label_22), axis=0, dtype=np.float32)

if leaveOut:
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(data, combined_labels, test_size=0.1)
    X_test = outEMG
    Y_test = outLabels
else:
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(data, combined_labels, test_size=0.2)
    X_validation, X_test, Y_validation, Y_test = model_selection.train_test_split(X_validation, Y_validation, test_size=0.5)

X_train = torch.from_numpy(X_train).to(torch.float32)
Y_train = torch.from_numpy(Y_train).to(torch.float32)
X_validation = torch.from_numpy(X_validation).to(torch.float32)
Y_validation = torch.from_numpy(Y_validation).to(torch.float32)
X_test = torch.from_numpy(X_test).to(torch.float32)
Y_test = torch.from_numpy(Y_test).to(torch.float32)
print(X_train.shape)
print(X_validation.shape)
print(X_test.shape)

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
test_loader = DataLoader(list(zip(X_test, Y_test)), batch_size=batch_size)



criterion = nn.CrossEntropyLoss()
learn = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learn)

# Training loop
import gc
gc.collect()
torch.cuda.empty_cache()

#run = wandb.init(name='CNN', project='emg_benchmarking', entity='msoh')
#wandb.config.lr = learn

num_epochs = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

#wandb.watch(model)

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

    '''
    wandb.log({
        "Epoch": epoch,
        "Train Loss": train_loss,
        "Train Acc": train_acc,
        "Valid Loss": val_loss,
        "Valid Acc": val_acc})
    '''
#run.finish()

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

cf_matrix = confusion_matrix(true, pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = np.arange(1, 11, 1),
                     columns = np.arange(1, 11, 1))
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, fmt=".3f")
plt.savefig('output.png')