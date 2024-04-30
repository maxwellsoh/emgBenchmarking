import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from scipy.signal import butter,filtfilt,iirnotch
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import gc

# mat to tensor
wLen = 250.0 #ms

def getEMG (subject, wLen):
    freq = 2000.0 #Hz
    sub = str(subject+1)
    mat_data = scipy.io.loadmat('./NinaproDB2/DB2_s' + sub + '/S' + sub + '_E1_A1.mat')
    mat_array = mat_data['emg']
    return torch.from_numpy(mat_array).unfold(dimension=0, size=int(wLen/1000.0*freq), step=int(wLen/5000.0*freq))

def getRestimulus (subject, wLen):
    freq = 2000.0 #Hz
    sub = str(subject+1)
    mat_data = scipy.io.loadmat('./NinaproDB2/DB2_s' + sub + '/S' + sub + '_E1_A1.mat')
    mat_array = mat_data['restimulus']
    return torch.from_numpy(mat_array).unfold(dimension=0, size=int(wLen/1000.0*freq), step=int(wLen/5000.0*freq))

# removing excess rest windows
def balance (restimulus):
    numZero = 0
    indices = []
    for x in range (len(restimulus)):
        L = torch.chunk(restimulus[x], 2, dim=1)
        if torch.equal(L[0], L[1]):
            if L[0][0][0] == 0:
                if (numZero < 550):
                    indices += [x]
                numZero += 1
            else:
                indices += [x]
    return indices

#b, a = butter(N=3, Wn=[20,450], btype='bandpass', analog=False, fs=2000.0)
#b, a = butter(N=3, Wn=5, btype='highpass', analog=False, fs=2000.0)
# sixth-order butterworth filter only as powerline interferenced removed in dataset
def filter(emg):
    #return emg
    #b, a = butter(N=1, Wn=300.0, btype='lowpass', analog=False, fs=2000.0)
    #b, a = butter(N=1, Wn=200.0, btype='highpass', analog=False, fs=2000.0)
    b, a = butter(N=1, Wn=100.0, btype='highpass', analog=False, fs=2000.0)
    #b, a = butter(N=1, Wn=(0.001, 150), btype='bandpass', analog=False, fs=2000.0)
    emgButter = torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())
    b, a = iirnotch(w0=50.0, Q=0.0001, fs=2000.0)
    return torch.from_numpy(np.flip(filtfilt(b, a, emgButter),axis=0).copy())
    return emgButter

numFeatures = 2
def extractFeatures (n):
    emg = filter(getEMG(n)[balance(getRestimulus(n))])
    #slope sign change
    '''
    threshold = 0.01  # volts (should range between 0.00005 (50 microV) and 0.1 (100 mV))
    differences = emg[:, :, 1:] - emg[:, :, :-1]
    sign_changes = torch.logical_and(differences[:, :, :-1] * differences[:, :, 1:] < 0, torch.abs(differences[:, :, 1:]) > threshold)
    SSC = torch.sum(sign_changes, dim=2)
    '''
    
    # sum of absolute values
    SAV = torch.sum(torch.abs(emg), dim=2)

    # standard deviation of fourier transform
    emgFFT = torch.abs(torch.fft.fft(emg, dim=2))
    STD = torch.std(emgFFT, dim=2)

    # combine active features
    #features = torch.cat((STD, SAV), dim=1)
    features = torch.cat((STD, SAV), dim=1)

    # z transform
    s = preprocessing.StandardScaler().fit(features)
    return s.transform(features)

emg = []
labels = []
for i in range (40):
    emg += [extractFeatures(i)]
    labels += [np.load("./NinaproDB2/label_" + str(i+1) + ".npy")]

leaveOut = True
leftOut = 37

if leaveOut:
    outEMG = emg[leftOut-1]
    outLabels = labels[leftOut-1]
    del emg[leftOut-1]
    del labels[leftOut-1]
    data = np.concatenate(emg, axis=0, dtype=np.float32)
    combined_labels = np.concatenate(labels, axis=0, dtype=np.float32)
else:
    data = np.concatenate(emg, axis=0, dtype=np.float32)
    combined_labels = np.concatenate(labels, axis=0, dtype=np.float32)

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
learn = 0.0003
optimizer = torch.optim.Adam(model.parameters(), lr=learn)

# Training loop
gc.collect()
torch.cuda.empty_cache()

#run = wandb.init(name='CNN', project='emg_benchmarking', entity='msoh')
#wandb.config.lr = learn

num_epochs = 100
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
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = np.arange(1, 19, 1),
                     columns = np.arange(1, 19, 1))
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, fmt=".3f")
plt.savefig('output.png')