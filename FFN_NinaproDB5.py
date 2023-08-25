# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from scipy.signal import butter,filtfilt,iirnotch
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import random

# %%
# emg and restimulus as tensors (E2 files only)
emg_S1 = torch.tensor(((pd.read_csv('./NinaproDB5/s1/emgS1_E2.csv')).to_numpy()), dtype=torch.int)
restimulus_S1 = torch.tensor((pd.read_csv('./NinaproDB5/s1/restimulusS1_E2.csv')).to_numpy(), dtype=torch.int)
emg_S2 = torch.tensor(((pd.read_csv('./NinaproDB5/s2/emgS2_E2.csv')).to_numpy()), dtype=torch.int)
restimulus_S2 = torch.tensor((pd.read_csv('./NinaproDB5/s2/restimulusS2_E2.csv')).to_numpy(), dtype=torch.int)
emg_S3 = torch.tensor(((pd.read_csv('./NinaproDB5/s3/emgS3_E2.csv')).to_numpy()), dtype=torch.int)
restimulus_S3 = torch.tensor((pd.read_csv('./NinaproDB5/s3/restimulusS3_E2.csv')).to_numpy(), dtype=torch.int)
emg_S4 = torch.tensor(((pd.read_csv('./NinaproDB5/s4/emgS4_E2.csv')).to_numpy()), dtype=torch.int)
restimulus_S4 = torch.tensor((pd.read_csv('./NinaproDB5/s4/restimulusS4_E2.csv')).to_numpy(), dtype=torch.int)
emg_S5 = torch.tensor(((pd.read_csv('./NinaproDB5/s5/emgS5_E2.csv')).to_numpy()), dtype=torch.int)
restimulus_S5 = torch.tensor((pd.read_csv('./NinaproDB5/s5/restimulusS5_E2.csv')).to_numpy(), dtype=torch.int)
emg_S6 = torch.tensor(((pd.read_csv('./NinaproDB5/s6/emgS6_E2.csv')).to_numpy()), dtype=torch.int)
restimulus_S6 = torch.tensor((pd.read_csv('./NinaproDB5/s6/restimulusS6_E2.csv')).to_numpy(), dtype=torch.int)
emg_S7 = torch.tensor(((pd.read_csv('./NinaproDB5/s7/emgS7_E2.csv')).to_numpy()), dtype=torch.int)
restimulus_S7 = torch.tensor((pd.read_csv('./NinaproDB5/s7/restimulusS7_E2.csv')).to_numpy(), dtype=torch.int)
emg_S8 = torch.tensor(((pd.read_csv('./NinaproDB5/s8/emgS8_E2.csv')).to_numpy()), dtype=torch.int)
restimulus_S8 = torch.tensor((pd.read_csv('./NinaproDB5/s8/restimulusS8_E2.csv')).to_numpy(), dtype=torch.int)
emg_S9 = torch.tensor(((pd.read_csv('./NinaproDB5/s9/emgS9_E2.csv')).to_numpy()), dtype=torch.int)
restimulus_S9 = torch.tensor((pd.read_csv('./NinaproDB5/s9/restimulusS9_E2.csv')).to_numpy(), dtype=torch.int)
emg_S10 = torch.tensor(((pd.read_csv('./NinaproDB5/s10/emgS10_E2.csv')).to_numpy()), dtype=torch.int)
restimulus_S10 = torch.tensor((pd.read_csv('./NinaproDB5/s10/restimulusS10_E2.csv')).to_numpy(), dtype=torch.int)

# windowing
wLen = 250 #ms
def window (e):
    return e.unfold(dimension=0, size=int(wLen / 5), step=10)

leaveOut = True
emgOut = torch.tensor(())
restimulusOut = torch.tensor(())
if leaveOut:
    emgOut = window(emg_S1)
    restimulusOut = window(restimulus_S1)
#print(emgOut.size())

emg = torch.cat((window(emg_S2), window(emg_S3), window(emg_S4), window(emg_S5), 
                 window(emg_S6), window(emg_S7), window(emg_S8), window(emg_S9), window(emg_S10)), dim=0)
#print(emg.size())

restimulus = torch.cat((window(restimulus_S2), window(restimulus_S3), 
                        window(restimulus_S4), window(restimulus_S5), window(restimulus_S6), window(restimulus_S7), 
                        window(restimulus_S8), window(restimulus_S9), window(restimulus_S10)), dim=0)

# %%
#delete windows with both 0's and other number (for restimulus labels) and excess rest windows
def filterIndices (labels):
    indices = []
    numLabel = {}
    for x in range(len(labels)):
        L = torch.chunk(labels[x], 2, dim=1)
        if torch.equal(L[0], L[1]):
            if L[0][0][0] == 0:
                r = random.random()
                if r > 0.96:
                    indices += [x]
                    if 0 in numLabel:
                        numLabel[0] += 1
                    else:
                        numLabel[0] = 1
            else:
                indices += [x]
                temp = int(L[0][0][0])
                if temp in numLabel:
                    numLabel[temp] += 1
                else:
                    numLabel[temp] = 1
    #print(numLabel)
    return indices

indices = filterIndices(restimulus)
emgDel = emg[indices]
restimulusDel = restimulus[indices]
#leave out
indices = filterIndices(restimulusOut)
emgOutDel = emgOut[indices]
restimulusOutDel = restimulusOut[indices]

# %%
# contract restimulus into length 1
def contract(R):
    labels = torch.tensor(())
    labels = labels.new_zeros(size=(len(R), 18))
    for x in range(len(R)):
        labels[x][int(R[x][0][0])] = 1.0
    #print(labels.size())
    return labels

labels = contract(restimulusDel)
#leave out
outLabels = contract(restimulusOutDel)

# %%
# sixth-order Butterworth highpass filter
b, a = butter(N=3, Wn=5, btype='highpass', analog=False, fs=200.0)
emgButter = torch.from_numpy(np.flip(filtfilt(b, a, emgDel),axis=0).copy())
#leave out
if leaveOut:
    emgOutButter = torch.from_numpy(np.flip(filtfilt(b, a, emgOutDel),axis=0).copy())

# %%
#second-order notch filter at 50 Hz
b, a = iirnotch(w0=50.0, Q=0.0001, fs=200.0)
emgNotch = torch.from_numpy(np.flip(filtfilt(b, a, emgButter),axis=0).copy())
#leave out
if leaveOut:
    emgOutNotch = torch.from_numpy(np.flip(filtfilt(b, a, emgOutButter),axis=0).copy())

# %%
numFeatures = 3
def extractFeatures (emg):
    #slope sign change
    threshold = 0.01  # volts (should range between 0.00005 (50 microV) and 0.1 (100 mV))
    differences = emg[:, :, 1:] - emg[:, :, :-1]
    sign_changes = torch.logical_and(differences[:, :, :-1] * differences[:, :, 1:] < 0, torch.abs(differences[:, :, 1:]) > threshold)
    SSC = torch.sum(sign_changes, dim=2)

    print(SSC.size())

    # feature extraction: sum of absolute values
    SAV = torch.sum(torch.abs(emg), dim=2)

    #mean absolute value
    MAV = torch.mean(torch.abs(emg), dim=2)

    # feature extraction: standard deviation of fourier transform
    # based on formula in section 3.2 of https://www.sciencedirect.com/science/article/pii/S0925231220303283?via%3Dihub#sec0006
    emgFFT = torch.abs(torch.fft.fft(emg, dim=2))
    STD = torch.std(emgFFT, dim=2)

    # root mean squared
    RMS = torch.sqrt(torch.mean(emg ** 2, dim=2))

    # waveform length
    wl = torch.sum(torch.abs(emg[:, :, 1:] - emg[:, :, :-1]), dim=2)

    # mean frequency
    power_spectrum = torch.abs(torch.fft.fft(emg, dim=2)) ** 2
    sampling_frequency = 200  # sampling frequency in Hz
    frequency = torch.fft.fftfreq(emg.shape[2], d=1/sampling_frequency)
    frequency = frequency.unsqueeze(0).unsqueeze(1)
    sum_product = torch.sum(power_spectrum * frequency, dim=2)
    total_sum = torch.sum(power_spectrum, dim=2)
    MNF = sum_product / total_sum

    # power spectrum ratio
    max_index = torch.argmax(power_spectrum, dim=2)
    window_size = 10  # Adjustable range around maximum value to consider for P0
    lower_index = torch.clamp(max_index.unsqueeze(2) - window_size, min=0)
    upper_index = torch.clamp(max_index.unsqueeze(2) + window_size + 1, max=power_spectrum.shape[2])
    P = torch.sum(power_spectrum, dim=2)
    P0 = torch.zeros_like(P)
    for i in range(lower_index.shape[0]):
        for j in range(lower_index.shape[1]):
            P0[i, j] = torch.sum(power_spectrum[i, j, lower_index[i, j, 0]:upper_index[i, j, 0]], dim=0)    
    PSR = P0 / P
    #print(PSR)

    # combine active features
    features = torch.cat((STD, SAV, SSC), dim=1)

    # z transform
    s = preprocessing.StandardScaler().fit(features)
    return torch.from_numpy(s.transform(features))

# combine features and labels
combined = torch.cat((extractFeatures(emgNotch), labels), dim=1)
#leaveOut
if leaveOut:
    combinedOut = torch.cat((extractFeatures(emgOutNotch), outLabels), dim=1)

# split into training, validation, and test set
train, validation = model_selection.train_test_split(combined.numpy(), test_size=0.2)
if leaveOut:
    test = combinedOut.to(torch.float32)
else:
    train, test = model_selection.train_test_split(train, test_size=0.2)
    test = torch.from_numpy(test).to(torch.float32)

train = torch.from_numpy(train).to(torch.float32)
validation = torch.from_numpy(validation).to(torch.float32)

#print(train.size())
#print(validation.size())
#print(test.size())

# %%
class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(FullyConnectedNet, self).__init__()

        layers = []
        layer_sizes = [input_size] + hidden_sizes

        # hidden layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # output layer
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# architecture parameters
input_size = 16*numFeatures
hidden_sizes = [1024, 1024, 1024]
output_size = 18  # 17 gestures + 1 rest gesture
dropout_rate = 0.5

model = FullyConnectedNet(input_size, hidden_sizes, output_size, dropout_rate)

# %%
class Data(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

batch_size = 64    
train_loader = DataLoader(Data(train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Data(validation), batch_size=batch_size)
test_loader = DataLoader(Data(test), batch_size=batch_size)

# %%
# loss function and optimizer
criterion = nn.CrossEntropyLoss()
learn = 0.0003
optimizer = torch.optim.Adam(model.parameters(), lr=learn)

# %%
# Training loop
run = wandb.init(name='Softmax', project='emg_benchmarking', entity='msoh')
wandb.config.lr = learn

num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

wandb.watch(model)

for epoch in range(num_epochs):
    model.train()
    train_acc = 0.0
    train_loss = 0.0
    for batch_data in train_loader:
        batch_data = batch_data.to(device)

        data, labels = torch.split(batch_data, [16*numFeatures, 18], dim=1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        train_loss += loss.item()

        train_acc += np.mean(np.argmax(output.cpu().detach().numpy(), axis=1) == np.argmax(labels.cpu().detach().numpy(), axis=1))

        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data.to(device)

            data, labels = torch.split(batch_data, [16*numFeatures, 18], dim=1)

            output = model(data)
            val_loss += criterion(output, labels).item()

            val_acc += np.mean(np.argmax(output.cpu().detach().numpy(), axis=1) == np.argmax(labels.cpu().detach().numpy(), axis=1))

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    #print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")

    wandb.log({
        "Epoch": epoch,
        "Train Loss": loss.item(),
        "Train Acc": train_acc,
        "Valid Loss": val_loss,
        "Valid Acc": val_acc})
    
run.finish()

# %%
# Testing
pred = []
true = []

model.eval()
test_loss = 0.0
test_acc = 0.0
with torch.no_grad():
    for batch_data in test_loader:
        batch_data = batch_data.to(device)

        data, labels = torch.split(batch_data, [16*numFeatures, 18], dim=1)

        output = model(data)
        test_loss += criterion(output, labels).item()

        test_acc += np.mean(np.argmax(output.cpu().detach().numpy(), axis=1) == np.argmax(labels.cpu().detach().numpy(), axis=1))

        output = np.argmax(output.cpu().detach().numpy(), axis=1)
        pred.extend(output)
        labels = np.argmax(labels.cpu().detach().numpy(), axis=1)
        true.extend(labels)

test_loss /= len(test_loader)
test_acc /= len(test_loader)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

cf_matrix = confusion_matrix(true, pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = np.arange(0, 18, 1),
                     columns = np.arange(0, 18, 1))
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, fmt=".2f")
plt.savefig('output.png')


