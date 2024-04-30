# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import scipy

# %%
# windowing
wLen = 250 #ms
def window (e):
    return e.unfold(dimension=0, size=wLen, step=50)

#mat to tensor
def getData (subject, gesture, trial):
    sub = str(subject)
    if subject < 10:
        sub = '0' + str(subject)
    name = '0' + sub + '-00' + str(gesture) + '-00' +str(trial)
    if trial == 10:
        name = '0' + sub + '-00' + str(gesture) + '-010'
    mat_data = scipy.io.loadmat('./CapgMyo_B/dbb-preprocessed-0' + sub + '/' + name + '.mat')
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

S1_emg = torch.cat((getData(1, 1, 1), getData(2, 1, 1)), dim=0)
S1_labels = makeLabels(S1_emg)
S2_emg = torch.cat((getData(3, 1, 1), getData(4, 1, 1)), dim=0)
S2_labels = makeLabels(S2_emg)
S3_emg = torch.cat((getData(5, 1, 1), getData(6, 1, 1)), dim=0)
S3_labels = makeLabels(S3_emg)
S4_emg = torch.cat((getData(7, 1, 1), getData(8, 1, 1)), dim=0)
S4_labels = makeLabels(S4_emg)
S5_emg = torch.cat((getData(9, 1, 1), getData(10, 1, 1)), dim=0)
S5_labels = makeLabels(S5_emg)
S6_emg = torch.cat((getData(11, 1, 1), getData(12, 1, 1)), dim=0)
S6_labels = makeLabels(S6_emg)
S7_emg = torch.cat((getData(13, 1, 1), getData(14, 1, 1)), dim=0)
S7_labels = makeLabels(S7_emg)
S8_emg = torch.cat((getData(15, 1, 1), getData(16, 1, 1)), dim=0)
S8_labels = makeLabels(S8_emg)
S9_emg = torch.cat((getData(17, 1, 1), getData(18, 1, 1)), dim=0)
S9_labels = makeLabels(S9_emg)
S10_emg = torch.cat((getData(19, 1, 1), getData(20, 1, 1)), dim=0)
S10_labels = makeLabels(S10_emg)

leaveOut = True
outEMG = torch.tensor(())
outLabels = torch.tensor(())
if leaveOut:
    outEMG = S8_emg
    outLabels = S8_labels
#print(outEMG.size())

emg = torch.cat((S1_emg, S2_emg, S3_emg, S4_emg, S5_emg, S6_emg, 
                 S7_emg, S9_emg, S10_emg), dim=0)
labels = torch.cat((S1_labels, S2_labels, S3_labels, S4_labels, 
                    S5_labels, S6_labels, S7_labels, S9_labels, S10_labels), dim=0)
#print(emg.size())

# %%
numFeatures = 3
def extractFeatures (emg):
    #slope sign change
    threshold = 0.1  # volts (should range between 0.00005 (50 microV) and 0.1 (100 mV))
    differences = emg[:, :, 1:] - emg[:, :, :-1]
    sign_changes = torch.logical_and(differences[:, :, :-1] * differences[:, :, 1:] < 0, 
                                     torch.abs(differences[:, :, 1:]) > threshold)
    SSC = torch.sum(sign_changes, dim=2)

    # sum of absolute values
    SAV = torch.sum(torch.abs(emg), dim=2)

    #mean absolute value
    MAV = torch.mean(torch.abs(emg), dim=2)

    # standard deviation of fourier transform
    # from section 3.2 of https://www.sciencedirect.com/science/article/pii/S0925231220303283?via%3Dihub#sec0006
    emgFFT = torch.abs(torch.fft.fft(emg, dim=2))
    STD = torch.std(emgFFT, dim=2)

    # root mean squared
    RMS = torch.sqrt(torch.mean(emg ** 2, dim=2))

    # waveform length
    wl = torch.sum(torch.abs(emg[:, :, 1:] - emg[:, :, :-1]), dim=2)

    # mean frequency
    power_spectrum = torch.abs(torch.fft.fft(emg, dim=2)) ** 2
    sampling_frequency = 1000  # sampling frequency in Hz
    frequency = torch.fft.fftfreq(emg.shape[2], d=1/sampling_frequency)
    frequency = frequency.unsqueeze(0).unsqueeze(1)
    sum_product = torch.sum(power_spectrum * frequency, dim=2)
    total_sum = torch.sum(power_spectrum, dim=2)
    mean_freq = sum_product / total_sum
    print(mean_freq.size())

    # combine active features
    features = torch.cat((STD, SAV, SSC), dim=1)

    # z transform
    s = preprocessing.StandardScaler().fit(features)
    return torch.from_numpy(s.transform(features))

# combine features and labels
combined = torch.cat((extractFeatures(emg), labels), dim=1)
#leaveOut
combinedOut = torch.tensor(())
if leaveOut:
    combinedOut = torch.cat((extractFeatures(outEMG), outLabels), dim=1)

# split into training, validation, and test set
train, validation = model_selection.train_test_split(combined.numpy(), test_size=0.2)
test = torch.tensor(())
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
input_size = 128*numFeatures
hidden_sizes = [1024, 1024]
output_size = 8  # 8 gestures
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
run = wandb.init(name='CapgMyo', project='emg_benchmarking', entity='msoh')
wandb.config.lr = learn

num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

wandb.watch(model)

for epoch in range(num_epochs):
    model.train()
    train_acc = 0.0
    for batch_data in train_loader:
        batch_data = batch_data.to(device)

        data, labels = torch.split(batch_data, [128*numFeatures, 8], dim=1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)

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

            data, labels = torch.split(batch_data, [128*numFeatures, 8], dim=1)

            output = model(data)
            val_loss += criterion(output, labels).item()

            val_acc += np.mean(np.argmax(output.cpu().detach().numpy(), axis=1) == np.argmax(labels.cpu().detach().numpy(), axis=1))

    val_loss /= len(val_loader)
    train_acc /= len(train_loader)
    val_acc /= len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
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

        data, labels = torch.split(batch_data, [128*numFeatures, 8], dim=1)

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
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = np.arange(1, 9, 1),
                     columns = np.arange(1, 9, 1))
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, fmt=".3f")
plt.savefig('output.png')


