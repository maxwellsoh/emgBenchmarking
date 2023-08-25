import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from sklearn import preprocessing, model_selection
from torch.utils.data import DataLoader, Dataset
from scipy.signal import butter,filtfilt,iirnotch
import h5py
import scipy
import gc

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

class Data(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# train, validation, test
def evaluate (learn, epochs, model, train_loader, val_loader, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn)

    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #run = wandb.init(name='CNN', project='emg_benchmarking', entity='msoh')
    #wandb.config.lr = learn
    #wandb.watch(model)

    for epoch in range(epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device).to(torch.float32)
            Y_batch = Y_batch.to(device).to(torch.float32)

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
                X_batch = X_batch.to(device).to(torch.float32)
                Y_batch = Y_batch.to(device).to(torch.float32)

                output = model(X_batch)
                val_loss += criterion(output, Y_batch).item()

                val_acc += np.mean(np.argmax(output.cpu().detach().numpy(), axis=1) == np.argmax(Y_batch.cpu().detach().numpy(), axis=1))

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
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
            X_batch = X_batch.to(device).to(torch.float32)
            Y_batch = Y_batch.to(device).to(torch.float32)

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
    return f"{test_acc:.4f}"
    '''
    cf_matrix = confusion_matrix(true, pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = np.arange(1, 19, 1),
                        columns = np.arange(1, 19, 1))
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, fmt=".3f")
    plt.savefig('output.png')
    '''

# removing excess rest windows - may want to apply to other gestures
def balance (restimulus):
    numZero = 0
    indices = []
    for x in range (len(restimulus)):
        L = torch.chunk(restimulus[x], 2, dim=1)
        if torch.equal(L[0], L[1]):
            if L[0][0][0] == 0:
                if (numZero < 550): # potentially modifiable
                    indices += [x]
                numZero += 1
            else:
                indices += [x]
    return indices

def filter(emg, freq):
    b, a = butter(N=1, Wn=120.0, btype='highpass', analog=False, fs=freq) # potentially modifiable
    if (freq == 4000): # no notch filter for JehanDataset; potentially modifiable
        return torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=-1).copy()) # filter on last axis vs first?
    
    emgButter = torch.from_numpy(np.flip(filtfilt(b, a, emg),axis=0).copy())
    b, a = iirnotch(w0=50.0, Q=0.0001, fs=freq) #potentially modifiable
    return torch.from_numpy(np.flip(filtfilt(b, a, emgButter),axis=0).copy())

# FC feature extraction; 5 other features implemented in FCNN_CapgMyo.py
def extractFeatures (emg):
    # sum of absolute values
    SAV = torch.sum(torch.abs(emg), dim=2)

    # standard deviation of fourier transform
    emgFFT = torch.abs(torch.fft.fft(emg, dim=2))
    STD = torch.std(emgFFT, dim=2)

    # combine active features
    features = torch.cat((STD, SAV), dim=1)

    # z transform
    s = preprocessing.StandardScaler().fit(features)
    return s.transform(features)


# FC NinaproDB2 specific helper functions
def NinaproDB2_getEMG (subject, wLen, step):
    freq = 2000.0 #Hz
    sub = str(subject+1)
    mat_data = scipy.io.loadmat('./NinaproDB2/DB2_s' + sub + '/S' + sub + '_E1_A1.mat')
    return torch.from_numpy(mat_data['emg']).unfold(dimension=0, size=int(wLen/1000.0*freq), step=int(step/1000.0*freq))

def NinaproDB2_getRestimulus (subject, wLen, step):
    freq = 2000.0 #Hz
    sub = str(subject+1)
    mat_data = scipy.io.loadmat('./NinaproDB2/DB2_s' + sub + '/S' + sub + '_E1_A1.mat')
    return torch.from_numpy(mat_data['restimulus']).unfold(dimension=0, size=int(wLen/1000.0*freq), step=int(step/1000.0*freq))

def FC_NinaproDB2(leftOut, learn, epochs, model, wLen, step):
    emg = []
    labels = []
    for i in range (40):
        emg += [extractFeatures(filter(NinaproDB2_getEMG(i, wLen, step)[balance(NinaproDB2_getRestimulus(i, wLen, step))]))]
        labels += [np.load("./NinaproDB2/label_" + str(i+1) + ".npy")]
    if leftOut == 0:
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(np.concatenate(emg), np.concatenate(labels), test_size=0.2)
        X_validation, X_test, Y_validation, Y_test = model_selection.train_test_split(X_validation, Y_validation, test_size=0.5)
    else:
        X_test = emg.pop(leftOut-1)
        Y_test = labels.pop(leftOut-1)
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(np.concatenate(emg), np.concatenate(labels), test_size=0.1)

    X_train = torch.from_numpy(X_train).to(torch.float32)
    Y_train = torch.from_numpy(Y_train).to(torch.float32)
    X_validation = torch.from_numpy(X_validation).to(torch.float32)
    Y_validation = torch.from_numpy(Y_validation).to(torch.float32)
    X_test = torch.from_numpy(X_test).to(torch.float32)
    Y_test = torch.from_numpy(Y_test).to(torch.float32)

    batch_size = 64 # potentially modifiable
    return evaluate(learn, epochs, model, DataLoader(list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True),
                    DataLoader(list(zip(X_validation, Y_validation)), batch_size=batch_size), DataLoader(list(zip(X_test, Y_test)), batch_size=batch_size))

# FC JehanDataset specific helper functions
def JehanDataset_getData(file, gesture, wLen, step):
    freq = 4000.0 #Hz
    data = filter(torch.from_numpy(np.array(file[gesture])).unfold(dimension=-1, size=int(wLen/1000*freq), step=int(step/1000*freq)))
    if (len(data) == 10):
        return torch.cat((data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]), axis=1).permute([1, 0, 2])
    elif (len(data) == 9):
        return torch.cat((data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]), axis=1).permute([1, 0, 2])
    else:
        return torch.cat((data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]), axis=1).permute([1, 0, 2])

def JehanDataset_getEMG(n, wLen, step):
    if (n<10):
        f = h5py.File('./Jehan_Dataset/p00' + str(n) +'/data_allchannels_initial.h5', 'r')
    else:
        f = h5py.File('./Jehan_Dataset/p0' + str(n) +'/data_allchannels_initial.h5', 'r')
    gestures = ["abduct_p1", "adduct_p1", "extend_p1", "grip_p1", "pronate_p1", "rest_p1", "supinate_p1", "tripod_p1", "wextend_p1", "wflex_p1"]
    return extractFeatures(torch.cat([(JehanDataset_getData(f, gestures[i], wLen, step)) for i in range(10)], axis=0))

def FC_JehanDataset(leftOut, learn, epochs, model, wLen, step):
    labels_8 = np.load("./Jehan_Dataset/labels_8.npy")
    labels_13 = np.load("./Jehan_Dataset/labels_13.npy")
    labels_18 = np.load("./Jehan_Dataset/labels_18.npy")
    labels_20 = np.load("./Jehan_Dataset/labels_20.npy")
    labels_21 = np.load("./Jehan_Dataset/labels_21.npy")
    labels_22 = np.load("./Jehan_Dataset/labels_22.npy")
    otherLabels = np.load("./Jehan_Dataset/labels.npy")
    participants = [8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22]
    emg = [(JehanDataset_getEMG(i, wLen, step)) for i in participants]
    labels = [labels_8, otherLabels, otherLabels, otherLabels, labels_13, otherLabels, otherLabels, 
                otherLabels, labels_18, otherLabels, labels_20, labels_21, labels_22]
    
    if leftOut == 0:
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(np.concatenate(emg), np.concatenate(labels), test_size=0.2)
        X_validation, X_test, Y_validation, Y_test = model_selection.train_test_split(X_validation, Y_validation, test_size=0.5)
    else:
        X_test = emg.pop(participants.index(leftOut))
        Y_test = labels.pop(participants.index(leftOut))
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(np.concatenate(emg), np.concatenate(labels), test_size=0.1)

    X_train = torch.from_numpy(X_train).to(torch.float32)
    Y_train = torch.from_numpy(Y_train).to(torch.float32)
    X_validation = torch.from_numpy(X_validation).to(torch.float32)
    Y_validation = torch.from_numpy(Y_validation).to(torch.float32)
    X_test = torch.from_numpy(X_test).to(torch.float32)
    Y_test = torch.from_numpy(Y_test).to(torch.float32)

    batch_size = 64 # potentially modifiable
    return evaluate(learn, epochs, model, DataLoader(list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True),
                    DataLoader(list(zip(X_validation, Y_validation)), batch_size=batch_size), DataLoader(list(zip(X_test, Y_test)), batch_size=batch_size))

def CNN_NinaproDB2(leftOut, learn, epochs, model):
    participants = range(40)
    labels = [np.load("./NinaproDB2/label_" + str(i+1) + ".npy") for i in range(40)]

    if leftOut == 0:
        combined_labels = np.concatenate(labels, axis=0, dtype=np.float16)
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(np.concatenate([(np.load("./NinaproDB2/rawImages_" + str(i+1) + ".npy").astype(np.float16))
                                                                                        for i in participants], axis=0, dtype=np.float16), combined_labels, test_size=0.2)
        X_validation, X_test, Y_validation, Y_test = model_selection.train_test_split(X_validation, Y_validation, test_size=0.5)
    else:
        Y_test = labels.pop(leftOut-1)
        combined_labels = np.concatenate(labels, axis=0, dtype=np.float16)
        del participants[leftOut-1]
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(np.concatenate([(np.load("./NinaproDB2/rawImages_" + str(i+1) + ".npy").astype(np.float16))
                                                                                        for i in participants], axis=0, dtype=np.float16), combined_labels, test_size=0.1)
        X_test = np.load("./NinaproDB2/rawImages_" + str(leftOut) + ".npy").astype(np.float16)

    X_train = torch.from_numpy(X_train).to(torch.float16)
    Y_train = torch.from_numpy(Y_train).to(torch.float16)
    X_validation = torch.from_numpy(X_validation).to(torch.float16)
    Y_validation = torch.from_numpy(Y_validation).to(torch.float16)
    X_test = torch.from_numpy(X_test).to(torch.float16)
    Y_test = torch.from_numpy(Y_test).to(torch.float16)

    batch_size = 64 # potentially modifiable
    return evaluate(learn, epochs, model, DataLoader(list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True),
                    DataLoader(list(zip(X_validation, Y_validation)), batch_size=batch_size), DataLoader(list(zip(X_test, Y_test)), batch_size=batch_size))


def CNN_JehanDataset(leftOut, learn, epochs, model):
    labels_8 = np.load("./Jehan_Dataset/labels_8.npy").astype(np.float16)
    labels_13 = np.load("./Jehan_Dataset/labels_13.npy").astype(np.float16)
    labels_18 = np.load("./Jehan_Dataset/labels_18.npy").astype(np.float16)
    labels_20 = np.load("./Jehan_Dataset/labels_20.npy").astype(np.float16)
    labels_21 = np.load("./Jehan_Dataset/labels_21.npy").astype(np.float16)
    labels_22 = np.load("./Jehan_Dataset/labels_22.npy").astype(np.float16)
    otherLabels = np.load("./Jehan_Dataset/labels.npy").astype(np.float16)
    participants = [8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22]
    labels = [labels_8, otherLabels, otherLabels, otherLabels, labels_13, otherLabels, otherLabels, 
                otherLabels, labels_18, otherLabels, labels_20, labels_21, labels_22]

    if leftOut == 0:
        combined_labels = np.concatenate(labels, axis=0, dtype=np.float16)
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(np.concatenate([(np.load("./Jehan_Dataset/rawImages_" + str(i) + ".npy").astype(np.float16))
                                                                                        for i in participants], axis=0, dtype=np.float16), combined_labels, test_size=0.2)
        X_validation, X_test, Y_validation, Y_test = model_selection.train_test_split(X_validation, Y_validation, test_size=0.5)
    else:
        Y_test = labels.pop(participants.index(leftOut))
        combined_labels = np.concatenate(labels, axis=0, dtype=np.float16)
        del participants[participants.index(leftOut)]
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(np.concatenate([(np.load("./Jehan_Dataset/rawImages_" + str(i) + ".npy").astype(np.float16))
                                                                                        for i in participants], axis=0, dtype=np.float16), combined_labels, test_size=0.1)
        X_test = np.load("./Jehan_Dataset/rawImages_" + str(leftOut) + ".npy").astype(np.float16)

    X_train = torch.from_numpy(X_train).to(torch.float16)
    Y_train = torch.from_numpy(Y_train).to(torch.float16)
    X_validation = torch.from_numpy(X_validation).to(torch.float16)
    Y_validation = torch.from_numpy(Y_validation).to(torch.float16)
    X_test = torch.from_numpy(X_test).to(torch.float16)
    Y_test = torch.from_numpy(Y_test).to(torch.float16)

    batch_size = 64 # potentially modifiable
    return evaluate(learn, epochs, model, DataLoader(list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True),
                    DataLoader(list(zip(X_validation, Y_validation)), batch_size=batch_size), DataLoader(list(zip(X_test, Y_test)), batch_size=batch_size))


# input format:
# dataset (e.g. "NinaproDB2")
# leaveOut (i.e. "T" or "F")
# learning rate (e.g. "0.0001")
# num of epochs (e.g. "50")
# model [FC];[layers];[window size];[step size] (e.g. "FC;512x512;250;50")
#       OR [CNN];[truncation (0<=i<=3 for num of layers to truncate)];
#           [freezing (0<=i<=4 for num of layers to freez)];[fc] (e.g. "CNN;2;1;512x512")
def process_input(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        testList = [[]]
        num = 0
        for line in lines:
            testList[len(testList)-1] += [line.strip()]
            num += 1
            if num > 4:
                num = 0
                testList += []
        results = []
        for test in testList:
            # maybe implement default values for missing inputs
            results += [runTest(test)]
    return results


def runTest(testParams):
    modelSpecs = testParams[4].split(";")
    
    # FC tests
    if (modelSpecs[0] == "FC"):
        numFeatures = 2 # potentially modifiable
        dropout_rate = 0.5 # potentially modifiable
        hidden_sizes = list(map(int, modelSpecs[1].split("x")))

        # FC JehanDataset tests
        if testParams[0] == "JehanDataset":
            input_size = 64*numFeatures
            output_size = 10
            model = FullyConnectedNet(input_size, hidden_sizes, output_size, dropout_rate)

            if testParams[1] == "T":
                trials = []
                for i in [8,9,11,12,13,15,16,17,18,19,20,21,22]: #LOSO
                    trials += FC_JehanDataset(i, float(testParams[2]), int(testParams[3]), model, int(modelSpecs[2]), int(modelSpecs[3]))
                return str(trials)
            else: #regular
                return FC_JehanDataset(0, float(testParams[2]), int(testParams[3]), model, int(modelSpecs[2]), int(modelSpecs[3]))
        
        # FC NinaproDB2 tests
        elif testParams[0] == "NinaproDB2":
            input_size = 12*numFeatures
            output_size = 18
            model = FullyConnectedNet(input_size, hidden_sizes, output_size, dropout_rate)

            if testParams[1] == "T":
                trials = []
                for i in range(40): #LOSO
                    trials += FC_NinaproDB2(i+1, float(testParams[2]), int(testParams[3]), model, int(modelSpecs[2]), int(modelSpecs[3]))
                return str(trials)
            else: #regular
                return FC_NinaproDB2(0, float(testParams[2]), int(testParams[3]), model, int(modelSpecs[2]), int(modelSpecs[3]))
        else:
            return "Dataset Not Found"
    
    # CNN tests
    elif (modelSpecs[0] == "CNN"):
        if (int(modelSpecs[1]) < 0 or int(modelSpecs[1]) > 3):
            return "Invalid Truncation"
        if (int(modelSpecs[2]) < 0 or int(modelSpecs[2]) > 4):
            return "Invalid Freezing"
        
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # truncation starts from the end of ResNet50
        model = nn.Sequential(*list(model.children())[:-(2 + int(modelSpecs[1]))])
        prev = model[-1][-1].conv3.out_channels
        model.add_module('avgpool', torch.nn.AdaptiveAvgPool2d(1))
        model.add_module('flatten', torch.nn.Flatten())
        
        dropout = 0.5 # potentially modifiable
        count = 0
        for layer in (modelSpecs[3].split("x")):
            count += 1
            model.add_module('fc' + str(count), torch.nn.Linear(prev, int(layer)))
            model.add_module('relu' + str(count), torch.nn.ReLU())
            model.add_module('dropout' + str(count), torch.nn.Dropout(dropout))
            prev = int(layer)

        # freezing starts from the beginning of ResNet50
        thresholds = [0, 33, 72, 129, 159]
        num = 0
        for name, param in model.named_parameters():
            num += 1
            if (num > thresholds[int(modelSpecs[2])]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # CNN JehanDataset tests
        if testParams[0] == "JehanDataset":
            model.add_module('fc' + str(count + 1), torch.nn.Linear(prev, 10))
            model.add_module('softmax', torch.nn.Softmax(dim=1))

            if testParams[1] == "T":
                trials = []
                for i in [8,9,11,12,13,15,16,17,18,19,20,21,22]: #LOSO
                    trials += CNN_JehanDataset(i, float(testParams[2]), int(testParams[3]), model)
                return str(trials)
            else: #regular
                return CNN_JehanDataset(0, float(testParams[2]), int(testParams[3]), model)
        
        # CNN NinaproDB2 tests
        elif testParams[0] == "NinaproDB2":
            model.add_module('fc' + str(count + 1), torch.nn.Linear(prev, 18))
            model.add_module('softmax', torch.nn.Softmax(dim=1))

            if testParams[1] == "T":
                trials = []
                for i in range(40): #LOSO
                    trials += CNN_NinaproDB2(i+1, float(testParams[2]), int(testParams[3]), model)
                return str(trials)
            else: #regular
                return CNN_NinaproDB2(0, float(testParams[2]), int(testParams[3]), model)
        else:
            return "Dataset Not Found"
    else:
        return "Model Not Found"

def output(output, file_path):
    with open(file_path, "w") as f:
        for s in range(len(output)):
            f.write("Test " + str(s) + " Accuracy: " + output[s])
            f.write("\n")
        #f.write(" ".join(output))
    return

if __name__ == "__main__":
    results = process_input("tests.txt")
    output(results, "output.txt")
    pass