import torch
#import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
#from sklearn import preprocessing, model_selection
from sklearn import model_selection
#from PyEMD import EMD
#import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

leaveOut = False

if leaveOut:
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(np.concatenate((np.load("./NinaproDB2/rawImages_2.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_3.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_4.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_5.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_6.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_7.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_8.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_1.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_10.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_11.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_12.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_9.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_14.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_15.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_16.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_13.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_18.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_19.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_20.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_17.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_22.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_23.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_24.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_21.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_26.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_27.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_28.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_25.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_30.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_31.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_32.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_29.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_34.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_35.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_36.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_33.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_38.npy").astype(np.float16),
                       np.load("./NinaproDB2/rawImages_39.npy").astype(np.float16), np.load("./NinaproDB2/rawImages_40.npy").astype(np.float16)), axis=0, dtype=np.float16),
                       np.concatenate((np.load("./NinaproDB2/label_2.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_3.npy").astype(np.float16), np.load("./NinaproDB2/label_4.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_5.npy").astype(np.float16), np.load("./NinaproDB2/label_6.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_7.npy").astype(np.float16), np.load("./NinaproDB2/label_8.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_1.npy").astype(np.float16), np.load("./NinaproDB2/label_10.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_11.npy").astype(np.float16), np.load("./NinaproDB2/label_12.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_9.npy").astype(np.float16), np.load("./NinaproDB2/label_14.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_15.npy").astype(np.float16), np.load("./NinaproDB2/label_16.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_13.npy").astype(np.float16), np.load("./NinaproDB2/label_18.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_19.npy").astype(np.float16), np.load("./NinaproDB2/label_20.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_17.npy").astype(np.float16), np.load("./NinaproDB2/label_22.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_23.npy").astype(np.float16), np.load("./NinaproDB2/label_24.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_21.npy").astype(np.float16), np.load("./NinaproDB2/label_26.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_27.npy").astype(np.float16), np.load("./NinaproDB2/label_28.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_25.npy").astype(np.float16), np.load("./NinaproDB2/label_30.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_31.npy").astype(np.float16), np.load("./NinaproDB2/label_32.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_29.npy").astype(np.float16), np.load("./NinaproDB2/label_34.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_35.npy").astype(np.float16), np.load("./NinaproDB2/label_36.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_33.npy").astype(np.float16), np.load("./NinaproDB2/label_38.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_39.npy").astype(np.float16), np.load("./NinaproDB2/label_40.npy").astype(np.float16)), axis=0, dtype=np.float16), test_size=0.1)
    X_test = np.load("./NinaproDB2/rawImages_37.npy").astype(np.float16)
    Y_test = np.load("./NinaproDB2/label_37.npy").astype(np.float16)
    X_train = torch.from_numpy(X_train).to(torch.float16)
    Y_train = torch.from_numpy(Y_train).to(torch.float16)
    X_validation = torch.from_numpy(X_validation).to(torch.float16)
    Y_validation = torch.from_numpy(Y_validation).to(torch.float16)
    X_test = torch.from_numpy(X_test).to(torch.float16)
    Y_test = torch.from_numpy(Y_test).to(torch.float16)
else:
    labels = np.concatenate((np.load("./NinaproDB2/label_1.npy").astype(np.float16), np.load("./NinaproDB2/label_2.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_3.npy").astype(np.float16), np.load("./NinaproDB2/label_4.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_5.npy").astype(np.float16), np.load("./NinaproDB2/label_6.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_7.npy").astype(np.float16), np.load("./NinaproDB2/label_8.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_9.npy").astype(np.float16), np.load("./NinaproDB2/label_10.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_11.npy").astype(np.float16), np.load("./NinaproDB2/label_12.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_13.npy").astype(np.float16), np.load("./NinaproDB2/label_14.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_15.npy").astype(np.float16), np.load("./NinaproDB2/label_16.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_17.npy").astype(np.float16), np.load("./NinaproDB2/label_18.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_19.npy").astype(np.float16), np.load("./NinaproDB2/label_20.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_21.npy").astype(np.float16), np.load("./NinaproDB2/label_22.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_23.npy").astype(np.float16), np.load("./NinaproDB2/label_24.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_25.npy").astype(np.float16), np.load("./NinaproDB2/label_26.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_27.npy").astype(np.float16), np.load("./NinaproDB2/label_28.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_29.npy").astype(np.float16), np.load("./NinaproDB2/label_30.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_31.npy").astype(np.float16), np.load("./NinaproDB2/label_32.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_33.npy").astype(np.float16), np.load("./NinaproDB2/label_34.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_35.npy").astype(np.float16), np.load("./NinaproDB2/label_36.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_37.npy").astype(np.float16), np.load("./NinaproDB2/label_38.npy").astype(np.float16),
                                  np.load("./NinaproDB2/label_39.npy").astype(np.float16), np.load("./NinaproDB2/label_40.npy").astype(np.float16)), axis=0, dtype=np.float16)
    train_indices, val_indices, test_indices = torch.utils.data.random_split(range(len(labels)), [0.8, 0.1, 0.1])
    Y_train = torch.from_numpy(labels[train_indices]).to(torch.float16)
    Y_validation = torch.from_numpy(labels[val_indices]).to(torch.float16)
    Y_test = torch.from_numpy(labels[test_indices]).to(torch.float16)
    del labels
    torch.set_default_tensor_type(torch.HalfTensor)
    data = torch.cat(((torch.from_numpy(np.load("./NinaproDB2/rawImages_1.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_2.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_3.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_4.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_5.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_6.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_7.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_8.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_9.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_10.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_11.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_12.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_13.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_14.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_15.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_16.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_17.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_18.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_19.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_20.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_21.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_22.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_23.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_24.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_25.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_26.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_27.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_28.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_29.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_30.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_31.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_32.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_33.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_34.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_35.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_36.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_37.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_38.npy")).to(torch.float16)),
                       (torch.from_numpy(np.load("./NinaproDB2/rawImages_39.npy")).to(torch.float16)), (torch.from_numpy(np.load("./NinaproDB2/rawImages_40.npy")).to(torch.float16))), dim=0).to(torch.float16)
    torch.set_default_tensor_type(torch.FloatTensor)
    X_train = data[train_indices].to(torch.float16)
    X_validation = data[val_indices].to(torch.float16)
    X_test = data[test_indices].to(torch.float16)
    del data



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

# loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
learn = 0.0001
#learn = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=learn)

import gc
gc.collect()
torch.cuda.empty_cache()

#run = wandb.init(name='CNN', project='emg_benchmarking', entity='msoh')
#wandb.config.lr = learn

num_epochs = 75
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

#wandb.watch(model)

for epoch in range(num_epochs):
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

cf_matrix = confusion_matrix(true, pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = np.arange(1, 19, 1),
                     columns = np.arange(1, 19, 1))
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, fmt=".3f")
plt.savefig('output.png')