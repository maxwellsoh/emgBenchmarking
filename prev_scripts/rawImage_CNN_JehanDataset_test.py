import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

leaveOut = True

labels_8 = np.load("./Jehan_Dataset/labels_8.npy").astype(np.float16)
labels_13 = np.load("./Jehan_Dataset/labels_13.npy").astype(np.float16)
labels_18 = np.load("./Jehan_Dataset/labels_18.npy").astype(np.float16)
labels_20 = np.load("./Jehan_Dataset/labels_20.npy").astype(np.float16)
labels_21 = np.load("./Jehan_Dataset/labels_21.npy").astype(np.float16)
labels_22 = np.load("./Jehan_Dataset/labels_22.npy").astype(np.float16)
otherLabels = np.load("./Jehan_Dataset/labels.npy").astype(np.float16)
# for 9, 11, 12, 15, 16, 17, 19

'''
combined_labels = np.concatenate((labels_8, otherLabels, otherLabels, otherLabels, labels_13, otherLabels, otherLabels, otherLabels, labels_18,
                                  otherLabels, labels_20, labels_21, labels_22), axis=0, dtype=np.float16)
'''
combined_labels = np.concatenate((labels_8, otherLabels, otherLabels, otherLabels, labels_13, otherLabels, otherLabels, otherLabels,
                                  labels_18, otherLabels, labels_20, labels_21), axis=0, dtype=np.float16)

if leaveOut:
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(np.concatenate((np.load("./Jehan_Dataset/rawImages_8.npy").astype(np.float16),
                       np.load("./Jehan_Dataset/rawImages_9.npy").astype(np.float16), np.load("./Jehan_Dataset/rawImages_11.npy").astype(np.float16),
                        np.load("./Jehan_Dataset/rawImages_12.npy").astype(np.float16), np.load("./Jehan_Dataset/rawImages_13.npy").astype(np.float16),
                        np.load("./Jehan_Dataset/rawImages_15.npy").astype(np.float16), np.load("./Jehan_Dataset/rawImages_16.npy").astype(np.float16),
                        np.load("./Jehan_Dataset/rawImages_17.npy").astype(np.float16), np.load("./Jehan_Dataset/rawImages_18.npy").astype(np.float16),
                        np.load("./Jehan_Dataset/rawImages_19.npy").astype(np.float16), np.load("./Jehan_Dataset/rawImages_20.npy").astype(np.float16),
                        np.load("./Jehan_Dataset/rawImages_21.npy").astype(np.float16)), axis=0, dtype=np.float16), combined_labels, test_size=0.1)
    X_test = np.load("./Jehan_Dataset/rawImages_22.npy").astype(np.float16)
    Y_test = labels_22
else:
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(np.concatenate((np.load("./Jehan_Dataset/rawImages_8.npy").astype(np.float16), np.load("./Jehan_Dataset/rawImages_9.npy").astype(np.float16),
                       np.load("./Jehan_Dataset/rawImages_11.npy").astype(np.float16), np.load("./Jehan_Dataset/rawImages_12.npy").astype(np.float16),
                        np.load("./Jehan_Dataset/rawImages_13.npy").astype(np.float16), np.load("./Jehan_Dataset/rawImages_15.npy").astype(np.float16),
                        np.load("./Jehan_Dataset/rawImages_16.npy").astype(np.float16), np.load("./Jehan_Dataset/rawImages_17.npy").astype(np.float16),
                        np.load("./Jehan_Dataset/rawImages_18.npy").astype(np.float16), np.load("./Jehan_Dataset/rawImages_19.npy").astype(np.float16),
                        np.load("./Jehan_Dataset/rawImages_20.npy").astype(np.float16), np.load("./Jehan_Dataset/rawImages_21.npy").astype(np.float16),
                        np.load("./Jehan_Dataset/rawImages_22.npy").astype(np.float16)), axis=0, dtype=np.float16), combined_labels, test_size=0.2)
    X_validation, X_test, Y_validation, Y_test = model_selection.train_test_split(X_validation, Y_validation, test_size=0.5)

X_train = torch.from_numpy(X_train).to(torch.float16)
Y_train = torch.from_numpy(Y_train).to(torch.float16)
X_validation = torch.from_numpy(X_validation).to(torch.float16)
Y_validation = torch.from_numpy(Y_validation).to(torch.float16)
X_test = torch.from_numpy(X_test).to(torch.float16)
Y_test = torch.from_numpy(Y_test).to(torch.float16)
#print(X_train.size())
#print(X_validation.size())
#print(X_test.size())

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
criterion = nn.CrossEntropyLoss()
learn = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learn)

# Training loop
import gc
gc.collect()
torch.cuda.empty_cache()

#run = wandb.init(name='CNN', project='emg_benchmarking', entity='msoh')
#wandb.config.lr = learn

num_epochs = 10
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
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = np.arange(1, 11, 1),
                     columns = np.arange(1, 11, 1))
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, fmt=".3f")
plt.savefig('output.png')