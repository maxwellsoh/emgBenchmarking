from sklearn import model_selection
import numpy as np

def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
): 
    if shuffle==False and stratify is not None:
        X_train_set = arrays[0]
        Y_train_set = arrays[1]

        if train_size is None and test_size is None:
            train_size = 0.75
        # assert only one of test_size or train_size is not None
        assert (test_size is None) != (train_size is None), "Either test_size or train_size should be None"

        train_size = train_size or 1 - test_size

        # find amount of data for each class to split
        if stratify.shape[1] != 1:
            # try to change into 1D array if hot one encoding
            stratify = np.argmax(stratify, axis=1)

        unique, counts = np.unique(stratify, return_counts=True)
        # split data for each class
        X_train = []
        X_test = []
        y_train = []
        y_test = []

        train_size_for_each_class = np.round(train_size * counts).astype(int)

        class_amount = dict(zip(unique, train_size_for_each_class))


        for key in class_amount.keys():
            train_size_for_current_class = class_amount[key]
            # get indices for current class
            indices = np.where(stratify == key)[0]
            indices_train = indices[:train_size_for_current_class]
            # get all data for current class
            X_train_class = X_train_set[indices_train]
            y_train_class = stratify[indices_train]
            # test set is the rest of the data
            indices_test = np.setdiff1d(indices, indices_train)
            X_test_class = X_train_set[indices_test]
            y_test_class = stratify[indices_test]

            X_train.append(X_train_class)
            X_test.append(X_test_class)
            y_train.append(y_train_class)
            y_test.append(y_test_class)

        X_train = np.concatenate(X_train)
        X_test = np.concatenate(X_test)
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)
        
        # if Y_train_set is one-hot-encoded, turn results into one-hot-encoded
        if Y_train_set.shape[1] != 1:
            y_train = np.eye(Y_train_set.shape[1])[y_train]
            y_test = np.eye(Y_train_set.shape[1])[y_test]
        
    else: # shuffle=True or stratify=None
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            *arrays,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

    return X_train, X_test, y_train, y_test

