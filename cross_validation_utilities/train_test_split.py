from sklearn import model_selection
import numpy as np

def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
    force_regression=False, # NOTE: assumes gesture classification if false 
): 
    if shuffle==False and not stratify is None:
        reshape_Y = False
        reshape_labels = False # only for regression

        X_train_set = arrays[0]
        Y_train_set = arrays[1]

        if train_size is None and test_size is None:
            train_size = 0.75

        assert (test_size is None) != (train_size is None), "Either test_size or train_size should be None"

        train_size = train_size or 1 - test_size

        # Convert from one-hot-encoding to 1D array
        if stratify.shape[1] != 1:
            # assumes stratify is always labels
            stratify = np.argmax(stratify, axis=1)
        if not force_regression and Y_train_set.shape[1] != 1:
            # if classification and Y_train_set is one hot encoded gestures (and not force)
            reshape_Y = True
            Y_train_set = np.argmax(Y_train_set, axis=1)
        if force_regression:
            reshape_labels = True
            # assumes that for force_regressions stratify is the gesture labels
            labels_train_set = stratify.clone()
        
    
        unique, counts = np.unique(stratify, return_counts=True)
        # split data for each class
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        labels_train = []
        labels_test = []
        

        train_size_for_each_class = np.round(train_size * counts).astype(int)
        class_amount = dict(zip(unique, train_size_for_each_class))


        for key in class_amount.keys():
            train_size_for_current_class = class_amount[key]
            # get indices for current class
            indices = np.where(stratify == key)[0]
            indices_train = indices[:train_size_for_current_class]
            # get all data for current class
            X_train_class = X_train_set[indices_train]
            y_train_class = Y_train_set[indices_train] 
            if force_regression:
                labels_train_class = labels_train_set[indices_train]

            # test set is the rest of the data
            indices_test = np.setdiff1d(indices, indices_train)
            X_test_class = X_train_set[indices_test]
            y_test_class = Y_train_set[indices_test]
            if force_regression:
                labels_test_class = labels_train_set[indices_test]

            X_train.append(X_train_class)
            X_test.append(X_test_class)
            y_train.append(y_train_class)
            y_test.append(y_test_class)
            if force_regression:
                labels_train.append(labels_train_class)
                labels_test.append(labels_test_class)

        X_train = np.concatenate(X_train)
        X_test = np.concatenate(X_test)
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)
        if force_regression:
            labels_train = np.concatenate(labels_train)
            labels_test = np.concatenate(labels_test)
        
        # for classification, convert Y_train_set gestures back to a one hot encoding
        if not force_regression and reshape_Y:
            y_train = np.eye(len(np.unique(y_train)))[y_train]
            y_test = np.eye(len(np.unique(y_test)))[y_test]
        
        # for regression, convert labels back to a one hot encoding
        if force_regression and reshape_labels:
            labels_train = np.eye(len(np.unique(labels_train)))[labels_train]
            labels_test = np.eye(len(np.unique(labels_test)))[labels_test]
            
    else:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            *arrays,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        
    if force_regression:
        assert y_test.shape[0] == labels_test.shape[0], "y_test and labels_test should have the same size"
        return X_train, X_test, y_train, y_test, labels_train, labels_test
    else:
        return X_train, X_test, y_train, y_test

