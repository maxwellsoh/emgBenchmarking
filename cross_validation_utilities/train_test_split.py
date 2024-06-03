from sklearn import model_selection
import numpy as np

def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
    force_regression=False, # TODO: tidy, but essentialyl always true because alwawys using the labels instead
): 
    if shuffle==False and not stratify is None:
        reshape_Y = False
        reshape_labels = False # only for regression

        X_train_set = arrays[0]
        Y_train_set = arrays[1]
        label_train_set = arrays[2]

        if train_size is None and test_size is None:
            train_size = 0.75

        assert (test_size is None) != (train_size is None), "Either test_size or train_size should be None"

        train_size = train_size or 1 - test_size

        # Convert stratify labels from one-hot-encoding to 1D array
        if stratify.shape[1] != 1:
            stratify = np.argmax(stratify, axis=1)
        if not force_regression and Y_train_set.shape[1] != 1:
            # if classification and Y_train_set is one hot encoded gestures (and not force)
            reshape_Y = True
            Y_train_set = np.argmax(Y_train_set, axis=1)
        
    
        unique, counts = np.unique(stratify, return_counts=True)
        # split data for each class
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        label_train = []
        label_test = []
        

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
            label_train_class = label_train_set[indices_train]

            # test set is the rest of the data
            indices_test = np.setdiff1d(indices, indices_train)
            X_test_class = X_train_set[indices_test]
            y_test_class = Y_train_set[indices_test]
            label_test_class = label_train_set[indices_test]

            X_train.append(X_train_class)
            X_test.append(X_test_class)
            y_train.append(y_train_class)
            y_test.append(y_test_class)
            label_train.append(label_train_class)
            label_test.append(label_test_class)

        X_train = np.concatenate(X_train)
        X_test = np.concatenate(X_test)
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)
       
        label_train = np.concatenate(label_train)
        label_test = np.concatenate(label_test)
        
        # for classification, convert Y_train_set gestures back to a one hot encoding
        if not force_regression and reshape_Y:
            y_train = np.eye(len(np.unique(y_train)))[y_train]
            y_test = np.eye(len(np.unique(y_test)))[y_test]
        
        # convert labels back to a one hot encoding
        label_train = np.eye(len(np.unique(label_train)))[label_train]
        label_test = np.eye(len(np.unique(label_test)))[label_test]
            
    else:
        X_train, X_test, y_train, y_test, label_train, label_test = model_selection.train_test_split(
            *arrays,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
     
    return X_train, X_test, y_train, y_test, label_train, label_test
  

