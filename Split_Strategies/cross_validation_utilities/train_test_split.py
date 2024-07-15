from sklearn import model_selection
import numpy as np

def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
    force_regression=False
): 
    """Splits data into training and testing sets. 

    If shuffle=False and stratify is not None, the function will split the data such that each class is represented in the training set according to the stratify array. Assumes stratify is always an array of labels.  
     
    If shuffle=True or stratify=None, the function will default to using the sklearn.model_selection.train_test_split function.

    Returns training and test sets for X, y, and labels. 

    Args:
        test_size: Proportion to take from test. Defaults to None.
        train_size: Proportion to take from train. Defaults to None.
        random_state: Argument passed to skl. Defaults to None.
        shuffle: Argument passed to skl. Defaults to True.
        stratify: Set used to stratify. Assumed to be labels.
        force_regression (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: Train and test sets for X, y, and labels.
    """

    if shuffle==False and stratify is not None:
        X_train_set = arrays[0]
        Y_train_set_og = arrays[1]

        if train_size is None and test_size is None:
            train_size = 0.75

        assert (test_size is None) != (train_size is None), "Either test_size or train_size should be None"

        train_size = train_size or 1 - test_size

        stratify_one_hot = False
        is_y_one_hot = False
        NUM_GESTURES = stratify.shape[1]

        # If labels are one-hot-encoded, convert to 1D array      
        if stratify.shape[1] != 1:
            stratify_one_hot = True
            stratify = np.argmax(stratify, axis=1)
        if not force_regression and Y_train_set_og.shape[1] != 1:
            is_y_one_hot = True
            Y_train_set = np.argmax(Y_train_set_og, axis=1)
        else:
            Y_train_set = Y_train_set_og

        label_train_set = stratify.clone()

        unique, counts = np.unique(stratify, return_counts=True) # (GESTURE, ITS WINDOWS)

        # split data for each class
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        label_train = []
        label_test = []

        # takes proportion of windows for each gesture  
        train_size_for_each_class = np.round(train_size * counts).astype(int)
        class_amount = dict(zip(unique, train_size_for_each_class)) # (GESTURE, NUMBER OF WINDOWS)

        for key in class_amount.keys():
            train_size_for_current_class = class_amount[key]
            # get indices for current class
            indices = np.where(stratify == key)[0]
            indices_train = indices[:train_size_for_current_class]
            # get data for current class
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
        
        # if input labels are one-hot-encoded, output labels are one-hot-encoded
        if not force_regression and is_y_one_hot:
            NUM_GESTURES = Y_train_set_og.shape[1] 
            y_train = np.eye(NUM_GESTURES)[y_train]
            y_test = np.eye(NUM_GESTURES)[y_test]
        if stratify_one_hot:
            label_train = np.eye(NUM_GESTURES)[label_train]
            label_test = np.eye(NUM_GESTURES)[label_test]
        
    else: # shuffle=True or stratify=None
        arrays = list(arrays) + [stratify]
        X_train, X_test, y_train, y_test, label_train, label_test = model_selection.train_test_split(
            *arrays,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

    # NOTE: always returning three makes it easier to add force_regression
    return X_train, X_test, y_train, y_test, label_train, label_test