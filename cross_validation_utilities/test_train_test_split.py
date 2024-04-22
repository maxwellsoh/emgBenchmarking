import unittest
from train_test_split import train_test_split
import numpy as np

class TestTrainTestSplit(unittest.TestCase):
    def test_train_test_split_shuffle_false(self):
        # Test case 1
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 0, 0, 1, 1, 1])
        stratify = np.array([[1, 0], [1, 0], [1,0], [0, 1], [0, 1], [0, 1]])
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify, shuffle=False)
        self.assertEqual(X_train.shape, (4, 2))
        self.assertEqual(X_test.shape, (2, 2))
        self.assertEqual(y_train.shape, (4,))
        self.assertEqual(y_test.shape, (2,))

        # Test case 2
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
        y = np.array([0, 0, 0, 1, 1, 1])
        stratify = np.array([[1, 0], [1, 0], [1,0], [0, 1], [0, 1], [0, 1]])
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify, shuffle=False)
        self.assertEqual(X_train.shape, (4, 3))
        self.assertEqual(X_test.shape, (2, 3))
        self.assertEqual(y_train.shape, (4,))
        self.assertEqual(y_test.shape, (2,))

    def test_train_test_split_shuffle_true(self):
        # Test case 1
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
        self.assertEqual(X_train.shape, (3, 2))
        self.assertEqual(X_test.shape, (1, 2))
        self.assertEqual(y_train.shape, (3,))
        self.assertEqual(y_test.shape, (1,))

        # Test case 2
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 0, 1, 1])
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
        self.assertEqual(X_train.shape, (3, 3))
        self.assertEqual(X_test.shape, (1, 3))
        self.assertEqual(y_train.shape, (3,))
        self.assertEqual(y_test.shape, (1,))

if __name__ == '__main__':
    unittest.main()