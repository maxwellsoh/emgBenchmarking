import unittest
from utils_FlexWearHD import getOnlineUnlabeledData

class TestUtils(unittest.TestCase):
    def test_getOnlineUnlabeledData(self):
        # Test case 1
        subject_number = 1
        data = getOnlineUnlabeledData(subject_number)
        self.assertIsNotNone(data)
        self.assertGreater(data.shape[0], 0)
        self.assertGreater(data.shape[1], 0)
        self.assertGreater(data.shape[2], 0)

        # Test case 2
        subject_number = 2
        data = getOnlineUnlabeledData(subject_number)
        self.assertIsNotNone(data)
        self.assertGreater(data.shape[0], 0)
        self.assertGreater(data.shape[1], 0)
        self.assertGreater(data.shape[2], 0)

        # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()