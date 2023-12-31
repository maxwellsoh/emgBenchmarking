import unittest
from utils_NinaproDB5 import getEMG, getRestim
import utils_NinaproDB5 as ut_NDB5

class TestUtils(unittest.TestCase):
    def test_getRestim(self):
        # Test case 1
        restim = getRestim(1)
        self.assertIsNotNone(restim)
        self.assertGreater(restim.shape[0], 1)
        self.assertEqual(restim.shape[1], 1)  # Update with the expected shape

        # Test case 2
        restim = getRestim(2)
        self.assertIsNotNone(restim)
        self.assertGreater(restim.shape[0], 1)
        self.assertEqual(restim.shape[1], 1) 
        # Add more test cases as needed
        
    def test_getEMG(self):
        # Test case 1
        emg = getEMG(1)
        self.assertIsNotNone(emg)
        self.assertGreater(emg.shape[0], 1)
        self.assertEqual(emg.shape[1], ut_NDB5.numElectrodes)
        self.assertEqual(emg.shape[2], ut_NDB5.wLenTimesteps)

        # Test case 2
        emg = getEMG(2)
        self.assertIsNotNone(emg)
        self.assertGreater(emg.shape[0], 1)
        self.assertEqual(emg.shape[1], ut_NDB5.numElectrodes)
        self.assertEqual(emg.shape[2], ut_NDB5.wLenTimesteps)

        # Add more test cases as needed

    

if __name__ == '__main__':
    unittest.main()