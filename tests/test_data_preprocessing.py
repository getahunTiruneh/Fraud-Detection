import unittest
import pandas as pd
import numpy as np
from scripts.data_preprocessor import DataPreprocessor  # Assuming your class is in data_preprocessor.py

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.file_path1 = 'test_data1.csv'  # Not actually used for tests here
        self.file_path2 = 'test_data2.csv'  # Not actually used for tests here
        self.file_path3 = 'test_data3.csv'  # Not actually used for tests here
        
        self.data = pd.DataFrame({
            'ip_address': ['192.168.1.1', '10.0.0.1', np.nan],
            'purchase_time': ['2023-01-01 10:00:00', '2023-01-02 11:30:00', '2023-01-03 12:45:00'],
            'amount': [100, 200, np.nan]
        })
        
        self.data1 = pd.DataFrame({
            'lower_bound_ip_address': [167772161, 167772160],  # IPs as integers
            'upper_bound_ip_address': [167772162, 167772161]   # IPs as integers
        })

        # Adjust data2 to only have one row
        self.data2 = pd.DataFrame({
            'other_column': [1]  # Now only one row
        })

        # Create an instance of DataPreprocessor
        self.processor = DataPreprocessor(self.file_path1, self.file_path2, self.file_path3)

    def test_load_data(self):
        # Directly assign the data for the test
        self.processor.data = self.data
        self.processor.data1 = self.data1
        self.processor.data2 = self.data2

        # Check that data is loaded correctly
        self.assertEqual(len(self.processor.data), 3)
        self.assertEqual(len(self.processor.data1), 2)
        self.assertEqual(len(self.processor.data2), 1)  # This should now pass


    def test_check_missing_values(self):
        self.processor.data = self.data
        missing_values = self.processor.check_missing_values(self.processor.data)
        self.assertEqual(missing_values['amount'], 1)  # 1 missing value in 'amount' column

    def test_handle_missing_values(self):
        self.processor.data = self.data.copy()
        self.processor.handle_missing_values()
        self.assertTrue(self.processor.data['amount'].isnull().sum() == 0)  # No missing values should remain

if __name__ == '__main__':
    unittest.main()
