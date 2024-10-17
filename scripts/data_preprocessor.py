import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logg = logging.getLogger(__name__)
class DataPreprocessor:
    def __init__(self, file_path1, file_path2, file_path3):
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.file_path3 = file_path3
        self.data = None
        self.data1 = None
        self.data2 = None

    def load_data(self):
        """Load data from the provided file paths."""
        try:
            self.data = pd.read_csv(self.file_path1)
            self.data1 = pd.read_csv(self.file_path2)
            self.data2 = pd.read_csv(self.file_path3)
            
            # Logging the successful data loading
            logg.info("Data loaded successfully!")
            
            return self.data, self.data1, self.data2
        
        except Exception as e:
            logg.error(f"An error occurred while loading data: {e}")
            return None, None, None
    def explore_data(self):
        """Exploring the first few rows of data."""
        logg.info("Exploring data:")
        print(self.data.head())

    def check_missing_values(self):
        """Checking for missing values in the dataset."""
        missing_values = self.data.isnull().sum()
        logg.info(f"Missing values: \n{missing_values}")
        return missing_values

    def handle_missing_values(self):
        """Handle missing values by imputing with the mean for numerical columns only."""
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[num_cols] = self.data[num_cols].fillna(self.data[num_cols].mean())
        logg.info("Missing values handled successfully!")
        return self.data

    def handle_outliers(self):
        """Handle outliers by capping the values at the 5th and 95th percentiles for numerical data."""
        num_cols = self.data.select_dtypes(include=[np.float64]).columns
        self.data[num_cols] = self.data[num_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))
        logg.info("Outliers handled successfully!")
        return self.data

    def normalize_data(self, dataset):
        """Normalize numerical columns of the specified dataset using min-max scaling."""
        num_cols = dataset.select_dtypes(include=[np.number]).columns
        dataset[num_cols] = (dataset[num_cols] - dataset[num_cols].min()) / (dataset[num_cols].max() - dataset[num_cols].min())
        logg.info("Dataset normalized successfully!")
        return dataset
