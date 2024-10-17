import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging as logg

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
    def load_data(self):
        
        # Logging the data
        logg.info("Data loaded successfully!")
        return self.data
    def explore_data(self):
        # Exploring the data
        print(self.data.head())
    def check_missing_values(self):
        # Checking for missing values
        missing_values = self.data.isnull().sum()
        # logging the missing values
        logg.info("Missing values: {}".format(missing_values))
        return missing_values
    def handle_missing_values(self):
        # Handle missing values by imputing with the mean of the column
        self.data = self.data.fillna(self.data.mean())
        return self.data

    def handle_outliers(self):
        # Handle outliers by capping the values at the 95th percentile
        self.data = self.data.apply(lambda x: x.clip(*x.quantile([0.05, 0.95])) if x.dtype == 'float64' else x)
        return self.data

    def normalize_data(self):
        # Normalize the data using min-max scaling
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        return self.data

    def preprocess_data(self):
        self.data = self.handle_missing_values()
        self.data = self.handle_outliers()
        self.data = self.normalize_data()
        return self.data