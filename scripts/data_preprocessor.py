import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import MinMaxScaler
import socket
import struct

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logg = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, file_path1, file_path2, file_path3):
        """
        Initialize the DataPreprocessor with file paths for three datasets.
        """
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.file_path3 = file_path3
        self.data = None
        self.data1 = None
        self.data2 = None

    def load_data(self):
        """
        Load data from the provided file paths into Pandas DataFrames.
        """
        try:
            self.data = pd.read_csv(self.file_path1)
            self.data1 = pd.read_csv(self.file_path2)
            self.data2 = pd.read_csv(self.file_path3)
            
            logg.info("Data loaded successfully!")
            return self.data, self.data1, self.data2
        
        except Exception as e:
            logg.error(f"An error occurred while loading data: {e}")
            return None, None, None

    def explore_data(self):
        """
        Display the first few rows of the main dataset for exploration purposes.
        """
        logg.info("Exploring data:")
        logg.info(f"\n{self.data.head()}")

    def check_missing_values(self, data):
        """
        Check for missing values in the provided dataset.
        Parameters:
        data (pd.DataFrame): The input DataFrame to check for missing values.
        Returns:
        pd.Series: A Series with the count of missing values for each column.
        """
        missing_values = data.isnull().sum()
        logg.info(f"Missing values: \n{missing_values}")
        return missing_values

    def handle_missing_values(self):
        """
        Handle missing values in numerical columns by imputing with the column mean.
        """
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        if self.data[num_cols].isnull().sum().any():
            self.data[num_cols] = self.data[num_cols].fillna(self.data[num_cols].mean())
            logg.info("Missing values in numerical columns handled with mean imputation.")
        else:
            logg.info("No missing values found in numerical columns.")
        return self.data

    def normalize_data(self, data, columns):
        """
        Normalize specified numerical columns of the dataset using Min-Max scaling.

        Parameters:
        data (pd.DataFrame): The input DataFrame containing numerical data.
        columns (list): A list of numerical column names to normalize.

        Returns:
        pd.DataFrame: DataFrame with normalized numerical columns.
        """
        logg.info("Normalizing data...")
        # Check if specified columns are in the DataFrame
        if all(col in data.columns for col in columns):
            scaler = MinMaxScaler()
            # Normalize the specified columns
            data[columns] = scaler.fit_transform(data[columns])
            logg.info("Dataset normalized successfully using Min-Max scaling.")
            return data
        else:
            missing_cols = [col for col in columns if col not in data.columns]
            logg.error(f"Missing columns for normalization: {missing_cols}")
            return data  # Return unchanged data if specified columns are missing

    def ip_to_int(self, ip):
        """
        Convert an IP address to its integer representation.
        """
        try:
            return struct.unpack("!I", socket.inet_aton(ip))[0]
        except socket.error:
            logg.warning(f"Invalid IP address encountered: {ip}")
            return None

    def merge_datasets(self):
        """
        Merge the fraud dataset and IP address dataset based on IP ranges.
        Converts IP addresses to integer format and merges on IP bounds.
        """
        logg.info("Converting IP addresses to integer format...")
        
        # Convert IPs in fraud_data to integer format
        self.data['ip_int'] = self.data['ip_address'].apply(lambda x: self.ip_to_int(str(int(x))) if not pd.isna(x) else None)

        # Drop rows with invalid IPs
        self.data.dropna(subset=['ip_int'], inplace=True)

        # Convert IP bounds in the country data to integer
        self.data1['lower_bound_ip_address'] = self.data1['lower_bound_ip_address'].astype(int)
        self.data1['upper_bound_ip_address'] = self.data1['upper_bound_ip_address'].astype(int)

        # Sort both datasets for merge_asof
        self.data.sort_values('ip_int', inplace=True)
        self.data1.sort_values('lower_bound_ip_address', inplace=True)

        # Merge the datasets using merge_asof
        merged_data = pd.merge_asof(
            self.data,
            self.data1,
            left_on='ip_int',
            right_on='lower_bound_ip_address',
            direction='backward'
        )

        # Filter rows where ip_int is within the lower and upper bounds
        merged_data = merged_data[(merged_data['ip_int'] >= merged_data['lower_bound_ip_address']) &
                                  (merged_data['ip_int'] <= merged_data['upper_bound_ip_address'])]

        # Drop unnecessary columns
        merged_data.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)

        logg.info("Datasets merged successfully!")
        return merged_data
    def feature_engineering(self, merged_data):
        """
        Perform feature engineering on the dataset by creating new features based on existing columns.

        This includes:
        - Converting 'purchase_time' to datetime and creating 'hour_of_day' and 'day_of_week' columns.

        Parameters:
        -----------
        merged_data : pd.DataFrame
            The DataFrame that has already been merged and needs further feature engineering.

        Returns:
        --------
        pd.DataFrame
            The modified DataFrame with new engineered features.
        """
        # Ensure 'purchase_time' is a valid datetime
        logg.info("Performing feature engineering...")

        # Convert 'purchase_time' to datetime
        merged_data['purchase_time'] = pd.to_datetime(merged_data['purchase_time'], errors='coerce')

        # Create 'hour_of_day' and 'day_of_week' features
        merged_data['hour_of_day'] = merged_data['purchase_time'].dt.hour
        merged_data['day_of_week'] = merged_data['purchase_time'].dt.dayofweek  # 0=Monday, 6=Sunday

        logg.info("Feature engineering completed: added 'hour_of_day' and 'day_of_week'.")
        
        return merged_data

    def encode_categorical_data(self, df, cat_columns):
        """
        Perform one-hot encoding on specified categorical columns.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame containing categorical data.
        cat_columns (list): A list of categorical column names to encode.
        
        Returns:
        pd.DataFrame: DataFrame with one-hot encoded categorical columns.
        """
        logg.info("Encoding categorical data...")
        
        # Check if specified columns are in the DataFrame
        if all(col in df.columns for col in cat_columns):
            # Perform one-hot encoding
            encoded_data = pd.get_dummies(df, columns=cat_columns, drop_first=True)
            logg.info("Categorical data encoded successfully!")

            # Convert only the newly created one-hot encoded columns to integers
            one_hot_cols = [col for col in encoded_data.columns if col.startswith(tuple(cat_columns))]
            encoded_data[one_hot_cols] = encoded_data[one_hot_cols].astype(int)
            
            return encoded_data
        else:
            missing_cols = [col for col in cat_columns if col not in df.columns]
            logg.error(f"Missing columns for encoding: {missing_cols}")
            return df  # Return unchanged data if categorical columns are missing



