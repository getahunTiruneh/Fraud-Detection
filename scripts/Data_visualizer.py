import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging as logg
logg.basicConfig(level=logg.INFO)

class DataVisualizer:
    def __init__(self, data):
        self.data = data
    def visualize_data(self):
        # Visualizing the data
        sns.pairplot(self.data)
        plt.show()
    
    def plot_histogram(self, numerical_features):
        plt.figure(figsize=(16, 5))
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(1, len(numerical_features), i)
            sns.histplot(self.data[feature], bins=20, kde=True)
            plt.title(f'Histogram for {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        logg.info("Histograms plotted successfully!")
    def plot_bar_chart(self, categorical_features: list):
        """Plot bar charts for each categorical feature in 2 columns.

        Args:
            categorical_features (list): List of categorical feature names.
        """
        try:
            # Set up number of rows and columns (2 columns max)
            num_features = len(categorical_features)
            num_cols = 2  # We want 2 columns
            num_rows = (num_features + num_cols - 1) // num_cols  # Calculate rows dynamically
            
            plt.figure(figsize=(num_cols * 6, num_rows * 4))  # Adjust the figure size based on columns and rows

            for i, feature in enumerate(categorical_features, 1):
                if feature not in self.data.columns:
                    logg.error(f"Feature '{feature}' not found in data!")
                    continue  # Skip this feature if it's not in the data

                plt.subplot(num_rows, num_cols, i)
                sns.barplot(
                    x=self.data[feature].value_counts().index,
                    y=self.data[feature].value_counts().values,
                    palette='viridis'
                )
                plt.title(f'Bar Chart for {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')

            plt.tight_layout()
            plt.show()

            logg.info("Bar charts plotted successfully!")
        except Exception as e:
            logg.error(f"An error occurred while plotting bar charts: {e}")

    
    def plot_scatter_matrix(self, numerical_features):
        plt.figure(figsize=(16, 10))
        sns.pairplot(self.data[numerical_features], palette='viridis')
        plt.title('Scatter Matrix')
        plt.tight_layout()
        plt.show()
        logg.info("Scatter matrix plotted successfully!")
        
    def scatter_plot(self, x_feature, y_feature):
        logg.info("Plotting scatter plot...")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.data[x_feature], y=self.data[y_feature], palette='viridis')
        plt.title(f'Scatter Plot: {x_feature} vs {y_feature}')
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.show()
        
    def plot_box_plot(self, numerical_features):
        plt.figure(figsize=(16, 5))
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(1, len(numerical_features), i)
            sns.boxplot(self.data[feature])
            plt.title(f'Box Plot for {feature}')
            plt.xlabel(feature)
    
    def plot_correlation_matrix(self, numerical_features):
        corr_matrix = self.data[numerical_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()
        logg.info("Correlation matrix plotted successfully!")
        
    def plot_distribution_with_class(self, features: list, target: str, feature_type: str = 'numerical'):
        """Plot the distribution of features with respect to the target class.

        Args:
            features (list): List of features to plot.
            target (str): The target variable (class).
            feature_type (str): 'numerical' for histograms and 'categorical' for bar plots. Defaults to 'numerical'.
        """
        try:
            num_features = len(features)
            num_cols = min(num_features, 2)  # Up to 3 columns per row
            num_rows = (num_features + num_cols - 1) // num_cols  # Calculate rows dynamically

            plt.figure(figsize=(num_cols * 5, num_rows * 4))

            for i, feature in enumerate(features, 1):
                if feature not in self.data.columns or target not in self.data.columns:
                    logg.error(f"Feature '{feature}' or target '{target}' not found in data!")
                    continue  # Skip invalid features or targets

                plt.subplot(num_rows, num_cols, i)

                if feature_type == 'numerical':
                    # Plot histogram with KDE for each class
                    sns.histplot(data=self.data, x=feature, hue=target, bins=20, kde=True, palette="viridis", element="step")
                    plt.title(f'Distribution of {feature} by {target}')
                    plt.xlabel(feature)
                    plt.ylabel('Frequency')

                elif feature_type == 'categorical':
                    # Plot countplot for each class
                    sns.countplot(x=feature, hue=target, data=self.data, palette="viridis")
                    plt.title(f'Distribution of {feature} by {target}')
                    plt.xlabel(feature)
                    plt.ylabel('Count')

            plt.tight_layout()
            plt.show()

            logg.info("Feature distributions with respect to target class plotted successfully!")

        except Exception as e:
            logg.error(f"An error occurred while plotting feature distributions: {e}")