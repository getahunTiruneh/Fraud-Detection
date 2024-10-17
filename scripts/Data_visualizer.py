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
        """Plot bar charts for each categorical feature.

        Args:
            categorical_features (list): List of categorical feature names.
        """
        try:
            # Set figure size dynamically based on number of features
            num_features = len(categorical_features)
            plt.figure(figsize=(num_features * 5, 5))

            for i, feature in enumerate(categorical_features, 1):
                if feature not in self.data.columns:
                    logg.error(f"Feature '{feature}' not found in data!")
                    continue  # Skip this feature if it's not in the data

                plt.subplot(1, num_features, i)
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
        sns.pairplot(self.data[numerical_features], palette='viridis')
        plt.show()
        logg.info("Scatter matrix plotted successfully!")
    
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