import pickle
import pandas as pd
import shap
from lime import lime_tabular
import numpy as np
import matplotlib.pyplot as plt

class ModelExplainability:
    def __init__(self, model_path, data_path, target_column):
        self.model = self.load_model(model_path)
        self.data = pd.read_csv(data_path)
        self.X = self.data.drop(target_column, axis=1)
        self.target_column = target_column

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)

    def explain_with_shap(self):
        explainer = shap.Explainer(self.model, self.X)
        shap_values = explainer(self.X)
        return shap_values

    def shap_summary_plot(self, shap_values):
        shap.summary_plot(shap_values, self.X)

    def shap_force_plot(self, shap_values, index=0):
        # Ensure shap.initjs() for notebook environments
        shap.initjs()
        # If using in a non-notebook environment, you might want to adjust this to plt.show() based output.
        plt.figure()
        shap.force_plot(shap_values[index].base_values, shap_values[index].values, self.X.iloc[index, :], matplotlib=True)
        plt.show()

    def shap_dependence_plot(self, shap_values, feature_name):
        shap.dependence_plot(feature_name, shap_values, self.X)

    def explain_with_lime(self, index=0):
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(self.X),
            mode='classification',
            feature_names=self.X.columns,
            class_names=['0', '1'],  # Update based on your class labels
            discretize_continuous=True
        )
        # Ensure to pass the row as a numpy array
        exp = explainer.explain_instance(self.X.iloc[index].values, self.model.predict_proba, num_features=10)
        return exp

    def lime_feature_importance_plot(self, exp):
        exp.show_in_notebook(show_table=True)