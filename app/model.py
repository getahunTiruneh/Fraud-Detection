# model.py

import joblib
import pandas as pd

class FraudModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        # Convert input data to DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)
        return prediction[0]  # Return the first prediction
