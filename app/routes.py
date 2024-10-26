# routes.py

from flask import Blueprint, request, jsonify, render_template
from model import FraudModel

# Create a Blueprint for routes
routes = Blueprint('routes', __name__)

# Load the model
model = FraudModel('model.pkl')  # Update with the correct path to your model

@routes.route('/')
def index():
    return render_template('index.html')

@routes.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        prediction = model.predict(data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
