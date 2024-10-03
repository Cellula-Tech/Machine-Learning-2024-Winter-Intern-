from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load selected features
with open('selected_features.pkl', 'rb') as file:
    selected_features = pickle.load(file)

@app.route('/', methods=["GET"])
def home():
    return render_template('index.html', selected_features=selected_features)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get values from the form
        feature_values = [float(request.form[feature]) for feature in selected_features]
        # Prepare input data for prediction
        input_data = np.array([feature_values])

        # Scale the data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = rf_model.predict(input_data_scaled)

        return render_template('index.html', prediction=prediction[0])

    except KeyError as e:
        return f"Missing form key: {str(e)}", 400

if __name__ == '__main__':
    app.run(port=3000, debug=True)
