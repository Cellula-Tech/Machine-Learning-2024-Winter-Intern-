from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and selector
with open('model.pkl', 'rb') as model_file:
    rf = pickle.load(model_file)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)

    # Apply feature selection
    features_selected = selector.transform(features)

    # Make prediction
    prediction = model.predict(features_selected)

    # Return result
    return render_template('index.html', prediction_text=f'Predicted class: {prediction[0]}')

if __name__ == "__main__":
    app.run(port=3000,debug=True)
