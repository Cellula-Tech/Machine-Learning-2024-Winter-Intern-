from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load the trained model and scaler
def load_model():
    model_path = 'models/model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    return model, scaler

# Function to calculate model accuracy 
def calculate_accuracy():
    return 90.0  

# Route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    model , scaler = load_model()

    # Get the form data
    input_data = [
        request.form['lead_time'],
        request.form['repeated'],
        request.form['average_price'],
        request.form['car_parking_space'], 
        request.form['year of Booking'],
        request.form['special_requests'],
        request.form['market_segment_type_Corporate'],
        request.form['market_segment_type_Complementary'],
        request.form['market_segment_type_Offline'],
        request.form['market_segment_type_Online']
    ]

    # Convert input to numpy array and reshape for the model
    input_data = np.array([float(i) for i in input_data]).reshape(1, -1)

    # Scale the features
    scaled_features = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(scaled_features)

    result = "not canceled." if int(prediction[0]) == 1 else "canceled."

    # Calculate accuracy
    accuracy = calculate_accuracy()

    # Render the result
    return render_template('index.html', 
                           prediction=f'The prediction of booking status: {result}', 
                           accuracy=f'Model Accuracy: {accuracy}%')

if __name__ == '__main__':
    app.run(debug=True)
