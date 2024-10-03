from flask import Flask, render_template, request
import pickle
import numpy as np

# Load your trained model using pickle
with open('hotel.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        weekend_nights = int(request.form['weekend_nights'])
        week_nights = int(request.form['week_nights'])
        parking_space = int(request.form['parking_space'])
        lead_time = int(request.form['lead_time'])
        repeated = int(request.form['repeated'])
        p_c = int(request.form['p_c'])
        p_not_c = int(request.form['p_not_c'])
        avg_price = float(request.form['avg_price'])
        special_requests = int(request.form['special_requests'])
        total_guest = int(request.form['total_guest'])
        room_type = int(request.form['room_type'])
        meal_plan = int(request.form['meal_plan'])
        market_segment_complementary = int(request.form['market_segment_complementary'])
        market_segment_corporate = int(request.form['market_segment_corporate'])
        market_segment_offline = int(request.form['market_segment_offline'])
        market_segment_online = int(request.form['market_segment_online'])

        # Create input array for prediction
        input_data = np.array([[weekend_nights, week_nights, parking_space, lead_time, repeated, p_c, p_not_c,
                                avg_price, special_requests, total_guest, room_type, meal_plan, 
                                market_segment_complementary, market_segment_corporate, market_segment_offline, 
                                market_segment_online]])

        # Make prediction
        prediction = model.predict(input_data)

        # Convert prediction to readable form
        result = 'Cancel' if prediction == 1 else 'Not Cancel'

        return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)
