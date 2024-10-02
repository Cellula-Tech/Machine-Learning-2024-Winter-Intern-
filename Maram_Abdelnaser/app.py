from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

app = Flask(__name__)

# Load the saved model and scaler
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    adults = int(request.form['adults'])
    children = int(request.form['children'])
    weekend_nights = int(request.form['weekend_nights'])
    week_nights = int(request.form['week_nights'])
    meal_plan = request.form['meal_plan']
    car_parking_space = int(request.form['car_parking_space'])
    room_type = request.form['room_type']
    market_segment = request.form['market_segment']
    reservation_date = request.form['reservation_date']
    special_requests = int(request.form['special_requests'])
    repeated = int(request.form['repeated'])
    p_c = int(request.form['p_c'])
    p_not_c = int(request.form['p_not_c'])
    average_price=float(request.form['average_price'])
    
    # Calculate lead time, day, and month from reservation date
    today = datetime.now()
    reservation_date = datetime.strptime(reservation_date, '%Y-%m-%d')
    lead_time = (today - reservation_date).days
    day = reservation_date.day
    month = reservation_date.month
    # Initialize variables for each meal type
    meal_plan_1 = 1 if meal_plan == "Meal Plan 1" else 0
    meal_plan_2 = 1 if meal_plan == "Meal Plan 2" else 0
    meal_plan_3 = 1 if meal_plan == "Meal Plan 3" else 0
    notSelected = 1 if meal_plan == "Not Selected" else 0
    # Initialize variables for each room type
    room_type_1 = 1 if room_type == "Room Type 1" else 0
    room_type_2 = 1 if room_type == "Room Type 2" else 0
    room_type_3 = 1 if room_type == "Room Type 3" else 0
    room_type_4 = 1 if room_type == "Room Type 4" else 0
    room_type_5 = 1 if room_type == "Room Type 5" else 0
    room_type_6 = 1 if room_type == "Room Type 6" else 0
    room_type_7 = 1 if room_type == "Room Type 7" else 0
    # Initialize variables for each market segment type
    Offline = 1 if market_segment == "Offline" else 0
    Online = 1 if market_segment == "Online" else 0
    Corporate = 1 if market_segment == "Corporate" else 0
    Aviation = 1 if market_segment == "Aviation" else 0
    Complementary = 1 if market_segment == "Complementary" else 0
    # Create a DataFrame from the input values
    data = {
        'lead_time': [lead_time],
        'day': [day],
        'number_of_week_nights': [week_nights],
        'average_price': [average_price],
        'month': [month],
        'special_requests': [special_requests],
        'number_of_weekend_nights': [weekend_nights],
        'Online': [Online],
        'number_of_children': [children],
        'Offline': [Offline],
        'number_of_adults': [adults],
        'p_not_c': [p_not_c],
        'Corporate': [Corporate],
        'meal_plan_1': [meal_plan_1],
        'repeated': [repeated],
        'p_c': [p_c],
        'room_type_3': [room_type_3],
        'Complementary': [Complementary],
        'Aviation': [Aviation],
        'car_parking_space': [car_parking_space],
        'meal_plan_2': [meal_plan_2],
        'room_type_1': [room_type_1],
        'room_type_6': [room_type_6],
        'room_type_4': [room_type_4],
        'room_type_2': [room_type_2],
        'room_type_5': [room_type_5],
        'notSelected': [notSelected],
        'meal_plan_3': [meal_plan_3],
        'room_type_7': [room_type_7],
    }
    
    # Reshape input for scaling
    df=pd.DataFrame(data)

    # Select the first 16 columns
    df_first_five = df.iloc[:, :16]

    features = df_first_five.values

    # Apply MinMax scaling
    scaler = MinMaxScaler()
    scaled_features=scaler.fit_transform(features)

    # Make the prediction using the scaled features
    prediction = model.predict(scaled_features)

    # Convert the prediction to a human-readable result
    result = 'Cancelled' if prediction[0] == 1 else 'Not Cancelled'
    return render_template('index.html',prediction_text="The Booking is {}".format(result))

    #return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run()
