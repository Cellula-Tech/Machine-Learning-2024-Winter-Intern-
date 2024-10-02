from collections import OrderedDict
from flask import Flask, jsonify, request, render_template, flash
import pickle
import pandas as pd
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


def validate_numeric_fields(form_data):
    numeric_fields = [
        'number_of_adults',
        'number_of_children',
        'number_of_weekend_nights',
        'number_of_week_nights',
        'lead time',
        'P-C',
        'P-not-C',
        'average price',
        'special requests'
    ]

    for field in numeric_fields:
        value = form_data.get(field)
        if value:
            try:
                num_value = float(value)
                if num_value < 0:
                    return f"The value for {field.replace('_', ' ').title()} cannot be negative."
            except ValueError:
                return f"The value for {field.replace('_', ' ').title()} must be a number."
    return None


def update_form_data(form_data):
    """Update form data by converting reservation date and removing original date."""
    reservation_date = pd.to_datetime(form_data['date_of_reservation'])
    form_data['reservation_year'] = reservation_date.year
    form_data['reservation_month'] = reservation_date.month
    form_data['reservation_day'] = reservation_date.day
    form_data.pop('date_of_reservation')
    return form_data


def create_ordered_data(form_data):
    """Create an ordered dictionary from the form data."""
    ordered_data = OrderedDict()
    relevant_columns = [
        'number of weekend nights',
        'number of week nights',
        'lead time',
        'P-C',
        'P-not-C',
        'average price ',
        'special requests',
        'reservation_year',
        'reservation_month',
        'reservation_day',
        'type of meal_Meal Plan 1',
        'type of meal_Not Selected',
        'room type_Room_Type 1',
        'market segment type_Offline'
    ]
    for key in relevant_columns:
        if key in form_data:
            ordered_data[key] = form_data[key]
        elif key.replace(" ", "_") in form_data:
            ordered_data[key] = form_data[key.replace(" ", "_")]
        elif key == "type of meal_Meal Plan 1":
            ordered_data[key] = 1 if form_data['type_of_meal'] == 'Meal Plan 1' else 0
        elif key == "type of meal_Not Selected":
            ordered_data[key] = 1 if form_data['type_of_meal'] == 'Not Selected' else 0
        elif key == "room type_Room_Type 1":
            ordered_data[key] = 1 if form_data['room_type'] == 'Room_Type 1' else 0
        elif key == "market segment type_Offline":
            ordered_data[key] = 1 if form_data['market_segment_type'] == 'Offline' else 0
        else:
            ordered_data[key] = 0  # Default value if not found

    return ordered_data


def process_ordered_data(ordered_data):
    """Convert ordered data into a DataFrame and scale it."""
    df = pd.DataFrame([ordered_data])
    input_array = scaler.transform(df)
    return input_array


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    form_data = update_form_data(form_data)
    error_message = validate_numeric_fields(form_data)
    if error_message:
        flash(error_message, 'error')
        return render_template('index.html', prediction=None)
    ordered_data = create_ordered_data(form_data)
    print("Ordered data:", ordered_data)  # For debugging

    input_array = process_ordered_data(ordered_data)
    print("----------------------\n")
    print(input_array)

    prediction = model.predict(input_array)
    print(prediction)

    prediction_text = ""
    if (prediction == 1):
        prediction_text = "Not Cancelled ان شاء الله"
    else:
        prediction_text = "Cancelled لا قدر الله"
    return render_template('index.html', prediction=prediction_text)


# Flask route to render the main page
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
