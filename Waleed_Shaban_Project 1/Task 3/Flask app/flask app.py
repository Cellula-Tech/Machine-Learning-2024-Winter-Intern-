from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
import joblib
import os
from utilites import feature_engineering, num_feature_selection, outlier_treatment

import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Set the model path
model_path = os.path.join(os.getcwd(), 'model', 'RF_model.pkl')

updated_dataframe = pd.DataFrame()
# Load the trained model
def load_model():
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}", "error")
        return None

# Make predictions
def make_prediction(data):
    model = load_model()
    print(type(model))
    if model:
        prediction = model.predict(data)
        probabilities = model.predict_proba(data)[0]
        predictions_string = np.full((prediction.shape[0],), 'Not-Canceled')
        predictions_string[prediction == 0] = 'Canceled'
        return predictions_string, probabilities
    return None, None

def generate_pie_chart(probabilities):
    # Create a pie chart
    labels = ['Canceled', 'Not-Canceled']
    plt.figure(figsize=(5, 5))
    plt.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Save the pie chart as an image
    image_path = os.path.join('static', 'pie_chart.png')
    plt.savefig(image_path)
    plt.close()  # Close the figure to avoid display
    return image_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = {
            'number of adults': int(request.form['adults']),
            'number of children': int(request.form['children']),
            'number of weekend nights': int(request.form['weekend_nights']),
            'number of week nights': int(request.form['week_nights']),
            'lead time': int(request.form['lead_time']),
            'average price ': int(request.form['avg_price']),
            'car parking space': int(request.form['parking']),
            'room type': request.form['room_type'].strip(),
            'type of meal': request.form['meal_type'].strip(),
            'market segment type': request.form['market_segment'].strip(),
            'repeated': int(request.form['repeated']),
            'P-C': int(request.form['p_c']),
            'P-not-C': int(request.form['p_not_c']),
            'special requests': int(request.form['special_requests'])
        }
        data = pd.DataFrame([data])
        prediction, probabilities = make_prediction(data)
        pie_chart_url = generate_pie_chart(probabilities)  # Generate the pie chart


        if prediction is not None:
            return render_template('results.html', prediction=prediction[0], probabilities=probabilities.round(2),
                                   pie_chart_url=pie_chart_url)

    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            updated_df = make_predictions_for_csv(df)
            if updated_df is not None:
                # Store the DataFrame to save later or flash a success message
                return render_template('upload_results.html', results=updated_df)
        else:
            flash('Invalid file format. Please upload a CSV file.', 'error')
    return render_template('upload.html')


@app.route('/download_updated_csv')
def download_updated_csv():
    global updated_dataframe  # Access the global DataFrame

    # Check if the DataFrame is not empty
    if not updated_dataframe.empty:
        # Save the DataFrame to a CSV file
        updated_dataframe.to_csv('updated_results.csv', index=False)

        # Send the CSV file for download
        return send_file('updated_results.csv', as_attachment=True)
    else:
        return "No updated data available for download.", 404

def make_predictions_for_csv(df):
    global updated_dataframe  # Access the global DataFrame
    model = load_model()
    if model is None:
        return None

    try:
        df['Predicted Class'] = model.predict(df)
        df['Predicted Class'][(df['Predicted Class'] == 0)] = 'Canceled'
        df['Predicted Class'][(df['Predicted Class'] == 1)] = 'Not-Canceled'
        probabilities = model.predict_proba(df).round(2)
        df['Probability'] = np.max(probabilities, axis=1)
        updated_dataframe = df
        return df
    except Exception as e:
        flash(f"Failed to make predictions: {e}", "error")
        return None

if __name__ == '__main__':
    app.run(debug=True)
