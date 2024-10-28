import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.views import View
import io
import base64

model_path = os.path.join(os.path.dirname(
    __file__), '..', 'fare_amount_regressor.pkl')
model = joblib.load(model_path)


def home(request):
    if request.method == 'POST':
        # Extracting input data from the request
        passenger_count = int(request.POST['passenger_count'])
        hour = int(request.POST['hour'])
        day = int(request.POST['day'])
        month = int(request.POST['month'])
        weekday = int(request.POST['weekday'])
        year = int(request.POST['year'])
        distance = float(request.POST['distance'])
        bearing = float(request.POST['bearing'])
        car_condition = request.POST['car_condition']
        weather = request.POST['weather']
        traffic_condition = request.POST['traffic_condition']

        # Create a DataFrame to match the model's input structure
        input_data = pd.DataFrame([{
            'Car Condition': car_condition,
            'Weather': weather,
            'Traffic Condition': traffic_condition,
            'passenger_count': passenger_count,
            'hour': hour,
            'day': day,
            'month': month,
            'weekday': weekday,
            'year': year,
            'distance': distance,
            'bearing': bearing,
        }])

        # Make predictions using the model
        prediction = model.predict(input_data)

        # Return the prediction to the rendered template
        return render(request, 'home.html', {'prediction': prediction[0]})
    else:
        return render(request, 'home.html')


class UploadCSVView(View):
    def get(self, request):
        return render(request, 'upload.html')

    def post(self, request):
        if request.FILES:
            uploaded_file = request.FILES['file']
            try:
                df = pd.read_csv(uploaded_file)
                predictions = self.process_data(df)

                # Create a new DataFrame for results
                results_df = df.copy()
                results_df['predicted_fare'] = predictions

                # Prepare analysis data
                mean_fare = predictions.mean()
                total_fare = predictions.sum()
                plot_url = self.create_plot(predictions)

                # Store analysis data in the context
                context = {
                    'mean_fare': mean_fare,
                    'total_fare': total_fare,
                    'plot_url': plot_url,
                    'results_df': results_df.to_csv(index=False),
                }

                return render(request, 'upload.html', context)

            except ValueError as e:
                return render(request, 'upload.html', {'error': str(e)})
            except Exception as e:
                return render(request, 'upload.html', {'error': 'An error occurred while processing the file.'})

        return render(request, 'upload.html')

    def process_data(self, df):
        required_columns = [
            'Car Condition', 'Weather', 'Traffic Condition',
            'passenger_count', 'hour', 'day', 'month', 'weekday', 'year', 'distance', 'bearing'
        ]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        input_data = df[required_columns].copy()

        if 'Traffic Condition' not in input_data:
            input_data['Traffic Condition'] = 'Moderate'
        if 'Car Condition' not in input_data:
            input_data['Car Condition'] = 'Very Good'
        if 'Weather' not in input_data:
            input_data['Weather'] = 'sunny'

        predictions = model.predict(input_data)
        return predictions

    def create_plot(self, predictions):
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(predictions)), predictions, color='blue')
        plt.xlabel('Test Cases')
        plt.ylabel('Predicted Fare Amount')
        plt.title('Predicted Fare Amount for Each Test Case')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        return f'data:image/png;base64,{plot_url}'


class DownloadCSVView(View):
    def post(self, request):
        csv_data = request.POST.get('csv_data')
        response = HttpResponse(csv_data, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="predicted_fares.csv"'
        return response
