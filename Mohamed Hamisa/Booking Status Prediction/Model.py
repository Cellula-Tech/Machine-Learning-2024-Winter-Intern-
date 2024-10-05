
!pip install flask-cors

!pip install pyngrok

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import threading

# Load data
data = pd.read_csv('/content/first inten project.csv', encoding='latin1')

# Check for null values and data types
print("Null values in each column:\n", data.isnull().sum())
print("\nData types of each column:\n", data.dtypes)

# Strip whitespace from column names and data
data.columns = data.columns.str.strip()
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Handle outliers using IQR
Q1 = data['average price'].quantile(0.25)
Q3 = data['average price'].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (data['average price'] < (Q1 - 1.5 * IQR)) | (data['average price'] > (Q3 + 1.5 * IQR))
data = data[~outlier_condition]

data['date of reservation'] = pd.to_datetime(data['date of reservation'], errors='coerce')

# Feature engineering: Total Guests, Total Nights, Special Request Count
data['Total Guests'] = data['number of adults'] + data['number of children']
data['Total Nights'] = pd.to_numeric(data['number of weekend nights'], errors='coerce') + pd.to_numeric(data['number of week nights'], errors='coerce')
data['Special Request Count'] = data['special requests'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

# Encode Meal Type
meal_type_mapping = {meal: idx for idx, meal in enumerate(data['type of meal'].unique())}
data['Ordered Meal Type'] = data['type of meal'].map(meal_type_mapping)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Define manual mappings for categorical features
meal_type_mapping = {'Meal Plan 1': 0, 'Meal Plan 2': 1, 'Meal Plan 3': 2, 'Not Selected': 3}
market_segment_mapping = {'Online': 0, 'Offline': 1, 'Corporate': 2, 'Others': 3}
room_type_mapping = {
    'Room_Type 1': 0,
    'Room_Type 2': 1,
    'Room_Type 3': 2,
    'Room_Type 4': 3,
    'Room_Type 5': 4,
    'Room_Type 6': 5,
    'Room_Type 7': 6
}

# Map categorical features to numeric values
data['Ordered Meal Type'] = data['type of meal'].map(meal_type_mapping)
data['market segment type'] = data['market segment type'].map(market_segment_mapping)
data['room type'] = data['room type'].map(room_type_mapping)

# Handle missing values in categorical features
data['Ordered Meal Type'] = data['Ordered Meal Type'].fillna(0)
data['market segment type'] = data['market segment type'].fillna(0)
data['room type'] = data['room type'].fillna(0)

# Separate features and target variable
features = data.drop(['booking status', 'Booking_ID', 'type of meal', 'P-not-C', 'P-C', 'repeated',
                      'car parking space', 'number of adults', 'number of children',
                      'number of weekend nights', 'number of week nights',
                      'date of reservation', 'special requests'], axis=1)

target = data['booking status']

# Fill NaN values in target
target = target.fillna(target.mode()[0])

# Convert non-numeric values in numerical features
num_features = ['lead time', 'average price', 'Total Guests', 'Total Nights', 'Special Request Count']
for feature in num_features:
    if feature in features.columns:
        features[feature] = pd.to_numeric(features[feature], errors='coerce')
        if features[feature].isnull().any():
            print(f"Non-numeric values found in {feature} and converted to NaN.")

# Create pipelines for numerical and categorical features
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_features = ['Ordered Meal Type', 'market segment type', 'room type']
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Combine the pipelines
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Convert target variables to strings
y_train = y_train.astype(str)
y_test = y_test.astype(str)

# Preprocess the data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Display the shapes of the transformed data
print("Transformed X_train shape:", X_train_transformed.shape)
print("Transformed X_test shape:", X_test_transformed.shape)

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Assuming 'data' is your DataFrame with features and target
# Ensure you have a DataFrame called 'data' with a 'booking status' column


features = data.drop(['booking status', 'Booking_ID', 'type of meal', 'P-not-C', 'P-C', 'repeated',
                      'car parking space', 'number of adults', 'number of children',
                      'number of weekend nights', 'number of week nights',
                      'date of reservation', 'special requests'], axis=1)
target = data['booking status']

# Check and handle NaN values in the target variable
target = target.fillna(target.mode()[0])  # Use the mode to fill NaN values

# Check unique values in the target variable for consistency
print("Unique values in target before mapping:", target.unique())

# Clean up target values to handle any inconsistencies
target = target.str.strip().str.lower()  # Convert to lowercase and strip whitespace

# Check for unexpected values
unexpected_values = target[~target.isin(['canceled', 'not_canceled'])]
if not unexpected_values.empty:
    print("Unexpected values found in target:", unexpected_values.unique())

# Map target variables to numeric values
target = target.map({'canceled': 0, 'not_canceled': 1})

# Convert object columns to numeric
for col in features.columns:
    if features[col].dtype == 'object':
        features[col] = pd.to_numeric(features[col], errors='coerce')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)

# Define parameter grid for GridSearchCV
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1.0]
}

# Perform grid search with cross-validation
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='accuracy')
grid_search_xgb.fit(X_train, y_train)

# Get the best model and parameters
best_xgb = grid_search_xgb.best_estimator_
print(f"Best parameters for XGBoost: {grid_search_xgb.best_params_}")

# Make predictions on the test set
y_pred_xgb = best_xgb.predict(X_test)

# Calculate and print accuracy
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.4f}")

import pickle

# Save the trained model
with open('best_xgb_model.pkl', 'wb') as f:
    pickle.dump(best_xgb, f) # Changed best_rf to best_xgb

from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
with open('best_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define manual mappings for categorical features
meal_type_mapping = {'Meal Plan 1': 0, 'Meal Plan 2': 1, 'Meal Plan 3': 2, 'Not Selected': 3}
market_segment_mapping = {'Online': 0, 'Offline': 1, 'Corporate': 2, 'Others': 3}
room_type_mapping = {
    'Room_Type 1': 0,
    'Room_Type 2': 1,
    'Room_Type 3': 2,
    'Room_Type 4': 3,
    'Room_Type 5': 4,
    'Room_Type 6': 5,
    'Room_Type 7': 6
}

@app.route('/predict_booking', methods=['POST'])
def predict_booking():
    try:
        input_data = request.json

        # Log received data for debugging
        print("Received input data:", input_data)

        required_features = ['room type', 'lead time', 'market segment type',
                             'average price', 'Total Guests', 'Total Nights',
                             'Special Request Count', 'Ordered Meal Type']

        # Extract and process features
        feature_values = []
        for feature in required_features:
            value = input_data.get(feature)
            if value is None:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

            # Map categorical features to numeric values
            if feature == 'Ordered Meal Type':
                mapped_value = meal_type_mapping.get(value)
                if mapped_value is None:
                    return jsonify({'error': f'Invalid Ordered Meal Type: {value}'}), 400
                feature_values.append(mapped_value)

            elif feature == 'market segment type':
                mapped_value = market_segment_mapping.get(value)
                if mapped_value is None:
                    return jsonify({'error': f'Invalid market segment type: {value}'}), 400
                feature_values.append(mapped_value)

            elif feature == 'room type':
                mapped_value = room_type_mapping.get(value)
                if mapped_value is None:
                    return jsonify({'error': f'Invalid room type: {value}'}), 400
                feature_values.append(mapped_value)

            elif feature in ['lead time', 'Total Guests', 'Total Nights', 'Special Request Count']:
                if not isinstance(value, (int, float)) or value < 0:
                    return jsonify({'error': f'Invalid value for {feature}: {value}. Must be a non-negative number.'}), 400
                feature_values.append(int(value))

            elif feature == 'average price':
                try:
                    feature_values.append(float(value))
                except ValueError:
                    return jsonify({'error': f'Invalid value for average price: {value}. Must be a number.'}), 400

        # Convert feature values to numpy array
        feature_array = np.array(feature_values).reshape(1, -1)

        # Make prediction
        prediction = model.predict(feature_array)[0]

        # Map prediction to booking status
        booking_status = "canceled" if prediction == 1 else "not canceled"

        return jsonify({'booking status': booking_status})

    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}'}), 400

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

!pip install Flask pyngrok joblib

!ngrok config add-authtoken 2mygcOgmrhPEpk4q5ZOvZv51wXh_7YTLcN5BrrDD7Kc746AFz

def start_ngrok():
    public_url = ngrok.connect(5016)
    print(" * Ngrok tunnel URL:", public_url)

if __name__ == '__main__':
    threading.Thread(target=start_ngrok).start()
    app.run(port=5016)  # Make sure app.run uses the same port as ngrok

