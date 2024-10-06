import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

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

target = data['booking status'].fillna(data['booking status'].mode()[0])

# Clean up target values to handle any inconsistencies
target = target.str.strip().str.lower()  # Convert to lowercase and strip whitespace
target = target.map({'canceled': 0, 'not_canceled': 1})

# Convert object columns to numeric
for col in features.columns:
    if features[col].dtype == 'object':
        features[col] = pd.to_numeric(features[col], errors='coerce')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Initialize models
models = {
    "XGBoost": xgb.XGBClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Store accuracy scores
accuracy_scores = {}

# Train and evaluate models
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[model_name] = accuracy
    
    # Print classification report
    print(f"\nClassification Report for {model_name}:\n", classification_report(y_test, y_pred))

# Display accuracy scores
print("\nAccuracy Scores:")
for model_name, score in accuracy_scores.items():
    print(f"{model_name}: {score:.4f}")

# Plot accuracy scores
plt.bar(accuracy_scores.keys(), accuracy_scores.values())
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.axhline(y=0.5, color='r', linestyle='--')  # Add a baseline
plt.show()
