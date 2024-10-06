from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

app = Flask(__name__)

# Read the CSV dataset file and preprocess
df = pd.read_csv('/home/omran-xy/Workspace/Cellula/Task one/data.csv')
df.drop(columns=['date of reservation', 'number of adults', 'number of children', 'number of weekend nights', 
                 'number of week nights', 'P-not-C', 'special requests'], inplace=True)

# Normalizing 'lead time' and 'average price'
scaler = MinMaxScaler()
df[['lead time', 'average price ']] = scaler.fit_transform(df[['lead time', 'average price ']])
df.dropna(inplace=True)

X = df.drop(columns=['booking status'])
y = df['booking status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(X.dot(weights))
    h = np.clip(h, 1e-10, 1 - 1e-10)
    cost = -(1/m) * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h)))
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations, tolerance=1e-6):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        prev_weights = weights.copy()
        weights = weights - (learning_rate/m) * X.T.dot(sigmoid(X.dot(weights)) - y)
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)
        
        # Stop early if convergence is reached
        if np.linalg.norm(weights - prev_weights, ord=1) < tolerance:
            break
    
    return weights, cost_history

def train_logistic_regression(X_train, y_train, learning_rate, iterations):
    weights = np.zeros(X_train.shape[1])
    weights, cost_history = gradient_descent(X_train, y_train, weights, learning_rate, iterations)
    return weights, cost_history

try:
    weights = joblib.load('weights.pkl')
except FileNotFoundError:
    weights, cost_history = train_logistic_regression(X_train, y_train, learning_rate=0.2, iterations=1500)
    joblib.dump(weights, 'weights.pkl')

# Predict function
def predict(X, weights):
    return sigmoid(np.dot(X, weights)) >= 0.5

# Evaluate the model
y_pred = predict(X_test, weights)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

@app.route('/')
def hello_world():
    return render_template('hotelform.html')

@app.route('/predictpage', methods=['POST'])
def predict_booking():
    try:
        # Extract form values and convert to appropriate types
        car_parking_space = int(request.form['car_parking_space'])
        room_type = int(request.form['room_type'])
        market_segment_type = int(request.form['market_segment_type'])
        repeated = int(request.form['repeated'])
        lead_time = int(request.form['lead_time'])
        average_price = float(request.form['average_price'])
    except ValueError:
        return render_template('hotelform.html', pred="Invalid input. Please enter valid numerical values.")
    
    # Normalize 'lead time' and 'average price'
    lead_time_normalized, average_price_normalized = scaler.transform([[lead_time, average_price]])[0]

    # Prepare the input array for prediction
    input_arr = np.array([[car_parking_space, room_type, market_segment_type, repeated, lead_time_normalized, average_price_normalized]], dtype=float)

    # Predict using the custom model
    prediction = predict(input_arr, weights)
    prob = sigmoid(np.dot(input_arr, weights))
    
    # Determine the output message based on probabilities
    if prob[0] >= 0.5:
        return render_template('hotelform.html', pred="You have a high probability of canceling your reservation.")
    else:
        return render_template('hotelform.html', pred="That's the spirit! You wouldn't cancel your precious reservation.")

if __name__ == '__main__':
    app.run(debug=True)
