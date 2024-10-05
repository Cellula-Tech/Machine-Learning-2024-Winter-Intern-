# -*- coding: utf-8 -*-
"""Second-Task.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JEGj9ADqMvgd5XmbhunXp0RDBYVRofa0

**Import Needed Libararies**
"""

import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

"""**Loading The Dataset**"""

df = pd.read_csv("/content/first-inten-project.csv")
print(df.head(5))

"""**Check Data Info**"""

df.info()

"""**Checking The Null**"""

df.isnull().sum()

"""**Statistical Analysis**"""

df.describe()

"""**Box Plot For Outliers**"""

fig, axs = plt.subplots(len(df.select_dtypes(include=np.number).columns),1,dpi=95, figsize=(7,17)) # Create as many subplots as numeric columns
i = 0
for col in df.columns:
    if df[col].dtype in [np.int64, np.float64]: # Check if the column is numeric
        axs[i].boxplot(df[col], vert=False)
        axs[i].set_ylabel(col)
        i+=1
plt.show()

"""**Dropping Outliers**"""

# Identify the quartiles For Lead Time
q1, q3 = np.percentile(df['lead time'], [25, 75])
# Calculate the interquartile range
iqr = q3 - q1
# Calculate the lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
# Drop the outliers
clean_data = df[(df['lead time'] >= lower_bound)
                & (df['lead time'] <= upper_bound)]
#-------------------------------------------------------------------------------
# Identify the quartiles For Average Price
q1, q3 = np.percentile(df['average price '], [25, 75])
# Calculate the interquartile range
iqr = q3 - q1
# Calculate the lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
# Drop the outliers
clean_data = df[(df['average price '] >= lower_bound)
                & (df['average price '] <= upper_bound)]

"""**Correlation & Heatmap**"""

corr = df.corr(numeric_only=True)

plt.figure(dpi=130)
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt= '.2f')
plt.show()

"""**Normlization**"""

X = df.select_dtypes(include=np.number)

scaler = MinMaxScaler(feature_range=(0, 1))

rescaledX = scaler.fit_transform(X)
rescaledX[:5]

import pandas as pd
from sklearn.preprocessing import Normalizer
import numpy as np

df = pd.read_csv('/content/first-inten-project.csv')


X = df.select_dtypes(include=np.number)

scaler = Normalizer()
scaled_data = scaler.fit_transform(X)


scaled_df = pd.DataFrame(scaled_data, columns=X.columns)
print(scaled_df.head())

#Standarization
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


df = pd.read_csv('/content/first-inten-project.csv')


X = df.select_dtypes(include=np.number)

scaler = StandardScaler()


scaled_data = scaler.fit_transform(X)


scaled_df = pd.DataFrame(scaled_data, columns=X.columns)


print(scaled_df.head())

status = pd.get_dummies(df[['booking status']])

print(status.head)

from sklearn import linear_model
from sklearn.model_selection import train_test_split

X= df['Booking_ID']
y=df['booking status']
X_train, X_test, y_train, y_test = train_test_split( X,y , random_state=104,test_size=0.25, shuffle=True)

from sklearn.metrics import mean_squared_error

status = pd.get_dummies(df['booking status'], prefix='status')

df = pd.concat([df, status], axis=1)

X = df.select_dtypes(include=np.number)

X_train, X_test, y_train, y_test = train_test_split(X, df['status_Not_Canceled'], random_state=104, test_size=0.25, shuffle=True) #Use one-hot encoded column as target

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LinearRegression


model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test)
print(y_pred)
print(mean_squared_error(y_test,y_pred))

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics


y_pred = model.predict(X_test_scaled) #Predict on the scaled test data

mse = mean_squared_error(y_test, y_pred) # Calculate Mean Squared Error
r2 = r2_score(y_test, y_pred)  # Calculate R-squared

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


#Accuracy = metrics.accuracy_score(y_test, y_pred) # Calculate accuracy using y_test and y_pred
#print(f"Accuracy: {Accuracy}")