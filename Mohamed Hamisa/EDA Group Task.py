

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('/content/first inten project.csv')

# Initial data check
print("Initial Data Shape:", df.shape)
print("Initial Data Info:")
print(df.info())
print("Initial Data Head:")
print(df.head())

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Convert 'date of reservation' column to datetime
df['date of reservation'] = pd.to_datetime(df['date of reservation'], errors='coerce')

# Drop irrelevant columns
df_cleaned = df.drop(['booking_id'], axis=1, errors='ignore')

# Convert categorical columns to numeric codes
df_cleaned['booking status'] = df_cleaned['booking status'].map({'canceled': 1, 'not_canceled': 0})
df_cleaned['type of meal'] = df_cleaned['type of meal'].astype('category').cat.codes
df_cleaned['room type'] = df_cleaned['room type'].astype('category').cat.codes
df_cleaned['market segment type'] = df_cleaned['market segment type'].astype('category').cat.codes

# Handle missing values
df_cleaned['average price'] = pd.to_numeric(df_cleaned['average price'], errors='coerce')

# Drop duplicates
df_cleaned = df_cleaned.drop_duplicates()

# Check for data types and basic statistics
print(df_cleaned.dtypes)
print(df_cleaned.describe(include='all'))

# Normalize numerical data
scaler = StandardScaler()
df_cleaned[['average price']] = scaler.fit_transform(df_cleaned[['average price']])

# Check for outliers
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df_cleaned['average price'])
plt.show()



# 1. Booking Count in Specific Time Period (Time Series Plot)
if 'date of reservation' in df_cleaned.columns:
    df_cleaned.set_index('date of reservation', inplace=True)
    monthly_bookings = df_cleaned.resample('M').size()

    # Plot Monthly Bookings
    plt.figure(figsize=(10,6))
    monthly_bookings.plot()
    plt.title('Monthly Bookings Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Bookings')
    plt.show()
else:
    print("Column 'date of reservation' is missing.")

# 2. Weekend vs Weekday Bookings (Bar Plot)
if 'number of weekend nights' in df_cleaned.columns and 'number of week nights' in df_cleaned.columns:
    weekend_vs_weekday = df_cleaned[['number of weekend nights', 'number of week nights']].sum()

    # Plot Weekend vs Weekday Bookings
    plt.figure(figsize=(8,6))
    weekend_vs_weekday.plot(kind='bar', color=['orange', 'blue'])
    plt.title('Weekend vs Weekday Bookings')
    plt.xlabel('Booking Type')
    plt.ylabel('Total Nights')
    plt.show()
else:
    print("Columns 'number of weekend nights' or 'number of week nights' are missing.")

# 3. Room Type Demand (Bar Plot)
if 'room type' in df_cleaned.columns:
    room_type_demand = df_cleaned['room type'].value_counts()

    plt.figure(figsize=(8,6))
    room_type_demand.plot(kind='bar', color='green')
    plt.title('Room Type Demand')
    plt.xlabel('Room Type')
    plt.ylabel('Number of Bookings')
    plt.show()
else:
    print("Column 'room type' is missing.")

# 4. Meal Plan Demand (Bar Plot)
if 'type of meal' in df_cleaned.columns:
    meal_type_demand = df_cleaned['type of meal'].value_counts()

    plt.figure(figsize=(8,6))
    meal_type_demand.plot(kind='bar', color='purple')
    plt.title('Meal Plan Demand')
    plt.xlabel('Meal Type')
    plt.ylabel('Number of Bookings')
    plt.show()
else:
    print("Column 'type of meal' is missing.")

# 5. Correlation Between Lead Time and Stay Duration (Scatter Plot)
if 'lead time' in df_cleaned.columns and 'number of week nights' in df_cleaned.columns:
    plt.figure(figsize=(8,6))
    plt.scatter(df_cleaned['lead time'], df_cleaned['number of week nights'], alpha=0.6)
    plt.title('Lead Time vs Stay Duration')
    plt.xlabel('Lead Time (days)')
    plt.ylabel('Number of Week Nights')
    plt.show()
else:
    print("Columns 'lead time' or 'number of week nights' are missing.")

# 6. Repeated Bookings (Pie Chart)
if 'repeated' in df_cleaned.columns:
    repeated_booking = df_cleaned['repeated'].value_counts()

    plt.figure(figsize=(6,6))
    repeated_booking.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    plt.title('Repeated Bookings')
    plt.ylabel('')
    plt.show()
else:
    print("Column 'repeated' is missing.")

# 7. Impact of Special Requests on Average Price (Bar Plot)
if 'special requests' in df_cleaned.columns:
    special_request_impact = df_cleaned.groupby('special requests')['average price'].mean()

    plt.figure(figsize=(8,6))
    special_request_impact.plot(kind='bar', color='brown')
    plt.title('Impact of Special Requests on Average Price')
    plt.xlabel('Number of Special Requests')
    plt.ylabel('Average Price')
    plt.show()
else:
    print("Column 'special requests' is missing.")

# 8. Lead Time Impact on Average Price (Line Plot)
if 'lead time' in df_cleaned.columns:
    lead_time_impact_price = df_cleaned.groupby('lead time')['average price'].mean()

    plt.figure(figsize=(10,6))
    lead_time_impact_price.plot()
    plt.title('Lead Time Impact on Average Price')
    plt.xlabel('Lead Time (days)')
    plt.ylabel('Average Price')
    plt.show()
else:
    print("Column 'lead time' is missing.")

# 9. Correlation Matrix (Heatmap)
correlation_matrix = df_cleaned.corr()

plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 10. Most Influential Factors on Booking Status (Bar Plot)
if 'booking status' in correlation_matrix.columns:
    influential_factors = correlation_matrix['booking status'].sort_values(ascending=False)

    # Filter out factors with zero or NaN correlation
    influential_factors = influential_factors[(influential_factors != 0) & (~influential_factors.isna())]

    # Check if there are any valid factors to plot
    if not influential_factors.empty:
        plt.figure(figsize=(8,6))
        influential_factors.plot(kind='bar', color='blue')
        plt.title('Influential Factors on Booking Status')
        plt.xlabel('Factors')
        plt.ylabel('Correlation with Booking Status')
        plt.show()
    else:
        print("No influential factors with valid correlation found.")
else:
    print("Column 'booking status' is missing.")

from scipy.stats import skew
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function to handle skewness
def handle_skewness(col, skewness_value):
    if skewness_value > 1:
        print(f"Handling positive skewness for {col}")
        # Apply log transformation (add 1 to avoid log(0))
        df_cleaned[col] = np.log1p(df_cleaned[col])  # log(1 + x)
    elif skewness_value < -1:
        print(f"Handling negative skewness for {col}")
        # Apply square transformation
        df_cleaned[col] = np.square(df_cleaned[col])
    else:
        print(f"No significant skewness for {col}")

# Apply skewness handling for numeric columns
numeric_columns = ['average price', 'lead time', 'number of week nights', 'number of weekend nights']

for col in numeric_columns:
    if col in df_cleaned.columns:
        # Plot before transformation
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(df_cleaned[col], kde=True, color='blue')
        plt.title(f'{col} - Before Transformation')
        plt.xlabel(col)

        # Calculate skewness before transformation
        skewness_before = skew(df_cleaned[col].fillna(0))  # Fill NaNs with 0 for skew calculation
        print(f"Skewness for {col} before: {skewness_before:.2f}")

        # Apply transformation to handle skewness
        handle_skewness(col, skewness_before)

        # Plot after transformation
        plt.subplot(1, 2, 2)
        sns.histplot(df_cleaned[col].fillna(0), kde=True, color='green')
        plt.title(f'{col} - After Transformation')
        plt.xlabel(col)

        # Calculate skewness after transformation
        skewness_after = skew(df_cleaned[col].fillna(0))
        print(f"Skewness for {col} after: {skewness_after:.2f}")

        plt.tight_layout()
        plt.show()
    else:
        print(f"Column '{col}' is missing.")

# Save cleaned data to a CSV file
df_cleaned.to_csv('cleaned_booking_data.csv', index=False)

print("Cleaned data has been saved to 'cleaned_booking_data.csv'.")
