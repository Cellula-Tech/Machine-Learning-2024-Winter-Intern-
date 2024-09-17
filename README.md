Booking Data Analysis - Exploratory Data Analysis (EDA)
Overview
This project performs an Exploratory Data Analysis (EDA) on a booking dataset. The goal is to analyze various aspects of the booking data, such as trends over time, room type demand, and the impact of special requests on pricing. The data is cleaned, transformed, and visualized using Python libraries like Pandas, Seaborn, and Matplotlib.

Features
Data Loading and Cleaning:

Load booking data from a CSV file.
Clean column names, convert data types, and handle missing values.
Remove duplicate records.
Data Visualization:

Time series plot to visualize booking trends over time.
Bar plots comparing weekend vs weekday bookings.
Visualizing room type and meal plan demand.
Scatter plots to analyze relationships between lead time and stay duration.
Pie chart showing repeated bookings.
Bar plot showing the impact of special requests on average prices.
Correlation heatmap to identify relationships between various features.
Correlation Analysis:

Investigate the most influential factors affecting the booking status using correlation analysis.
Visualizations
Monthly Bookings Over Time: Shows how the number of bookings fluctuates on a monthly basis.
Weekend vs Weekday Bookings: Compares total nights booked on weekends versus weekdays.
Room Type and Meal Plan Demand: Analyzes the popularity of different room types and meal plans.
Special Requests Impact: Investigates how the number of special requests affects the average price.
Correlation Heatmap: Displays the correlation between different features of the dataset.
Files
first inten project.csv: The original booking dataset used for analysis.
cleaned_booking_data.csv: The cleaned version of the dataset after processing.
Setup
Install the necessary libraries:

bash
Copy code
pip install pandas numpy seaborn matplotlib scikit-learn
Run the Python script or Jupyter Notebook to perform the EDA.

Output
The cleaned dataset is saved as cleaned_booking_data.csv.
Various visualizations and insights from the data are displayed using plots.
