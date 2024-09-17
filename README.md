# Booking Data Analysis - Exploratory Data Analysis (EDA)

## Project Overview

This project presents a comprehensive **Exploratory Data Analysis (EDA)** on a **booking dataset**. The goal is to uncover trends, patterns, and key insights from the data to better understand booking behaviors, room preferences, and pricing strategies. By cleaning, transforming, and visualizing the data, we derive meaningful conclusions to help improve booking strategies.

### Key Features

1. **Data Loading and Cleaning**:
   - Load data from the booking dataset. 
   - Clean column names, correct data types, and handle missing values.
   - Remove duplicates for a cleaner dataset.

2. **Data Visualization**:
   - Time series plots to show booking trends over time.
   - Bar plots to compare **weekend vs weekday bookings**.
   - Visualize demand for different **room types** and **meal plans**.
   - Scatter plots to explore relationships like **lead time** vs **stay duration**.
   - Pie chart displaying the proportion of **repeated bookings**.
   - Analyze how **special requests** affect the average booking price.
   - Correlation heatmaps to find relationships between various booking features.

3. **Correlation Analysis**:
   - Examine the most influential factors that impact **booking status** using correlation coefficients.

---

## Visualizations

- **Monthly Bookings Over Time**: Shows the fluctuation of booking volumes on a monthly basis.
- **Weekend vs Weekday Bookings**: Visualizes the comparison between weekend and weekday night stays.
- **Room Type and Meal Plan Demand**: Highlights which room types and meal plans are most popular.
- **Impact of Special Requests**: Analyzes how special requests influence the average price per booking.
- **Correlation Heatmap**: Displays correlations between various features in the booking data.

---

## Project Structure

- **`first_inten_project.csv`**: The original dataset used for the analysis.
- **`cleaned_booking_data.csv`**: The cleaned and processed dataset.
- **Plots and Visualizations**: Various plots are generated for in-depth data analysis.

---

## Setup Instructions

### Step 1: Install Required Libraries

Ensure the following libraries are installed:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
