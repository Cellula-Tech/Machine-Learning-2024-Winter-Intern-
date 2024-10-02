import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from io import BytesIO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utilites import feature_engineering, num_feature_selection, outlier_treatment
import warnings
warnings.filterwarnings('ignore')

# Set the model path (this can be updated to point to the correct model file)
model_path = os.path.join(os.getcwd(), 'model', 'RF_model.pkl')

# Global variable to track the popup window
popup = None

# Function to load the trained model
def load_model():
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        messagebox.showerror("Model Load Error", f"Failed to load model: {e}")
        return None


# Function to predict using the model and show results in a popup
def make_prediction(data):
    model = load_model()
    if model:
        # Make the prediction
        prediction = model.predict(data)
        probabilities = model.predict_proba(data)[0]
        predictions_string = np.full((prediction.shape[0],), 'Not-Canceled')
        predictions_string[prediction == 0] = 'Canceled'
        # Display the results in a pop-up window
        return predictions_string, probabilities


# Function to show prediction and probability in a popup window with pie chart
def show_prediction_popup(prediction, probabilities):
    global popup

    # Destroy the previous popup window if it exists
    if popup is not None and popup.winfo_exists():
        popup.destroy()

    popup = tk.Toplevel()
    popup.title("Prediction Results")
    popup.geometry("300x400+600+250")

    cls = ['Canceled', 'Not-Canceled']
    # Display the prediction
    ttk.Label(popup, text=f"Predicted Class: {prediction[0]}").pack(pady=10)

    # Plot the probabilities in a pie chart
    fig, ax = plt.subplots()
    ax.pie(probabilities, labels=[cls[i] for i in range(len(probabilities))], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the pie chart in the popup
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack()



# Manual data submission
def manual_submit():
    # Collect data from the manual entry fields
    data = {
        'number of adults': adults_var.get(),
        'number of children': children_var.get(),
        'number of weekend nights': weekend_nights_var.get(),
        'number of week nights': week_nights_var.get(),
        'lead time': lead_time_var.get(),
        'average price ': avg_price_var.get(),
        'car parking space': parking_var.get(),
        'room type': room_type_var.get().strip(),
        'type of meal': meal_type_var.get().strip(),
        'market segment type': market_segment_var.get().strip(),
        'repeated': repeated_var.get(),
        'P-C': p_c_var.get(),
        'P-not-C': p_not_c_var.get(),
        'special requests': special_requests_var.get()
    }
    print("Manual Data Submitted:", data)
    data = pd.DataFrame([data])
    prediction, probabilities = make_prediction(data)
    show_prediction_popup(prediction, probabilities)


# Function to upload and read CSV file (remaining unchanged)
def make_predictions_for_csv(df):
    model = load_model()
    if model is None:
        return None  # Exit if the model fails to load

    try:
        # Make predictions for all rows
        df['Predicted Class'] = model.predict(df)
        df['Predicted Class'][(df['Predicted Class'] == 0)] = 'Canceled'
        df['Predicted Class'][(df['Predicted Class'] == 1)] = 'Not-Canceled'
        probabilities = model.predict_proba(df).round(2)

        # Add the probabilities for each class as new columns
        df[f'Probability'] = np.max(probabilities, axis=1)

        return df

    except Exception as e:
        messagebox.showerror("Prediction Error", f"Failed to make predictions: {e}")
        return None

# Modified upload_csv function
def upload_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            messagebox.showinfo("File Upload", "CSV file uploaded successfully!")

            # Make predictions and add them to the DataFrame
            updated_df = make_predictions_for_csv(df)

            if updated_df is not None:
                # Store the updated DataFrame to save later
                global updated_dataframe
                updated_dataframe = updated_df

                messagebox.showinfo("Predictions Complete", "Predictions have been made and added to the file.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file: {e}")

# Save the modified CSV file
def save_csv():
    if 'updated_dataframe' in globals():
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                # Save the updated DataFrame to the selected path
                updated_dataframe.to_csv(file_path, index=False)
                messagebox.showinfo("File Saved", "CSV file saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
    else:
        messagebox.showerror("No Data", "No updated data to save. Please upload a file first.")


# GUI Code for the main application window, tabs, and input fields

root = tk.Tk()
root.title("Hotel Reservation Prediction")
#root.state('zoomed')

padx = 10
pady = 5

tab_control = ttk.Notebook(root)

manual_tab = ttk.Frame(tab_control)
tab_control.add(manual_tab, text='Manual Data Entry')

# Fields for manual input
fields = ['Number of Adults', 'Number of Children', 'Number of Weekend Nights', 'Number of Week Nights',
          'Lead Time', 'Average Price', 'Car Parking Space', 'Room Type', 'Type of Meal', 'Market Segment Type',
          'Repeated', 'P-C', 'P-not-C', 'Special Requests']

# Create a dictionary for storing field variables
adults_var = tk.IntVar()
children_var = tk.IntVar()
weekend_nights_var = tk.IntVar()
week_nights_var = tk.IntVar()
lead_time_var = tk.IntVar()
avg_price_var = tk.DoubleVar()
parking_var = tk.IntVar()
room_type_var = tk.StringVar()
meal_type_var = tk.StringVar()
market_segment_var = tk.StringVar()
repeated_var = tk.IntVar()
p_c_var = tk.IntVar()
p_not_c_var = tk.IntVar()
special_requests_var = tk.IntVar()

# Configure row and column resizing
for i in range(15):
    manual_tab.grid_rowconfigure(i, weight=1)
manual_tab.grid_columnconfigure(0, weight=1)
manual_tab.grid_columnconfigure(1, weight=1)

# Create labels and entry boxes for each field in the manual data entry tab
ttk.Label(manual_tab, text="Number of Adults").grid(column=0, row=0, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=adults_var).grid(column=1, row=0, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="Number of Children").grid(column=0, row=1, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=children_var).grid(column=1, row=1, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="Number of Weekend Nights").grid(column=0, row=2, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=weekend_nights_var).grid(column=1, row=2, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="Number of Week Nights").grid(column=0, row=3, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=week_nights_var).grid(column=1, row=3, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="Lead Time").grid(column=0, row=4, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=lead_time_var).grid(column=1, row=4, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="Average Price").grid(column=0, row=5, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=avg_price_var).grid(column=1, row=5, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="Car Parking Space").grid(column=0, row=6, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=parking_var).grid(column=1, row=6, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="Room Type").grid(column=0, row=7, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=room_type_var).grid(column=1, row=7, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="Type of Meal").grid(column=0, row=8, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=meal_type_var).grid(column=1, row=8, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="Market Segment Type").grid(column=0, row=9, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=market_segment_var).grid(column=1, row=9, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="Repeated").grid(column=0, row=10, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=repeated_var).grid(column=1, row=10, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="P-C").grid(column=0, row=11, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=p_c_var).grid(column=1, row=11, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="P-not-C").grid(column=0, row=12, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=p_not_c_var).grid(column=1, row=12, padx=padx, pady=pady, sticky="ew")

ttk.Label(manual_tab, text="Special Requests").grid(column=0, row=13, padx=padx, pady=pady, sticky="ew")
ttk.Entry(manual_tab, textvariable=special_requests_var).grid(column=1, row=13, padx=padx, pady=pady, sticky="ew")

# Add a submit button
ttk.Button(manual_tab, text="Submit", command=manual_submit).grid(column=1, row=14, padx=padx, pady=pady, sticky="ew")

# Tab 2 - CSV File Input
file_tab = ttk.Frame(tab_control)
tab_control.add(file_tab, text='CSV File Input')

# Configure row and column resizing for the file tab
file_tab.grid_rowconfigure(0, weight=1)
file_tab.grid_columnconfigure(0, weight=1)

# Create a style for the buttons with increased padding (affects height)
style = ttk.Style()
style.configure('TButton', padding=(10, 10))  # Increase padding (left/right, top/bottom)

# Add a button to upload the CSV file with increased height
upload_button = ttk.Button(file_tab, text="Upload CSV", style='TButton', command=upload_csv)
upload_button.grid(column=0, row=0, padx=padx, pady=20, sticky="ew")  # Increase pady for more space between buttons

# Add a button to save the CSV file with increased height
save_button = ttk.Button(file_tab, text="Save CSV", style='TButton', command=save_csv)
save_button.grid(column=0, row=1, padx=padx, pady=20, sticky="ew")

# Configure row and column resizing for centering the buttons
file_tab.grid_rowconfigure(0, weight=1)
file_tab.grid_rowconfigure(1, weight=1)
file_tab.grid_columnconfigure(0, weight=1)

# Add some extra rows and columns to further center everything in the frame
#file_tab.grid_rowconfigure(2, weight=1)  # Extra space below the buttons
#file_tab.grid_columnconfigure(1, weight=1)  # Extra space to the right

# Add the tab control to the main window
tab_control.pack(expand=1, fill="both")

# Run the application
root.mainloop()
