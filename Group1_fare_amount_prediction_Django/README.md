# Machine Learning Fare Prediction Application

This Django application predicts fare amounts based on specific features and provides analysis on the predictions. Users can also update a CSV file, which the application will process to offer insights and visualizations.

## Features

- **Fare Prediction**: Input specific features to predict the fare amount.
- **CSV Upload**: Upload a CSV file to update data and perform analysis.
- **Data Analysis**:
  - Mean fare amount prediction.
  - Total fare amount predictions.
  - Visualizations for each rider with corresponding fare amount predictions.
- **Downloadable Predictions**: Download the updated CSV file with predicted fare amounts.

## Requirements

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2. Set up a virtual environment (optional but recommended):

    ```bash
    python3 -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Run database migrations:

    ```bash
    python manage.py migrate
    ```

5. Start the development server:

    ```bash
    python manage.py runserver
    ```

6. Open your browser and go to `http://127.0.0.1:8000/` to access the application.

## Usage

1. Navigate to the input page to enter the features for fare prediction.
2. Upload your CSV file using the provided form.
3. View the analysis results, including:
   - Mean fare amount.
   - Total fare amount.
   - Visual plots for fare predictions.
4. Download the CSV file with predictions for your records.
