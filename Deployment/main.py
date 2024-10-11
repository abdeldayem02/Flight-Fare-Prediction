# main.py
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form

    # Create a DataFrame with the input data
    df = pd.DataFrame([input_data])

    # Preprocessing steps
    df['Journey_day'] = pd.to_datetime(df['date_of_journey'], format="%d/%m/%Y").dt.day
    df['Journey_month'] = pd.to_datetime(df['date_of_journey'], format="%d/%m/%Y").dt.month
    df['Journey_weekday'] = pd.to_datetime(df['date_of_journey'], format="%d/%m/%Y").dt.weekday
    df['Dep_hour'] = pd.to_datetime(df['dep_time']).dt.hour
    df['Dep_min'] = pd.to_datetime(df['dep_time']).dt.minute
    df['Arrival_hour'] = pd.to_datetime(df['arrival_time']).dt.hour
    df['Arrival_min'] = pd.to_datetime(df['arrival_time']).dt.minute

    df['Total_Stops'] = df['total_stops'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})

    duration = df['duration'].iloc[0]
    if "h" not in duration:
        duration = "0h " + duration
    elif "m" not in duration:
        duration = duration + " 0m"
    hours = int(duration.split('h')[0].strip())
    minutes = int(duration.split('h')[1].replace('m', '').strip())
    df['Duration_hours'] = hours
    df['Duration_mins'] = minutes

    df['minutes'] = df['Duration_hours'] * 60 + df['Duration_mins']
    df.drop(["Duration_hours", "Duration_mins"], axis=1, inplace=True)

    df['Price'] = pd.to_numeric(df['price'], errors='coerce')

    df = df.drop(['date_of_journey', 'duration', 'dep_time', 'arrival_time', 'additional_info'], axis=1)

    df['Price_per_stop'] = df['Price'] / (df['Total_Stops'] + 1)
    df['Is_weekend'] = df['Journey_weekday'].isin([5, 6]).astype(int)
    df['Is_night_flight'] = ((df['Dep_hour'] >= 20) | (df['Dep_hour'] <= 5)).astype(int)

    # Encoding categorical variables
    categorical_columns = ['airline', 'source', 'destination', 'route']
    for col in categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Select features used in the model
    features = ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour', 'Dep_min', 
                'Arrival_hour', 'Arrival_min', 'Price_per_stop', 'Is_weekend', 
                'Is_night_flight', 'airline', 'source', 'destination']
    
    X = df[features]

    # Make prediction
    prediction = model.predict(X)

    return render_template("index.html", prediction=f"Predicted Fare: ₹{prediction[0]:.2f}")

@app.route('/preload')
def preload_data():
    # Load some sample data
    sample_data = {
        'airline': 'IndiGo',
        'source': 'Banglore',
        'destination': 'New Delhi',
        'route': 'BLR → DEL',
        'dep_time': '22:20',
        'arrival_time': '01:10',
        'duration': '2h 50m',
        'total_stops': 'non-stop',
        'additional_info': 'No info',
        'price': 3897,
        'date_of_journey': '24/03/2019'
    }
    return render_template("index.html", **sample_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)