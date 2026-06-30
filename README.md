Air Quality Index (AQI) Forecast

A machine learning web app that forecasts next-hour Air Quality Index (AQI) from real-time air pollutant and weather sensor readings.

Overview

This project uses the UCI Air Quality dataset — real hourly sensor data collected in an Italian city — to:


Calculate a standardized AQI value from pollutant concentrations (CO and NO2) using EPA-style breakpoint formulas
Train a Random Forest model to forecast the next hour's AQI based on current sensor readings
Serve predictions through a Flask web app with health advisory messages and trend visualizations


Tech Stack


Data processing: pandas, numpy
Model: scikit-learn (Random Forest Regressor)
Backend: Flask
Frontend: HTML, CSS, JavaScript (Chart.js for visualizations)
Deployment: Vercel


Features


Predicts AQI for the next hour (not just current conditions) using lag-based time-series forecasting
Color-coded AQI health categories (Good, Moderate, Unhealthy for sensitive groups, Unhealthy, Very unhealthy, Hazardous)
Health advisory message tailored to the predicted AQI category
Recent AQI trend chart (last 24 hours from historical data)
Pollutant breakdown chart for the submitted reading


How It Works

Input features (11 sensor readings):
CO, Benzene (C6H6), NOx, NO2, O3 sensor readings, plus temperature, relative humidity, and absolute humidity.

Target: Next-hour AQI, calculated using EPA breakpoint formulas applied to CO and NO2 concentrations, then shifted forward by one timestep to create a genuine forecasting task (not just describing current conditions).

Model performance:


R² Score: 0.71
Mean Absolute Error: ~11 AQI points


This is evaluated using a chronological train/test split (no shuffling) to properly simulate real-world forecasting, where the model only has access to past data when predicting the future.

Project Structure

aqi-forecast/
├── App.py                
├── requirements.txt       
├── vercel.json              
├── data/
│   └── AirQuality.csv      
├── model/
│   └── aqi_forecast_model.pkl  
├── templates/
│   └── index.html          


Running Locally

bashpip install -r requirements.txt
python App.py

Then open http://127.0.0.1:5000 in your browser.

Dataset Source

UCI Machine Learning Repository — Air Quality dataset, donated by S. De Vito et al., containing hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an air quality multisensor device, located in a significantly polluted area of an Italian city.

Author

Shaik Raza
