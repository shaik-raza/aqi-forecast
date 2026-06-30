from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the next-hour AQI forecasting model
model = joblib.load('model/aqi_forecast_model.pkl')

# Must match the exact order of features used during training
FEATURES = ['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
            'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']


def get_aqi_info(aqi):
    """Standard EPA-style AQI health category, color, and advice."""
    if aqi <= 50:
        return "Good", "#22c55e", "Air quality is satisfactory. Enjoy outdoor activities as normal."
    elif aqi <= 100:
        return "Moderate", "#eab308", "Air quality is acceptable. Unusually sensitive individuals should consider reducing prolonged outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy for sensitive groups", "#f97316", "People with respiratory or heart conditions, children, and older adults should limit prolonged outdoor exertion."
    elif aqi <= 200:
        return "Unhealthy", "#ef4444", "Everyone may begin to experience health effects. Limit outdoor activity, especially strenuous exercise."
    elif aqi <= 300:
        return "Very unhealthy", "#a855f7", "Health alert: avoid outdoor activity. Sensitive groups should remain indoors."
    else:
        return "Hazardous", "#7f1d1d", "Health emergency. Everyone should avoid all outdoor exertion and stay indoors if possible."


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form[feature]) for feature in FEATURES]
        input_array = np.array(values).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        prediction = round(prediction, 1)

        status, color, advice = get_aqi_info(prediction)

        # Pollutant breakdown for the bar chart (raw GT readings entered by user)
        pollutants = {
            "CO": values[FEATURES.index('PT08.S1(CO)')],
            "Benzene": values[FEATURES.index('C6H6(GT)')],
            "NOx": values[FEATURES.index('NOx(GT)')],
            "NO2": values[FEATURES.index('NO2(GT)')],
        }

        return render_template('index.html', prediction=prediction, status=status,
                                color=color, advice=advice, pollutants=pollutants)

    except Exception as e:
        return render_template('index.html', error=str(e))


@app.route('/history')
def history():
    """Returns the last 24 AQI values from the dataset for the trend chart."""
    try:
        df = pd.read_csv('data/AirQuality.csv', sep=';', decimal=',')
        df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], errors='ignore')
        df = df.replace(-200, np.nan).dropna()

        def calculate_aqi(concentration, breakpoints):
            for low_c, high_c, low_aqi, high_aqi in breakpoints:
                if low_c <= concentration <= high_c:
                    return ((high_aqi - low_aqi) / (high_c - low_c)) * (concentration - low_c) + low_aqi
            return None

        co_breakpoints = [(0, 1, 0, 50), (1, 2, 51, 100), (2, 10, 101, 150),
                           (10, 17, 151, 200), (17, 34, 201, 300), (34, 50, 301, 500)]
        no2_breakpoints = [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 150),
                            (180, 280, 151, 200), (280, 400, 201, 300), (400, 1000, 301, 500)]

        df['AQI_CO'] = df['CO(GT)'].apply(lambda x: calculate_aqi(x, co_breakpoints))
        df['AQI_NO2'] = df['NO2(GT)'].apply(lambda x: calculate_aqi(x, no2_breakpoints))
        df['AQI'] = df[['AQI_CO', 'AQI_NO2']].max(axis=1)

        recent = df['AQI'].tail(24).round(1).tolist()
        labels = [f"-{len(recent) - i}h" for i in range(len(recent))]

        return jsonify({"labels": labels, "values": recent})

    except Exception as e:
        return jsonify({"labels": [], "values": [], "error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)