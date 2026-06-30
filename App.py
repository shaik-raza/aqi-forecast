from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model once when the app starts
model = joblib.load('model/co_predictor.pkl')

# Must match the exact order of features used during training
FEATURES = ['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
            'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form[feature]) for feature in FEATURES]
        input_array = np.array(values).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        prediction = round(prediction, 2)

        if prediction < 2:
            status = "Good"
        elif prediction < 4:
            status = "Moderate"
        elif prediction < 6:
            status = "Unhealthy"
        else:
            status = "Hazardous"

        return render_template('index.html', prediction=prediction, status=status)

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)