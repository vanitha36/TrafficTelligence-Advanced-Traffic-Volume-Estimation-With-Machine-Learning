from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        year = float(request.form['year'])
        month = float(request.form['month'])
        day = float(request.form['day'])
        hour = float(request.form['hour'])
        minute = float(request.form['minute'])
        weather = request.form['weather']
        holiday = request.form['holiday']

        weather_list = ['Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist',
                        'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm']
        weather_dummies = [1 if weather == w else 0 for w in weather_list]

        holiday_list = [
            'Columbus Day', 'Independence Day', 'Labor Day',
            'Martin Luther King Jr Day', 'Memorial Day',
            'New Years Day', 'State Fair', 'Thanksgiving Day',
            'Veterans Day', 'Washingtons Birthday'
        ]
        holiday_dummies = [1 if holiday == h else 0 for h in holiday_list]

        final_input = np.array([[
            temp, rain, snow, year, month, day, hour, minute
        ] + weather_dummies + holiday_dummies])

        print("Input shape:", final_input.shape)
        print("Input:", final_input)

        final_input_scaled = scale.transform(final_input)
        prediction = model.predict(final_input_scaled)

        print("Prediction:", prediction)

        return render_template("result.html",
                               prediction_text=str(round(prediction[0], 2)))
    except Exception as e:
        print("Error:", e)
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)