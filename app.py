from flask import Flask, request, render_template
import joblib
import pandas as pd

pipeline = joblib.load('random_forest_price_prediction.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    user_input = {
        'from' : request.form['from'],
        'to' : request.form['to'],
        'flightType' : request.form['flightType'],
        'agency' : request.form['agency']
    }

    input_df = pd.DataFrame([user_input])

    prediction = pipeline.predict(input_df)[0]
    rounded_prediction = round(prediction)

    return render_template('index.html', prediction = rounded_prediction, user_from = user_input['from'],
                           user_to = user_input['to'], user_flighttype = user_input['flightType'], user_agency = user_input['agency'])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
