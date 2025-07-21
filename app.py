from flask import Flask, request, render_template
import pickle
import pandas as pd
from custom_transformers import *

app = Flask(__name__)

model = pickle.load(open('Day_ypred_gcv_gb_pipeline.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input',methods=['POST'])
def predict():
    # get features
    input_data = {
        'season': int(request.form.get('season')),
        'yr': int(request.form.get('yr')),
        'holiday': int(request.form.get('holiday')),
        'weekday': int(request.form.get('weekday')),
        'workingday': int(request.form.get('workingday')),
        'weathersit': int(request.form.get('weathersit')),
        'temp': float(request.form.get('temp')),
        'hum': float(request.form.get('hum')),
        'windspeed': float(request.form.get('windspeed'))
    }
    # create df
    df = pd.DataFrame([input_data])
    
    # do prediction
    res = model.predict(df)
    return str(res[0])

if __name__ == '__main__':
    app.run(debug=True)
