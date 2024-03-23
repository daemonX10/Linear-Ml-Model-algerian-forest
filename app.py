from flask import Flask , request , jsonify , render_template 
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('./models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('./models/scaler.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        data = request.form
        # get the data from the form
        Temperature = float(data['temperature'])
        RH = float(data['rh'])
        Ws = float(data['ws'])
        Rain = float(data['rain'])
        FFMC = float(data['ffmc'])
        DMC = float(data['dmc'])
        ISI = float(data['isi'])
        Classes = data['classes']
        Region = data['region']
        
        ## Transform the data
        new_data_scaled=standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result=ridge_model.predict(new_data_scaled)
        
        return render_template('index.html', prediction_text=' The FWI Prediction is  {}'.format(result[0]))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()