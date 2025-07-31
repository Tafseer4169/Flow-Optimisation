from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

dtr = pickle.load(open('LinearModelVehicleVSPollution.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('first.html')

@app.route('/visualize')
def input():
    return render_template('index1.html')

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        Car = request.form['Car']
        Bus = request.form['Bus']
        Truck = request.form['Truck']
        Motorcycle = request.form['Motorcycle']

        features = np.array([[Car,Bus,Truck,Motorcycle]],
                            dtype=object)
        prediction = dtr.predict(features)
        prediction = ', '.join(map(str, prediction.flatten()))


        return render_template('index1.html', prediction=prediction)










if __name__ == '__main__':
    app.run(debug=True)