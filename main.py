from dis import dis
from time import time
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    longhair = request.form['longhair']
    fwidth = request.form['fwidth']
    fheight = request.form['fheight']
    nosewide = request.form['nosewide']
    noselong = request.form['noselong']
    lipsthin = request.form['lipsthin']
    disntp = request.form['disntp']
    features = (longhair, fwidth, fheight, nosewide, noselong, lipsthin, disntp)
    c = [float(i) for i in features]
    print(c)
    final_features = [np.array(c)]
    prediction = model.predict(final_features)

    if prediction[0] == 1:
        pred = 'MALE'
    else:
        pred = 'FEMALE'

    return render_template('index.html', prediction_text=pred)

if __name__ == "__main__":
    app.run(debug=True)
