from __future__ import division
from flask import Flask, render_template, request, jsonify
import requests
import json
import pandas as pd
import time
from datetime import datetime, timezone
import re
from math import cos,radians
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import warnings
import pickle
import feeder
from feeder import update_weather,get_ready,load_models,predict,predict_prob

app = Flask(__name__)
api_forecast = update_weather()
time.sleep(10)
data = get_ready(api_forecast)
max_alt,good_day,XC = load_models()
pred_max_alt = predict(data, max_alt)
pred_good_day = predict_prob(data, good_day)
pred_XC = predict_prob(data, XC)
dates = [str(i) for i in data['date'].as_matrix()]
experience = ['Beginner','Intermediate','Expert']*2
docs = []
for i in range(len(dates)):
    # docs.append(dates[i])
    # docs.append(experience[i])
    # docs.append(pred_max_alt[i])
    # docs.append(pred_good_day[:,1][i])
    # docs.append(pred_XC[:,1][i])
    docs.append([dates[i],experience[i],int(pred_max_alt[i]),
                 round(pred_good_day[i][1],2),round(pred_XC[i][1],2)])
# print(docs)
# print('Max altitude predictions: {}'.format(pred_max_alt))
# print('Good day predictions: {}'.format(pred_good_day))
# print('XC predictions: {}'.format(pred_XC))

@app.route('/')
def index():
    return render_template('forecast.html', docs = docs)

if __name__ == "__main__":
    app.run()
