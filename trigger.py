import requests
import json
import pandas as pd
import time
from datetime import datetime, timezone
import re
from math import cos,radians
import psycopg2
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import warnings
import pickle
import feeder
from feeder import update_weather,get_ready,load_models

api_forecast = update_weather()
data = get_ready(api_forecast)
max_alt,good_day,XC = load_models()
pred_max_alt = predict(data, max_alt)
pred_good_day = predict(data, good_day)
pred_XC = predict(data, XC)
print('Max altitude predictions: {}'.format(pred_max_alt))
print('Good day predictions: {}'.format(pred_good_day))
print('XC predictions: {}'.format(pred_XC))
