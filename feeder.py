import requests
import json
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
import re
from math import cos,radians
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import warnings
import pickle
import pytz

warnings.filterwarnings('ignore')


def update_weather():
    hourly = requests.get('http://api.wunderground.com/api/a1a0721d8c62d9ec/hourly/q/WA/Issaquah.json')
    return hourly.content

def get_ready(api_forecast):

    '''Input: jspn response from API (.content)
       Ouput: DF ready to be used for predictions'''

    prediction = json.loads(api_forecast)
    output = []
    for item in prediction['hourly_forecast']:
        output.append({'epoch':item['FCTTIME']['epoch'],
                   'hour':item['FCTTIME']['hour'],
                    'alti':item['mslp']['english'],
                    'drct': item['wdir']['degrees'],
                    'dwpf': item['dewpoint']['english'],
                    'p01i' : item ['qpf']['english'],
                    'relh':item ['humidity'],
                    'sknt':int(item['wspd']['metric'])/1.8 ,
                    'skyc1':item['sky'],
                    'tmpf':item['temp']['english']})
    TEMP = pd.DataFrame(output, columns = ['epoch','hour','alti','drct','dwpf',
                                            'p01i','relh','sknt','skyc1',
                                           'psgr','tmpf'])
    TEMP['TS'] = TEMP['epoch'].apply(ts)
    TEMP['TS'] = pd.to_datetime(TEMP['TS'])

    # Assigning as index
    TEMP.set_index('TS',inplace=True)

    # converting strings to numbers
    lst = ['alti','drct','dwpf','p01i','relh','sknt','skyc1','tmpf']
    for col in lst:
        TEMP[col] = TEMP[col].apply(pd.to_numeric, errors='coerce')

    # Aggregating in 4h periods
    lst = ['alti','drct','dwpf','p01i','relh','sknt','skyc1']
    BASE = pd.DataFrame(TEMP.tmpf.resample('4H').mean())
    for col in lst:
        ADD = pd.DataFrame(TEMP[col].resample('4H').mean())
        frames = [BASE,ADD]
        RDF = pd.concat(frames,axis=1)
        BASE = RDF

    # Creating pressure gradient
    RDF['psgr'] = pd.DataFrame(RDF['alti'].resample('4H',label='press_gr').diff())

    # resetting the index to take the timestamp out
    RDF.reset_index(inplace=True)

    #creating date and hore columns
    RDF['date']=RDF['TS'].apply(get_date)
    RDF['hour']=RDF['TS'].apply(get_hour)

    #Getting one row per day - different rows become columns
    RDF = RDF.pivot_table(index='date',
                    columns='hour',
                    values=['tmpf','dwpf','relh','drct','sknt',
                            'p01i','alti','skyc1','psgr'])
    #Flattening the multindex
    RDF = pd.DataFrame(RDF.to_records())

    #removing 'date' to make it easier
    mi = RDF.columns
    mi = list(mi)
    mi = mi[1:]

    # putting names in the same format as the model
    words = re.findall('(\w{4,10})', str(mi))
    hours = ['0','4','8','12','16','20'] * len(words)
    idx = [w + '_' + h for w,h in zip(words,hours)]
    idx = ['date']+idx
    RDF.columns = idx

    # filling holes
    for col in idx:
        RDF[col]= RDF[col].interpolate(method='linear', axis=0).ffill().bfill()

    # transforming wind direction
    direction = ['drct_0', 'drct_4', 'drct_8', 'drct_12', 'drct_16', 'drct_20']

    for col in direction:
        RDF[col]=RDF[col].apply(get_cos)

    #filters for only today and tomorrow
    tz = pytz.timezone('US/Pacific')
    today = datetime.now(tz)
    tomorrow = today + timedelta(days=1)
    RDF = RDF[(RDF['date']== today.date()) | (RDF['date']==tomorrow.date())]
    RDF.sort('date',inplace=True)
    RDF.reset_index(inplace=True, drop=True)


    #creatinf fiels for pilot rank
    RDF = pd.concat([RDF]*3, ignore_index=True)
    RDF['overall_rank_0']=0
    RDF['overall_rank_1']=0
    RDF['overall_rank_2']=0
    RDF['overall_rank_0'][0]=1
    RDF['overall_rank_0'][3]=1
    RDF['overall_rank_1'][1]=1
    RDF['overall_rank_1'][4]=1
    RDF['overall_rank_2'][2]=1
    RDF['overall_rank_2'][5]=1

    return RDF



def get_date(ts):
    return ts.date()

def get_hour(ts):
    return ts.hour

# create timespamp
def ts(epoch):
    return datetime.fromtimestamp(int(epoch)).strftime('%Y-%m-%d %H:%M:%S')


def get_cos(direction):
    return (cos(radians(direction)))**2

def load_models():

    with open('/home/ubuntu/soaring_predictor/max_alt.pkl', 'rb') as f:
        max_alt = pickle.load(f)
    with open('/home/ubuntu/soaring_predictor/good_day.pkl', 'rb') as f:
        good_day = pickle.load(f)
    with open('/home/ubuntu/soaring_predictor/XC.pkl', 'rb') as f:
        XC = pickle.load(f)
    return max_alt,good_day,XC


def label_X(RDF):

    features = ['overall_rank_0','overall_rank_1', 'overall_rank_2',
         'alti_0', 'alti_4', 'alti_8', 'alti_12', 'alti_16',
         'drct_0', 'drct_4', 'drct_8', 'drct_12', 'drct_16',
         'dwpf_0', 'dwpf_4', 'dwpf_8', 'dwpf_12', 'dwpf_16',
         'p01i_0', 'p01i_4', 'p01i_8', 'p01i_12', 'p01i_16',
         'relh_0', 'relh_4', 'relh_8', 'relh_12', 'relh_16',
         'sknt_0', 'sknt_4', 'sknt_8', 'sknt_12', 'sknt_16',
         'skyc1_0', 'skyc1_4', 'skyc1_8', 'skyc1_12', 'skyc1_16',
         'psgr_0', 'psgr_4', 'psgr_8', 'psgr_12', 'psgr_16',
         'tmpf_0', 'tmpf_4', 'tmpf_8', 'tmpf_12', 'tmpf_16']
    return RDF[features]

def predict(RDF, model):
    '''Input:
        RDF = formated date from weather api
        model = unpicked max_alt model
        Ouput: np array with 6 predictions'''

    X = label_X(RDF)
    return model.predict(X)
