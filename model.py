import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# df = pd.read_json('data/data.json')

def fit_gbrmodel(df):
    X_train, X_test, y_train, y_test=feature_selection(df)
    gbr=GradientBoostingRegressor()
    gbrmodel=gbr.fit(X_train,y_train)
    accuracy=gbrmodel.score(X_test,y_test)
    return print ("accuracy score is {}".format(accuracy))

def fit_rfrmodel(df):
    X_train, X_test, y_train, y_test=feature_selection(df)
    model = RandomForestRegressor()
    model.fit(X_train,y_train)
    accuracy= model.score(X_test,y_test)
    return print ("accuracy score is {}".format(accuracy))


def fit_linear(df):
    X_train, X_test, y_train, y_test=feature_selection(df)
    model = LinearRegression()
    model.fit(X_train,y_train)
    accuracy= model.score(X_test,y_test)
    return print ("accuracy score is {}".format(accuracy))



def feature_selection(df):
    x_df=df[[ 'alti_0', 'alti_4', 'alti_8', 'alti_12',
       'alti_16', 'drct_0', 'drct_4', 'drct_8', 'drct_12',
       'drct_16', 'dwpf_0', 'dwpf_4', 'dwpf_8', 'dwpf_12',
       'dwpf_16', 'p01i_0', 'p01i_4', 'p01i_8', 'p01i_12',
       'p01i_16', 'relh_0', 'relh_4', 'relh_8', 'relh_12',
       'relh_16', 'sknt_0', 'sknt_4', 'sknt_8', 'sknt_12',
       'sknt_16', 'skyc1_0', 'skyc1_4', 'skyc1_8', 'skyc1_12',
       'skyc1_16', 'tmpf_0', 'tmpf_4', 'tmpf_8', 'tmpf_12',
       'tmpf_16']]
    y_df=df['max_alt']
    X=x_df.as_matrix()
    y=np.asarray(y_df,dtype="int_")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test
