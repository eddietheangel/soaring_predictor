import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold

# df = pd.read_json('data/data.json')



def data_split(data):
    x_df=data[[ 'rank_alt_0', 'rank_alt_1', 'rank_alt_2',
            'overall_rank_0','overall_rank_1', 'overall_rank_2',
            'pilot_rank_0', 'pilot_rank_1','pilot_rank_2',
            'alti_0', 'alti_4', 'alti_8', 'alti_12', 'alti_16',
            'drct_0', 'drct_4', 'drct_8', 'drct_12', 'drct_16',
            'dwpf_0', 'dwpf_4', 'dwpf_8', 'dwpf_12', 'dwpf_16',
            'p01i_0', 'p01i_4', 'p01i_8', 'p01i_12', 'p01i_16',
            'relh_0', 'relh_4', 'relh_8', 'relh_12', 'relh_16',
            'sknt_0', 'sknt_4', 'sknt_8', 'sknt_12', 'sknt_16',
            'skyc1_0', 'skyc1_4', 'skyc1_8', 'skyc1_12', 'skyc1_16',
            'psgr_0', 'psgr_4', 'psgr_8', 'psgr_12', 'psgr_16',
            'tmpf_0', 'tmpf_4', 'tmpf_8', 'tmpf_12', 'tmpf_16']]
    y_df=data['max_alt']
    X=x_df.as_matrix()
    y=np.asarray(y_df,dtype="int_")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # y_train = soft_class(y_train)
    # y_test = soft_class(y_test)
    return X_train, X_test, y_train, y_test

def soft_class(predictions):
    #Converts predictions [in meters] to class [thousands of ft]
    alt_range = predictions * (3.28/1000)
    alt_range = np.rint(alt_range)
    alt_range = alt_range.astype(int)
    return alt_range

def cv_test(data):


    kf = KFold(n_splits=5)
    train_list = []
    test_list = []

    # GBC_grid = {'learning_rate': 0.05,'max_depth': 2,'max_features': 6,
    #             'min_samples_leaf': 3,'min_samples_split': 4,
    #             'subsample': 0.75}
    # m = GradientBoostingClassifier(** GBC_grid)

    # param = {'max_depth': 5, 'max_features': 7,'min_samples_leaf': 4,
             # 'min_samples_split': 3, 'n_estimators': 100}
    m = RandomForestClassifier()

    features = [ 'rank_alt_0', 'rank_alt_1', 'rank_alt_2',
            'overall_rank_0','overall_rank_1', 'overall_rank_2',
            'pilot_rank_0', 'pilot_rank_1','pilot_rank_2',
            'alti_0', 'alti_4', 'alti_8', 'alti_12', 'alti_16',
            'drct_0', 'drct_4', 'drct_8', 'drct_12', 'drct_16',
            'dwpf_0', 'dwpf_4', 'dwpf_8', 'dwpf_12', 'dwpf_16',
            'p01i_0', 'p01i_4', 'p01i_8', 'p01i_12', 'p01i_16',
            'relh_0', 'relh_4', 'relh_8', 'relh_12', 'relh_16',
            'sknt_0', 'sknt_4', 'sknt_8', 'sknt_12', 'sknt_16',
            'skyc1_0', 'skyc1_4', 'skyc1_8', 'skyc1_12', 'skyc1_16',
            'psgr_0', 'psgr_4', 'psgr_8', 'psgr_12', 'psgr_16',
            'tmpf_0', 'tmpf_4', 'tmpf_8', 'tmpf_12', 'tmpf_16']

    X = data[features].as_matrix()
    y = np.asarray(data['left_perimeter'],dtype="int_")

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # y_train = soft_class(y_train)
        # y_test = soft_class(y_test)
        m.fit(X_train,y_train)
        train_score = m.score(X_train,y_train)
        test_score = m.score(X_test,y_test)
        train_list.append(train_score)
        test_list.append(test_score)
        print(' Train score: {}'.format(train_score))
        print(' Test score: {}'.format(test_score))
    print(' Final Train score: {}'.format(np.mean(train_list)))
    print(' Final Test score: {}'.format(np.mean(test_list)))
