### IMPORTS ###
import os
import pandas as pd
import json
import numpy as np
from bson import json_util 
from pymongo import MongoClient
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import patsy
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import statsmodels.api as stat # statistical models (including regression) 
import statsmodels.formula.api as smf # R-like model specification 
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.inspection import permutation_importance
import random
from sklearn.metrics import plot_roc_curve


### MONGO COLLECTIONS WITH ###

volley = 'GE1AW11-DXEMXYR-PZQDJC1-AD573C3'
lokal = 'ZVVPVHY-02P4ZEP-PAN7JCD-869W3GZ'
tide = '3MAM5B7-BCKMW4X-Q1N1AMD-DXYHEVH'


### SQL CONNECT ###
def pg_connect():
    prod_user = 'ids_2020'
    prod_database = 'dcj43q99j16fub'
    prod_port = '5432'
    prod_pwd = 'p76c1b1e9ac953732b048d6eca1fd59820f53a27e97d8a7289da0ab9b0804ec5c'
    prod_host = 'ec2-54-221-81-132.compute-1.amazonaws.com'
    connection = psycopg2.connect(user=prod_user, password=prod_pwd, host=prod_host, port=prod_port, database=prod_database)
    cursor = connection.cursor()
    return cursor

def pg_execute(cursor, query_str):
    cursor.execute(query_str)
    return pd.DataFrame(cursor.fetchall())




### SQL QUERY ###
tbl = """
SELECT o.name, m.name, st, en, en-st AS dur

FROM (SELECT s.anomaly_id, st, en 
        FROM (SELECT anomaly_id, occurred_at AS st 
                FROM anomaly_event WHERE event = ('STARTED')) AS s
        JOIN (SELECT anomaly_id, occurred_at AS en
                FROM anomaly_event WHERE event = ('AUTO_CLOSED')) AS e ON s.anomaly_id = e.anomaly_id ) AS ae
    JOIN anomaly AS an ON ae.anomaly_id = an.id
    JOIN metric AS m ON  an.metric_id = m.id
    JOIN app AS ap ON  m.app_id = ap.id
    JOIN organization AS o ON ap.organization_id = o.id
WHERE
    m.type = 'track'
     AND o.name = 'volleythat.com'
 
ORDER BY st DESC
 
"""

cursor = pg_connect()
tbl = pg_execute(cursor, tbl)
tbl.head(3)








### INPUTS ###

# anomaly 21 for Volley is a good example
collection = volley
anomaly = 21

event = tbl[1][anomaly]
date_min = tbl[2][anomaly]
date_max = tbl[3][anomaly]



### DATA CLEANING -- FIRST PASS ###

#NO DUMMIES
#numbers
df['user_id'] = df['anonymousId'] 
df['days_active'] = df.apply(lambda row: row['properties'].get('days_active_count',np.nan), axis = 1) 
df['days_active_mobile'] = df.apply(lambda row: row['properties'].get('days_active_mobile_count',np.nan), axis = 1) 
df['session_st'] = df.apply(lambda row: row['properties'].get('most_recent_session_start_time',np.nan), axis = 1) 

df['time_since_prev_sess'] = df.apply(lambda row: row['properties'].get('time_since_previous_session_start',0), axis = 1) 
df['lifetime_sess'] = df.apply(lambda row: row['properties'].get('user_lifetime_session_count',0), axis = 1) 
df['lifetime_mobile_sess'] = df.apply(lambda row: row['properties'].get('user_lifetime_mobile_session_count',np.nan), axis = 1) 


#NEED DUMMIES
#single item
df['app_vsn'] = df.apply(lambda row: row['context'].get('app',{}).get('version',np.nan), axis = 1)
df['tzone'] = df.apply(lambda row: row['context'].get('timezone','Google'), axis = 1) 
df['platform'] = df.apply(lambda row: row['properties'].get('platform',np.nan), axis = 1) 

##lists
df['char_list_cohort'] = df.apply(lambda row: row['properties'].get('characterLists',{}).get('COHORTS',np.nan), axis = 1) 
df['char_list_shows'] = df.apply(lambda row: row['properties'].get('characterLists',{}).get('SHOWS',np.nan), axis = 1) 
df['char_traits'] = df.apply(lambda row: row['properties'].get('characterTraits',np.nan), axis = 1) 
df['cohort_ids'] = df.apply(lambda row: row['properties'].get('cohort_ids',np.nan), axis = 1)
df['shows_subbed'] = df.apply(lambda row: row['properties'].get('shows_subscribed',np.nan), axis = 1) 

df = df[(df['app_vsn'] == '19.545') | (df['app_vsn'] == '22.553')]
df = df[df['lifetime_sess'] > 0]





### DATA CLEANING -- SECOND PASS ###

cln = pd.DataFrame()

#numbers
cln['days_active'] = df['days_active']
cln['days_active_mobile'] = df['days_active_mobile'] 
cln['session_st'] = df['session_st']
cln['time_since_prev_sess'] = df['time_since_prev_sess'] 
cln['lifetime_sess'] = df['lifetime_sess'] 
cln['lifetime_mobile_sess'] = df['lifetime_mobile_sess'] 
cln['tzone'] = df['tzone'] 
cln['platform'] = df['platform'] 
cln['show_sub'] = df[['char_list_shows']].notnull().astype('int')
cln['RANDOM'] = np.random.randint(0, 2, cln.shape[0]) #to verify that the feature extraction is working properly


cln = pd.get_dummies(cln)

#special treatment for list vars
ct = pd.Series(df['char_traits'])
ct = pd.get_dummies(ct.apply(pd.Series).stack()).sum(level=0)




### DATA CLEANING -- FINAL PASS ###

fcln = pd.DataFrame()

#dataframes
fcln = cln.merge(ct, 'left', left_index=True, right_index=True)

#columns
fcln['user_id'] = df['anonymousId'] 
fcln['event'] = df['event']
fcln['timestamp'] = df['timestamp']



fcln['anomaly'] = np.where((fcln['event'] == event) & (fcln['timestamp'] > date_min) & (fcln['timestamp'] < date_max), 1, 0)
#test = fcln.dropna()
#fcln.to_csv('data_export6.csv')

class_wt = len(fcln['anomaly'])/fcln['anomaly'].sum()
class_wt



features = fcln.drop(columns = ['user_id','event','timestamp','anomaly'])
labels = fcln['anomaly']

class_wt = int(len(labels)/labels.sum())
class_wt


#Dividing into training(70%) and testing(30%)
x_train, x_test, y_train, y_test  = tts(features1, labels, test_size=0.3, random_state=None)



#Running new regression on training data
treeclass = RandomForestClassifier(n_estimators=100, oob_score = True, class_weight={0:1,1:class_wt})
#treeclass = RandomForestClassifier(n_estimators=100, oob_score=True)
treeclass.fit(x_train, y_train)
#Calculating the accuracy of the training model on the testing data
y_pred = treeclass.predict(x_test)
#y_pred_prob_res = treeclass.predict_proba(x_test_res)
accuracy = treeclass.score(x_test, y_test)
accuracytr = treeclass.score(x_train, y_train)
print('The test accuracy is: ' + str(accuracy *100) + '%')
print('The train accuracy is: ' + str(accuracytr *100) + '%')
