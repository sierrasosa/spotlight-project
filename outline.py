### IMPORTS ###
import pandas as pd
import numpy as np
from mongo_connect import mongo_connect, mongo_execute
from sql_connect import pg_connect, pg_execute
from datetime import timedelta
from data_cleaning import unwrap_df, get_all_dummies


### MONGO COLLECTIONS ###
volley = 'GE1AW11-DXEMXYR-PZQDJC1-AD573C3'
lokal = 'ZVVPVHY-02P4ZEP-PAN7JCD-869W3GZ'
tide = '3MAM5B7-BCKMW4X-Q1N1AMD-DXYHEVH'


### INPUTS ###
# anomaly 21 for Volley is a good example
collection = volley
anomaly = 21


### SQL QUERY ###
sql_query = """
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
         AND st BETWEEN  DATE('2020-01-01') AND DATE('2020-01-20')
    ORDER BY st DESC
    """
cursor = pg_connect()
event_tbl = pg_execute(cursor, sql_query)


### GET INFO BASED ON INPUTS FOR MONGO QUERY ###
#event type
#event = event_tbl[1][anomaly]
event = 'Choice Information'
#time around anomaly
date_min = event_tbl[2][anomaly]
date_max = event_tbl[3][anomaly]

#same time last week for comparison
date_min_lw = date_min - timedelta(weeks=1)
date_max_lw = date_max - timedelta(weeks=1)


### MONGO QUERY ###
#anomaly time
mongo_query = {"type":"track", "timestamp":{'$gte':date_min, '$lte':date_max}}
projection = {'_id':0,"anonymousId":1, "timestamp":1, "event":1, "context":1, "properties":1}

#last week comparison
mongo_query_lw = {"type":"track", "timestamp":{'$gte':date_min_lw, '$lte':date_max_lw},"event":event}
projection_lw = {'_id':0,"anonymousId":1, "timestamp":1, "event":1, "context":1, "properties":1}

db = mongo_connect()
df1 = mongo_execute(db, collection, mongo_query, projection) #anomaly time
df2 = mongo_execute(db, collection, mongo_query_lw, projection_lw) #last week comparison
df = pd.concat([df1,df2])

#mark which events are part of the anomaly
df['anomaly'] = np.where((df['event'] == event) & (df['timestamp'] > date_min) & (df['timestamp'] < date_max), 1, 0)


### DATA CLEANING -- ONE COLUMN PER FEATURE ###
#make a list of the lowest level names you want as columns (eg. if you want context.app.version just put 'version')
#then use them in the unwrap function with the corresponding nest (eg. context)
context_cols =  ['version','timezone']
df = unwrap_df(df, 'context', context_cols)

properties_cols =['days_active_count','most_recent_session_start_time','time_since_previous_session_start','user_lifetime_session_count','platform','characterTraits','shows_subscribed']
df = unwrap_df(df, 'properties', properties_cols)


### DATA CLEANING -- NAN REMOVAL ###
#I know that volley has a lot of api calls causing NANs and these lines help remove them
#this isn't really generalizable to other companies but helps for the model proof-of-concept
df = df[(df['version'] == '19.545') | (df['version'] == '22.553')]
df = df[df['user_lifetime_session_count'] > 0]

percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})

print(missing_value_df)




### DATA CLEANING -- DUMMY VARIABLES ###
#the model can't handle variables that aren't coded (like timezone) so this codes them correctly
#normal_cols = ['version','timezone']
#list_cols = ['characterTraits']
#df = get_all_dummies(df, normal_cols, list_cols)




#df.describe()
print(df.info())


















