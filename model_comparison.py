### IMPORTS ###
import random
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as stat # statistical models (including regression) 
import statsmodels.formula.api as smf # R-like model specification 

import xgboost
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
#from xgboost import XGBClassifier



data = pd.read_csv('spotlight_data_volley21.csv')

features = data.drop(columns = ['user_id','event','timestamp','anomaly'])
labels = data['anomaly']


class_wt = int(len(labels)/labels.sum())

#Dividing into training(70%) and testing(30%)
x_train, x_test, y_train, y_test  = tts(features, labels, test_size=0.3, random_state=None)

balanced_class_ratio = float((y_train==0).sum())/(y_train==1).sum()

param_grid_logistic = {

    'logistic__C': np.logspace(-4, 4, 4),
    'logistic__solver' : [ 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

    'logistic__max_iter' : [500, 1000, 2000]
}
param_grid_rf = {
    'rf__max_depth' : [4, 8, 16],

    'rf__n_estimators' : [10, 50, 100, 500, 1000]
}
param_grid_xgb = {
    'xgb__minchild_weight': [1, 5, 10],
    'xgb__gamma': [0.5, 1, 1.5, 2, 5],
    #'xgb__subsample': [0.6, 0.8, 1.0],

    'xgb__colsample_bytree': [0.6, 0.8, 1.0],
    'xgb__learning_rate': [0.01, 0.02, 0.05, 0.1],

    'xgb__max_depth': [3, 4, 5]

    }


pipe = Pipeline([ ( 'rf', RandomForestClassifier(class_weight='balanced') ) ])
grid_search = GridSearchCV(pipe, param_grid_rf, cv=5, scoring='f1', n_jobs=-1) #roc_auc


grid_search.fit(x_train, y_train)

print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
print(grid_search.best_params_)

y_pred_test = grid_search.best_estimator_.predict(x_test)

print ("F1 Score:", metrics.f1_score(y_test, y_pred_test))
print ("Precision Score:", metrics.precision_score(y_test, y_pred_test))
print ("Recall Score:", metrics.recall_score(y_test, y_pred_test))
print ("Accuracy Score:", metrics.accuracy_score(y_test, y_pred_test))

print(confusion_matrix(y_true=y_test, y_pred=y_pred_test))

ax = plt.axes()
metrics.plot_roc_curve(grid_search.best_estimator_, x_test, y_test, ax = ax)

ax.plot([0,1],[0,1], linestyle='--')

metrics.plot_confusion_matrix(grid_search.best_estimator_, x_test, y_test, normalize='true')