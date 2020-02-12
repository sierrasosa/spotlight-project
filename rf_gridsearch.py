### IMPORTS ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

### DATA STRUCTURING ###
data = pd.read_csv('spotlight_data_volley21.csv')  #pre-cleaned anomaly dataset

features = data.drop(columns = ['user_id','event','timestamp','anomaly']) #drop unnecessary columns
labels = data['anomaly']

class_wt = int(len(labels)/labels.sum())

#Dividing into training(70%) and testing(30%)
x_train, x_test, y_train, y_test  = tts(features, labels, test_size=0.3, random_state=None)

param_grid_rf = {
    'rf__max_depth' : [4, 8, 16],

    'rf__n_estimators' : [10, 50, 100, 500, 1000]
}

pipe = Pipeline([ ( 'rf', RandomForestClassifier(class_weight='balanced') ) ])
grid_search = GridSearchCV(pipe, param_grid_rf, cv=5, scoring='recall', n_jobs=-1) #roc_auc

treeclass = RandomForestClassifier(n_estimators=1000, class_weight = 'balanced')
model = treeclass.fit(x_train, y_train)

grid_search.fit(x_train, y_train)

print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

rf = grid_search.best_estimator_

y_pred_test = grid_search.best_estimator_.predict(x_test)

print ("F1 Score:", metrics.f1_score(y_test, y_pred_test))
print ("Precision Score:", metrics.precision_score(y_test, y_pred_test))
print ("Recall Score:", metrics.recall_score(y_test, y_pred_test))
print ("Accuracy Score:", metrics.accuracy_score(y_test, y_pred_test))

print(confusion_matrix(y_true=y_test, y_pred=y_pred_test))

ax = plt.axes()
metrics.plot_roc_curve(grid_search.best_estimator_, x_test, y_test, ax = ax)

ax.plot([0,1],[0,1], linestyle='--')

metrics.plot_confusion_matrix(grid_search.best_estimator_, x_test, y_test, normalize='true', cmap = plt.cm.Blues)


#explainer = shap.TreeExplainer(model)
#shap_values = explainer.shap_values(x_test)

