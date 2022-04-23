##################################################################################################
#
# Introduction to Machine Learning ETH
# Task 2 - Coding Exercise
# Author: Dejan & Sindi
# 
##################################################################################################

import pandas as pd
import numpy as np
import sklearn

from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import svm

import xgboost
from xgboost import XGBRegressor
from xgboost import XGBClassifier

import os 
os.chdir(r'C:\Users\Damja\OneDrive\Damjan\FS22\Intro_to_ML\projects\project2')


# Importing the Data
# ------------------------------------------------------------------------------------------------
train_features = pd.read_csv('train_features.csv')
train_labels = pd.read_csv('train_labels.csv')
test_features = pd.read_csv('test_features.csv')


# ------------------------------------------------------------------------------------------------
Y = train_labels.sort_values('pid')
X = train_features.sort_values(['pid', 'Time'])
X_test =  test_features.sort_values(['pid', 'Time'])

x_for_y_names = ['pid', "BaseExcess", "Fibrinogen",
		"AST", "Alkalinephos", "Bilirubin_total",
		"Lactate", "TroponinI", "SaO2", 
		"Bilirubin_direct", "EtCO2", 
		'RRate', 'ABPm', 'SpO2', 'Heartrate']
x_lag = X[x_for_y_names]
x_test_lag = X_test[x_for_y_names]


# ------------------------------------------------------------------------------------------------
shifted = x_lag.groupby('pid').shift(+1)
shifted.columns = ["BaseExcess_lag", "Fibrinogen_lag",
		"AST_lag", "Alkalinephos_lag", "Bilirubin_total_lag",
		"Lactate_lag", "TroponinI_lag", "SaO2_lag", 
		"Bilirubin_direct_lag", "EtCO2_lag",'RRate_lag', 
		'ABPm_lag', 'SpO2_lag', 'Heartrate_lag']

shifted_T = x_test_lag.groupby('pid').shift(+1)
shifted_T.columns = ["BaseExcess_lag", "Fibrinogen_lag",
		"AST_lag", "Alkalinephos_lag", "Bilirubin_total_lag",
		"Lactate_lag", "TroponinI_lag", "SaO2_lag", 
		"Bilirubin_direct_lag", "EtCO2_lag",'RRate_lag', 
		'ABPm_lag', 'SpO2_lag', 'Heartrate_lag']

X = pd.concat([X, shifted], axis = 1)
X_test = pd.concat([X_test, shifted_T], axis = 1)

X_fit = X.groupby('pid', group_keys = False).mean()
X_test_fit = X_test.groupby('pid', group_keys = False).mean()
r_pid = pd.Series(X_fit.index.copy()).astype('int')
r_pid_T = pd.Series(X_test_fit.index.copy()).astype('int')

X_fit.index = range(X_fit.shape[0])
X_test_fit.index = range(X_test_fit.shape[0])

X_fit = pd.concat([r_pid, X_fit], axis = 1)
X_test_fit = pd.concat([r_pid_T, X_test_fit], axis = 1)

# split the data
# ------------------------------------------------------------------------------------------------
X_fit = X_fit.drop(['pid', 'Time'], axis = 1)
X_test_pid = X_test_fit['pid']
X_test_fit = X_test_fit.drop(['pid', 'Time'], axis = 1)


# x_train, x_test, y_train, y_test = train_test_split(X_fit, Y, test_size = 0.2)

# names for classification
# ------------------------------------------------------------------------------------------------
n_class = ["LABEL_BaseExcess", "LABEL_Fibrinogen",
		"LABEL_AST", "LABEL_Alkalinephos", "LABEL_Bilirubin_total",
		"LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2", 
		"LABEL_Bilirubin_direct", "LABEL_EtCO2", "LABEL_Sepsis"]




i = 0
output_classifiaction = np.zeros((X_test_fit.shape[0], len(n_class)))
for y_name in n_class:

	# Model Fitting
	# ------------------------------------------------------------------------------------------------
	rf_model = XGBClassifier()
	y_fit = Y[y_name]


	rf_model.fit(X_fit, y_fit)
	y_test_p = rf_model.predict_proba(X_test_fit)
	output_classifiaction[:,i] = y_test_p[:,1]
	i = i + 1

output_classifiaction = pd.DataFrame(output_classifiaction)
output_classifiaction.columns = n_class


n_reg = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
output_regression =np.zeros((X_test_fit.shape[0], len(n_reg)))
i = 0
for y_name in n_reg:

	# Model Fitting
	# ------------------------------------------------------------------------------------------------
	rf_model = XGBRegressor()
	y_fit = Y[y_name]


	rf_model.fit(X_fit, y_fit)
	y_test_p = rf_model.predict(X_test_fit)
	output_regression[:,i] = y_test_p
	i = i + 1

output_regression = pd.DataFrame(output_regression)
output_regression.columns = n_reg

all_pred = pd.concat([X_test_pid, output_classifiaction, output_regression], axis = 1)

# get the predictions in the right order
# ------------------------------------------------------------------------------------------------
main_order = pd.read_csv('test_features.csv')
temp=main_order.groupby('pid',  group_keys = False, sort = False).mean()
r_pid = pd.Series(temp.index.copy()).astype('int')
temp.index = range(temp.shape[0])
temp = pd.concat([r_pid, temp], axis = 1)
all_pred = all_pred.set_index('pid')
all_pred = all_pred.reindex(index=temp['pid'])
all_pred = all_pred.reset_index()



print(all_pred)
all_pred.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')