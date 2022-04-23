#!/usr/bin/env python
# coding: utf-8

# # Import packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 500)


# # Read in and preprocess Data

# In[7]:


# read in data
train_features = pd.read_csv("train_features.csv")
train_labels = pd.read_csv("train_labels.csv")
test_features = pd.read_csv("test_features.csv")


# In[3]:


# assign labels for the 2 subtasks
train_labels1 = train_labels.iloc[:, 1:11].to_numpy()
train_labels2 = train_labels.iloc[:, 11].to_numpy()
train_labels3 = train_labels.iloc[:, [12, 13, 14, 15]].to_numpy()

#label names
labels1 = train_labels.columns.values.tolist()[1:11]
labels2 = train_labels.columns.values.tolist()[11]
labels3 = train_labels.columns.values.tolist()[12:]
labels_all = train_labels.columns.values.tolist()


# In[4]:


print([train_features.shape, train_labels.shape]) 
# [(227940, 37), (18995, 16)], since 227'940 / 12 = 18995


# In[6]:


test_features.iloc[13321:13323, :]

test_features.columns


# In[6]:


#FROM LONG TO WIDE!

# transform all "times" to be from 1 to 12 for train and test features
#unique patient ID's 
patient_ids = set(list(train_features['pid']))
times = [x for x in range(1,13)]
times.extend(times * 18994)
train_features.loc[:,'Time'] = times

times = [x for x in range(1,13)]
times.extend(times * ((int(test_features.shape[0]/12)) - 1))
test_features.loc[:,'Time'] = times

# need col names for each variable after Time
col_names = train_features.columns.values.tolist()[2:]
#to get only 1 row per pid, need to transform data from long to wide format
train_features = train_features.pivot(index='pid', columns="Time")
test_features = test_features.pivot(index='pid', columns="Time")
# rename colnames, number each variable for the respective time point
new_colnames = [string+str(i) for string in col_names for i in range(1, 13)]
train_features.columns = train_features.columns.droplevel()
train_features.columns = new_colnames
test_features.columns = test_features.columns.droplevel()
test_features.columns = new_colnames
# needed for the submit dataframe
pid = test_features.index.to_numpy().astype(int)


# # Data Imputation

# In[11]:


# train features

# Create our imputer to replace missing values with the mean e.g.
imp_train = IterativeImputer(missing_values=np.nan, random_state = 23)
imp_train = imp_train.fit(train_features)
# Impute our data, then train
col_names = train_features.columns.values.tolist()
train_features = pd.DataFrame(imp_train.transform(train_features))
train_features.columns = col_names

# test features

# Create our imputer to replace missing values with the mean e.g.
imp_test = IterativeImputer(missing_values=np.nan, random_state = 23)
imp_test = imp_test.fit(test_features)
# Impute our data, then train
col_names = test_features.columns.values.tolist()
test_features = pd.DataFrame(imp_test.transform(test_features))
test_features.columns = col_names

missing = train_labels.isnull().sum().sum()
print("Nb. of NaN train features: ", train_features.isnull().sum().sum())
print("Nb. of NaN test features:  ", test_features.isnull().sum().sum())


# In[12]:


train_features


# # Subtask 1

# In[13]:


print(datetime.datetime.now())
clf1 = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state = 10, n_jobs = -1, class_weight = "balanced"))
clf1 = clf1.fit(train_features, train_labels1)
print(datetime.datetime.now())


# In[14]:


train_pred1 = np.array([element[:,1] for element in clf1.predict_proba(train_features)]).transpose()
for lab in range(len(labels1)):
    print(labels1[lab] + 5*" -")
    print(confusion_matrix(train_labels1[:, lab], np.around(train_pred1[:, lab])))
    print(" ")


# In[15]:


test_pred1 = np.array([element[:,1] for element in clf1.predict_proba(test_features)]).transpose()


# # Subtask 2

# In[16]:


print(datetime.datetime.now())
clf2 = RandomForestClassifier(n_estimators=50, random_state = 17, n_jobs = -1, class_weight = "balanced")
clf2 = clf2.fit(train_features, train_labels2)
print(datetime.datetime.now())


# In[17]:


train_pred2 = np.around(clf2.predict_proba(train_features)[:,1])
print(roc_auc_score(train_labels2, train_pred2))
confusion_matrix(train_labels2, train_pred2)


# In[18]:


test_pred2 = clf2.predict_proba(test_features)[:,1]


# # Subtask 3

# In[19]:


print(datetime.datetime.now())
clf3 = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state = 7, n_jobs = -1))
clf3 = clf3.fit(train_features, train_labels3)
print(datetime.datetime.now())


# In[20]:


train_pred3 = clf3.predict(train_features)
for lab in range(len(labels3)):
    print(labels3[lab] + 5*" -")
    print(r2_score(train_labels3[:, lab], train_pred3[:, lab]))
    print(" ")


# In[21]:


test_pred3 = clf3.predict(test_features)


# # Save Data



preds = pd.DataFrame(np.column_stack((pid, test_pred1, test_pred2, test_pred3)))
preds.columns = labels_all


# same format as submit file
#pd.read_csv("sample.csv")


# In[23]:


# preds is a pandas dataframe containing the result
preds.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')