import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.impute import SimpleImputer
from skmultilearn.problem_transform import ClassifierChain
import sys
import os


os.chdir(r"C:\Users\Damja\OneDrive\Damjan\FS22\Intro_to_ML\projects\project2")


test_features = pd.read_csv("test_features.csv")
train_features = pd.read_csv("train_features.csv")
train_labels = pd.read_csv("train_labels.csv")

#print(test_features.head())
#print(train_labels.head())

print([train_features.shape, train_labels.shape]) 
# [(227940, 37), (18995, 16)], since 227'940 / 12 = 18995

test_features.columns

#Data Imputation --> I don't know yet if that's necessary
train_features[train_features['pid'] == 0]

#unique patient ID's
patient_ids = set(list(train_features['pid']))
#print(patient_ids)


#i.e. for patient with ID = 1, with following data we want to predict the labels/the values
print(train_features[train_features['pid'] == 1])
print(train_labels[train_labels['pid'] == 1])
print(train_labels)

##########
#IMPUTATION#

train_features.replace('NaN',np.NaN,inplace=True)
imp = SimpleImputer(missing_values=np.NaN, strategy='median')
imp.fit(train_features)
SimpleImputer( copy=True, missing_values=np.NaN, strategy='median', verbose=0)
train_features_imp = imp.transform(train_features)

############


#pid is the patient ID
### subtask 1
# binary classification
# LABELS: LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2
# !!To achieve a good performance, it is important to produce (probabilistic) real-valued predictions in the interval [0, 1]!!

#we need the pid!?
labels1 = ["pid", "LABEL_BaseExcess", "LABEL_Fibrinogen", "LABEL_AST", "LABEL_Alkalinephos", "LABEL_Bilirubin_total", "LABEL_Lactate", "LABEL_TroponinI", "LABEL_SaO2", "LABEL_Bilirubin_direct", "LABEL_EtCO2"]
train_labels_1 = train_labels[labels1]  

print( [train_labels_1.shape, train_features.shape])
print( [train_labels_1.shape, train_features_imp.shape])


classifier = ClassifierChain(svm.SVC())
classifier.fit(train_features_imp, train_labels_1)
