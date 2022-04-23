import pandas as pd

test_features = pd.read_csv("test_features.csv")
train_features = pd.read_csv("train_features.csv")
train_labels = pd.read_csv("train_labels.csv")

#print(test_features.head())
#print(train_labels.head())

print([test_features.shape, train_labels.shape])



#pid is the patient ID
### subtask 1
# binary classification
# LABELS: LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2
# !!To achieve a good performance, it is important to produce (probabilistic) real-valued predictions in the interval [0, 1]!!



### subtask 2
# binary classification
# LABEL: LABEL_Sepsis


### subtask 3
# regression task
# LABELS: LABEL_RRate, LABEL_ABPm, LABEL_SpO2, LABEL_Heartrate.