import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

train_dat = pd.read_csv("train.csv")
train_dat.shape
X_train = train_dat.iloc[:, 2:]
y_train = train_dat.iloc[:, 1]

y_train


def feature_map(df):
    x1 = df.iloc[0]
    x2 = df.iloc[1]
    x3 = df.iloc[2]
    x4 = df.iloc[3]
    x5 = df.iloc[4]

    phi1 = x1
    phi2 = x2
    phi3 = x3
    phi4 = x4
    phi5 = x5
    phi6 = (x1)**2
    phi7 = (x2)**2
    phi8 = (x3)**2
    phi9 = (x4)**2
    phi10 = (x5)**2
    phi11 = np.exp(x1)
    phi12 = np.exp(x2)
    phi13 = np.exp(x3)
    phi14 = np.exp(x4)
    phi15 = np.exp(x5)
    phi16 = np.cos(x1)
    phi17 = np.cos(x2)
    phi18 = np.cos(x3)
    phi19 = np.cos(x4)
    phi20 = np.cos(x5)
    phi21 = 1

    return ([phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9, phi10,
             phi11, phi12, phi13, phi14, phi15, phi16, phi17, phi18, phi19, phi20, phi21])


X_transf = []
for row in range(X_train.shape[0]):
    X_transf.append(feature_map(X_train.iloc[row, :]))


X_transformed = pd.DataFrame(X_transf)

model = LinearRegression(fit_intercept=False)
model_fitted = model.fit(X_transformed, y_train)

res_task1b = model.coef_
res_task1b = pd.DataFrame(res_task1b)

print(res_task1b)

#res_task1b.to_csv(r"C:\Users\Damja\OneDrive\Damjan\FS22\Intro_to_ML\project1\Task1b\result_task1B.csv", index = False, header = False)
