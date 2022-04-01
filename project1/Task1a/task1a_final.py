import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge

reg_params = [0.1, 1, 10, 100, 200]
reg_params = np.array(reg_params)

train_dat = pd.read_csv("train.csv")
train_dat.shape
X_train = train_dat.iloc[:, 1:]
y_train = train_dat.iloc[:, 0]

res = list(range(5))

for i in range(5):
    model = Ridge(alpha = reg_params[i]) 
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)

    #evaluation
    scores = cross_val_score(model, X_train, y_train, scoring = "neg_root_mean_squared_error", cv = cv)

    res[i] = sum(scores * -1)/10


## save dataframe to csv
print(res)
res = pd.DataFrame(res)
#res.to_csv(r"C:\Users\Damja\OneDrive\Damjan\FS22\Intro_to_ML\project1\Task1a\result_task1A.csv", index = False, header = False)