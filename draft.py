import pandas as pd
#import matplotlib.pylab as plt
import numpy as np
import matplotlib as plt
from matplotlib import pyplot
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder


train_csv = "train.csv"
df_train = pd.read_csv(train_csv)

list_nograde = []
for result in df_train.NU_NOTA_MT:
    if np.isnan(result) == True:
        list_nograde.append(False)
    else:
        list_nograde.append(True)

filtered = pd.Series(list_nograde)
df_train = df_train[filtered]

df_train.to_csv('testing.csv',index=False)