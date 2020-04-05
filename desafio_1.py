import pandas as pd
#import matplotlib.pylab as plt
import numpy as np
import matplotlib as plt
from matplotlib import pyplot
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder



def drop_columns(df,percentage,skip_list):

    """Drop columns according to percentage of NaN values.

    Args: 
        df: dataframe
        percentage: maximum percentage value accepted. If above, column will be dropped
        skip_list: does not drop columns from this list

    Returns:
        Dropped column names.

    """
    missing_data = df.isnull() #boolean value indicating whether the value that is passed into the argument is in fact missing data.
    count_dropped = 0
    list_dropped = []
    for column in missing_data.columns.values.tolist():
        if column in skip_list:
            continue
        else:
            missing_data_eval = missing_data[column].value_counts()
            if len(missing_data_eval.keys()) == 2:
                missing_percentage = missing_data_eval[True]/(missing_data_eval[True]+missing_data_eval[False])
                if missing_percentage > percentage:
                    list_dropped.append(column)
                    df.drop([column], axis=1,inplace=True)
                    #print("Column",column,"dropped",missing_percentage)
                    count_dropped += 1
    #print(count_dropped)
    return list_dropped


def most_common_fct(df):
    """Calculates most common value.

    Args:
        df: Dataframe
    
    Returns:

    """
    mostcommon = df[column].value_counts(dropna=True).idxmax(skipna=True)
    
    #print("Most Common Value: ",column,mostcommon)
    df[column].replace(np.nan,mostcommon, inplace=True)


############# Data wrangling ############

# Import training set:
train_csv = "train.csv"
df_train = pd.read_csv(train_csv)

# Import test set:
test_csv = "test.csv"
df_test = pd.read_csv(test_csv)
#df_train.describe()

# Drop columns with many missing values:
harddrop_list_train = ['Q027','Q028','NO_MUNICIPIO_ESC','SG_UF_ESC','NO_ENTIDADE_CERTIFICACAO','TX_RESPOSTAS_CN', 'TX_RESPOSTAS_CH', 'TX_RESPOSTAS_LC', 'TX_RESPOSTAS_MT', 'TX_GABARITO_CN', 'TX_GABARITO_CH', 'TX_GABARITO_LC', 'TX_GABARITO_MT']
enem_keep_columns = ['CO_PROVA_CN','CO_PROVA_CH','CO_PROVA_LC','CO_PROVA_MT','NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT','TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO']

df_train.drop(harddrop_list_train,axis=1,inplace=True)
#print(df_train.columns)
list_dropped = drop_columns(df_train,0.2,enem_keep_columns)
#print("Dropped",list_dropped)

# Making sure both datasets have the same columns (this could be improved and should be on top of the code)
for column in df_test.columns:
    if column not in df_train.columns:
        df_test.drop([column], axis=1,inplace=True)

for column in df_train.columns:
    if column in df_test.columns or column == 'NU_NOTA_MT':
        continue
    else:
        df_train.drop([column], axis=1,inplace=True)

#print(df_train.columns)

# Store IDs in new DataFrame:
df_answer = pd.DataFrame({})
df_answer['NU_INSCRICAO'] = df_test['NU_INSCRICAO'].values

# Replaces missing values of remaining columns with most common values
# also drops NU_inscricao
for data in [df_train,df_test]:
    for column in data.columns:
        #print("column",df_train[column].dtype)
        if column == "NU_INSCRICAO": 
            data.drop([column], axis=1,inplace=True)
            continue
        else:
            if data[column].dtype in [int, float]:
                most_common_fct(data)
            elif data[column].dtype == object:
                data[column] = data[column].astype(str)
                most_common_fct(data)

# Deal with conversion errors:
le = LabelEncoder()
for column in df_train.columns:
    if df_train[column].dtype not in [int, float]:
        le.fit(df_train[column].astype(str))
        df_train[column] = le.transform(df_train[column].astype(str))
        df_test[column] = le.transform(df_test[column].astype(str))
        #print(column,data[column].dtype)


############ ML #############

# Prediction Target (y):
train_y = df_train.NU_NOTA_MT


# Features (X):
columns_test = df_test.columns.tolist()
#print(columns_test)
columns_train = df_train.columns.tolist()

columns_train.remove('NU_NOTA_MT')

train_X = df_train[columns_train]
test_X = df_test[columns_test]

# Define model. Specify a number for random_state to ensure same results each run:
enem_model = DecisionTreeRegressor(random_state=1)

# Fit model
enem_model.fit(train_X, train_y)

# Predict:
df_answer['NU_NOTA_MT'] = enem_model.predict(test_X)

#Check:
print('The predictions are')
print(df_answer)

df_answer.to_csv('answer.csv')