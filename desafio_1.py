import pandas as pd
#import matplotlib.pylab as plt
import numpy as np
import matplotlib as plt
from matplotlib import pyplot

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
                    print("Column",column,"dropped",missing_percentage)
                    count_dropped += 1
    print(count_dropped)
    return list_dropped


def most_common_fct(df):
    """Calculates most common value.

    Args:
        df: Dataframe
    
    Returns:

    """
    mostcommon = df[column].value_counts(dropna=True).idxmax(skipna=True)
    
    print("Most Common Value: ",column,mostcommon)
    df[column].replace(np.nan,mostcommon, inplace=True)


def mean_fct(df):
    """Calculates the mean.

    Args:
        df: Dataframe
    
    Returns:

    """
    if df[column].dtype in [int, float]:
        #print("Column Type: ",df[column].dtype)
        number_mean = df[column].mean(axis=0)
        print("Mean Value: ",column,number_mean)


############# Data wrangling ############
# Import training set:
url = "train.csv"
df = pd.read_csv(url)
#df.describe()

# Drop columns with many missing values:
hard_drop_list = ['Q027','Q028','NO_MUNICIPIO_ESC','SG_UF_ESC','NO_ENTIDADE_CERTIFICACAO','TX_RESPOSTAS_CN', 'TX_RESPOSTAS_CH', 'TX_RESPOSTAS_LC', 'TX_RESPOSTAS_MT', 'TX_GABARITO_CN', 'TX_GABARITO_CH', 'TX_GABARITO_LC', 'TX_GABARITO_MT']
df.drop(hard_drop_list,axis=1,inplace=True)
print(df.columns)

enem_keep_columns = ['CO_PROVA_CN','CO_PROVA_CH','CO_PROVA_LC','CO_PROVA_MT','NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT','TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO']
list_dropped = drop_columns(df,0.2,enem_keep_columns)
print("Dropped",list_dropped)

# Replaces missing values of remaining columns with most common values:
for column in df.columns:
    #print("column",df[column].dtype)
    if column in ["NU_INSCRICAO"]: 
        continue
    else:
        if df[column].dtype in [int, float]:
            most_common_fct(df)
        elif df[column].dtype == object:
            df[column] = df[column].astype(str)
            most_common_fct(df)

#print(df)

############ ML #############



# columns_with_nan = df.columns[nan_columns_sum].tolist()