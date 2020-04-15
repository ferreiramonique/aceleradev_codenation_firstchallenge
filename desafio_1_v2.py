import pandas as pd
#import matplotlib.pylab as plt
import numpy as np
import matplotlib as plt
from matplotlib import pyplot



from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler #new
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', None)

############# Functions #############

def most_common_fct(df):
    """Calculates most common value.

    Args:
        df: Dataframe
    
    Returns:

    """
    mostcommon = df[column].value_counts(dropna=True).idxmax(skipna=True)
    
    #print("Most Common Value: ",column,mostcommon)
    df[column].replace(np.nan,mostcommon, inplace=True)


############# Dataset import ############

# Import training set:
train_csv = "train.csv"
df_train = pd.read_csv(train_csv, dtype={col: np.float32 for col in ['NU_NOTA_MT', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO']}).dropna()

# Import test set:
test_csv = "test.csv"
df_test = pd.read_csv(test_csv, dtype={col: np.float32 for col in ['NU_NOTA_MT', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO']})
#df_train.describe()

############# Data wrangling ############

# Drop unwanted columns:
harddrop_list_train = ['NU_ANO', 'CO_MUNICIPIO_RESIDENCIA',
       'NO_MUNICIPIO_RESIDENCIA', 'CO_UF_RESIDENCIA', 'SG_UF_RESIDENCIA',
       'NU_IDADE', 'TP_SEXO', 'TP_ESTADO_CIVIL', 'TP_COR_RACA',
       'TP_NACIONALIDADE', 'CO_MUNICIPIO_NASCIMENTO',
       'NO_MUNICIPIO_NASCIMENTO', 'CO_UF_NASCIMENTO', 'SG_UF_NASCIMENTO',
       'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO',
       'IN_TREINEIRO', 'CO_ESCOLA', 'CO_MUNICIPIO_ESC',
       'NO_MUNICIPIO_ESC', 'CO_UF_ESC', 'SG_UF_ESC',
       'TP_DEPENDENCIA_ADM_ESC', 'TP_LOCALIZACAO_ESC', 'TP_SIT_FUNC_ESC',
       'IN_BAIXA_VISAO', 'IN_CEGUEIRA', 'IN_SURDEZ',
       'IN_DEFICIENCIA_AUDITIVA', 'IN_SURDO_CEGUEIRA',
       'IN_DEFICIENCIA_FISICA', 'IN_DEFICIENCIA_MENTAL',
       'IN_DEFICIT_ATENCAO', 'IN_DISLEXIA', 'IN_DISCALCULIA',
       'IN_AUTISMO', 'IN_VISAO_MONOCULAR', 'IN_OUTRA_DEF', 'IN_SABATISTA',
       'IN_GESTANTE', 'IN_LACTANTE', 'IN_IDOSO',
       'IN_ESTUDA_CLASSE_HOSPITALAR', 'IN_SEM_RECURSO', 'IN_BRAILLE',
       'IN_AMPLIADA_24', 'IN_AMPLIADA_18', 'IN_LEDOR', 'IN_ACESSO',
       'IN_TRANSCRICAO', 'IN_LIBRAS', 'IN_LEITURA_LABIAL',
       'IN_MESA_CADEIRA_RODAS', 'IN_MESA_CADEIRA_SEPARADA',
       'IN_APOIO_PERNA', 'IN_GUIA_INTERPRETE', 'IN_MACA', 'IN_COMPUTADOR',
       'IN_CADEIRA_ESPECIAL', 'IN_CADEIRA_CANHOTO',
       'IN_CADEIRA_ACOLCHOADA', 'IN_PROVA_DEITADO', 'IN_MOBILIARIO_OBESO',
       'IN_LAMINA_OVERLAY', 'IN_PROTETOR_AURICULAR', 'IN_MEDIDOR_GLICOSE',
       'IN_MAQUINA_BRAILE', 'IN_SOROBAN', 'IN_MARCA_PASSO', 'IN_SONDA',
       'IN_MEDICAMENTOS', 'IN_SALA_INDIVIDUAL', 'IN_SALA_ESPECIAL',
       'IN_SALA_ACOMPANHANTE', 'IN_MOBILIARIO_ESPECIFICO',
       'IN_MATERIAL_ESPECIFICO', 'IN_NOME_SOCIAL', 'IN_CERTIFICADO',
       'NO_ENTIDADE_CERTIFICACAO', 'CO_UF_ENTIDADE_CERTIFICACAO',
       'SG_UF_ENTIDADE_CERTIFICACAO', 'CO_MUNICIPIO_PROVA','NO_MUNICIPIO_PROVA', 'CO_UF_PROVA', 'SG_UF_PROVA',
       'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC',
       'TP_PRESENCA_MT', 'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC',
       'CO_PROVA_MT',
        'TX_RESPOSTAS_CN', 'TX_RESPOSTAS_CH',
       'TX_RESPOSTAS_LC', 'TX_RESPOSTAS_MT', 'TP_LINGUA',
       'TX_GABARITO_CN', 'TX_GABARITO_CH', 'TX_GABARITO_LC',
       'TX_GABARITO_MT', 'TP_STATUS_REDACAO',
      'Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006',
       'Q007', 'Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014',
       'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021', 'Q022',
       'Q023', 'Q024', 'Q025', 'Q026', 'Q027', 'Q028', 'Q029', 'Q030',
       'Q031', 'Q032', 'Q033', 'Q034', 'Q035', 'Q036', 'Q037', 'Q038',
       'Q039', 'Q040', 'Q041', 'Q042', 'Q043', 'Q044', 'Q045', 'Q046',
       'Q047', 'Q048', 'Q049', 'Q050']

df_train.drop(harddrop_list_train,axis=1,inplace=True)


# Making sure both datasets have the same columns (this could be improved)

for column in df_train.columns:
    if column in df_test.columns or column == 'NU_NOTA_MT' :
        continue
    else:
        df_train.drop([column], axis=1,inplace=True)

for column in df_test.columns:
    if column not in df_train.columns:
        df_test.drop([column], axis=1,inplace=True)


# Drop NaN rows:
# df_train.dropna(inplace=True)
# df_train.reset_index(drop=True,inplace=True)

# Drop rows without grade:
# list_nograde = []
# for result in df_train.NU_NOTA_MT:
#     if np.isnan(result) == True:
#         list_nograde.append(False)
#     else:
#         list_nograde.append(True)

# filtered = pd.Series(list_nograde)
# df_train = df_train[filtered]
# ##

#print(df_train.columns)


df_test.dropna(inplace=True)

# Store IDs in new DataFrame:
df_answer = pd.DataFrame({})
df_answer['NU_INSCRICAO'] = df_test['NU_INSCRICAO'].values

# Drop NU_inscricao:
for data in [df_train,df_test]:
    for column in data.columns:
        #print("column",df_train[column].dtype)
        if column == "NU_INSCRICAO": 
            data.drop([column], axis=1,inplace=True)

#new # Transform data for distribution of mean value 0 and standard deviation 1.
# scaler = StandardScaler()
# for data in [df_train,df_test]:
#     data = scaler.fit_transform(data)

# Replaces missing values of remaining columns with most common values
for data in [df_train,df_test]:
    for column in data.columns:
        #print("column",df_train[column].dtype)
        if data[column].dtype in [int, float, np.float32]:
            most_common_fct(data)
        else:
            data[column] = data[column].astype(str)
            most_common_fct(data)

# # Deal with conversion errors:
# for column in df_train.columns:
#     pd.to_numeric(df_train[column],downcast='float')
#     print(df_train[column].dtype)



############ ML #############

# Prediction Target (y):
train_y = df_train.NU_NOTA_MT


# Features (X):
columns_test = df_test.columns.tolist()
columns_train = df_train.columns.tolist()
print(columns_train)

columns_train.remove('NU_NOTA_MT')

train_X = df_train[columns_train]
test_X = df_test[columns_test]



# # Multiple Linear Regression model:
regressor = LinearRegression()
regressor.fit(train_X,train_y)
 # Predicting the Test set results:
df_answer['NU_NOTA_MT'] = regressor.predict(test_X)


# #RandomForestRegressor model:
# regr = RandomForestRegressor(max_depth = 3,random_state=5)
# regr.fit(train_X, train_y)
# df_answer['NU_NOTA_MT'] = regr.predict(test_X)



# DecisionTreeRegressor Model:
#  Define model. Specify a number for random_state to ensure same results each run:
# enem_model = DecisionTreeRegressor(random_state=1)

#  # Fit model
# enem_model.fit(train_X, train_y)

#  # Predict:
# df_answer['NU_NOTA_MT'] = enem_model.predict(test_X)


#Check:
print('The predictions are')
print(df_answer)

df_answer.to_csv('answer.csv',index=False)

