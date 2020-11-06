import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Importamos la Información
train = pd.read_csv(r'C:\Users\jl_ap\CursoPython-Final\IA-Fisica\Cayiyo\Proyecto 1\train.csv')
test =  pd.read_csv(r'C:\Users\jl_ap\CursoPython-Final\IA-Fisica\Cayiyo\Proyecto 1\test.csv')
target = pd.read_csv(r'C:\Users\jl_ap\CursoPython-Final\IA-Fisica\Cayiyo\Proyecto 1\sample_submission.csv')

#Separamos el Target del conjunto train
Columns = train.columns
train_x = train[Columns[:-1]]
train_y = train[['Id','SalePrice']]


# Arreglamos los datos del train y limpiamos
print(train_x.columns[0])
aux_train_x = train[Columns[1:-1]]
for col in aux_train_x.columns:
    DataCounts = train_x[col].value_counts(normalize=True,dropna=False)
    if float(DataCounts.iloc[0])>= 0.90:
        train_x.drop(col, axis=1,inplace=True)
        continue
    if sum(DataCounts.iloc[0:9])<0.46:
        train_x.drop(col, axis=1,inplace=True)
print(len(train_x.columns))

###############################################################################
#Arreglamos las columnas

def SameColumns(data1,data2):
    col1 = data1.columns
    col2 = data2.columns
    DelCol1 = []
    DelCol2 = []
    for col in col1:
        if col not in col2:
            DelCol1.append(col)
    for col in col2:
        if col not in col1:
            DelCol2.append(col)
    data1.drop(DelCol1, axis = 1,inplace=True)
    data2.drop(DelCol2, axis = 1,inplace=True)
    
SameColumns(train_x, test) 

#Convertimos valores categoricos a numericos
train_x = pd.get_dummies(train_x, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

#Volvemos a tener las mismas columnas
SameColumns(train_x, test) 


#Quitamos nans
train_x =train_x.dropna()
test = test.dropna()

#Arreglamos renglones
def SameRows(data1,data2):
    row1 = data1.index.tolist()
    row2 = data2.index.tolist()
    for row in row2:
        if row not in row1:
            try:
                data2.drop(row, inplace=True)
            except:
                print("An exception occurred")

            
SameRows(train_x, train_y) 
SameRows(test,target)
#Eliminamos el ID

aux_target = target.drop('Id',axis=1)
aux_test = test.drop('Id',axis=1)
aux_train_x = train_x.drop('Id',axis=1)
aux_train_y = train_y.drop('Id',axis=1)
###############################################################################
#Ajustamos mediante Lasso

Lasso_model = Lasso(alpha=0.5, normalize=True, max_iter = 1e6)
Lasso_model.fit(aux_train_x,aux_train_y)
coeficientes = Lasso_model.coef_

#Prediccion y MSE
PrecioPrediccion=Lasso_model.predict(aux_test)
MSE=mean_squared_error(aux_target['SalePrice'], PrecioPrediccion)
print(MSE)


##############################################################################
#Grafica
X=test['Id']
plt.plot(X,PrecioPrediccion,'b o', label='Prediccion')
plt.plot(X,target['SalePrice'],'r*', label='Data')
plt.xlabel('Id')
plt.ylabel('SalePrice')
plt.grid(True)
plt.legend()