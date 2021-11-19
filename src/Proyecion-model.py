import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import holidays_co
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as pipe
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , make_scorer, f1_score
import pickle


def DataModificada(Data_EPM,Hora_Proyectar,cantDatos):
    if Hora_Proyectar>=cantDatos+1:
      Train=Data_EPM.iloc[:,4+Hora_Proyectar-cantDatos-1:Hora_Proyectar+4]
      Train.insert(0,'Day',list(Data_EPM["Day"]),True)
      Train.insert(1,'Holiday',list(Data_EPM["Holiday"]),True)
    else:
      Corte=cantDatos+1-Hora_Proyectar
      Train=Data_EPM.iloc[1::,4:Hora_Proyectar+4]
      for i in range(Corte):
        Train.insert(0,f"P_H_{24-i}",list(Data_EPM.iloc[0:len(Data_EPM)-1,Data_EPM.shape[1]-2-i]),True)
      Train.insert(0,'Day',list(Data_EPM["Day"])[1:],True)
      Train.insert(1,'Holiday',list(Data_EPM["Holiday"])[1:],True)
    return Train


Hora_Proyectar=1


#Datasets: Prep

Demandas="https://raw.githubusercontent.com/cubides0905/TareaFinal/main/Demanda_por_OR_2020.xlsx"
Data=pd.read_excel(Demandas,skiprows=range(0, 2))
Data.columns = ['Fecha','Codigo_OR','P_H_1','P_H_2','P_H_3','P_H_4','P_H_5','P_H_6','P_H_7','P_H_8','P_H_9','P_H_10','P_H_11','P_H_12','P_H_13','P_H_14','P_H_15','P_H_16','P_H_17','P_H_18','P_H_19','P_H_20','P_H_21','P_H_22','P_H_23','P_H_24','Tx']
Data_EPM=Data.loc[(Data.Codigo_OR=='EPMD')]

Fechas=list(Data_EPM.Fecha)
DiaSemana=[]
Holiday=[]
for i in range(len(Fechas)):
  dt=Fechas[i]
  year, month, day = (int(x) for x in dt.split('-'))  
  ans = datetime.date(year, month, day)
  if holidays_co.is_holiday_date(ans)==True:
    Holiday.append(1)
  else:
    Holiday.append(0)
  if ans.weekday()==6:
    Holiday[i]=2
  if ans.weekday()==5:
    Holiday[i]=3
  DiaSemana.append(ans.weekday()) # 0:lunes, 1:Martes, 2: Miercoles, 3:Jueves, 4 : viernes, 5 : Sabado, 6 Domingo 
Data_EPM.insert(1,'Day',DiaSemana,True)
Data_EPM.insert(2,'Holiday',Holiday,True)
Train=DataModificada(Data_EPM,Hora_Proyectar,4)

#print(Train)
#Train and test data

X2 = Train.iloc[:,0:Train.shape[1]-1]
y2 = Train.iloc[:,Train.shape[1]-1:Train.shape[1]]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=1)

#ML Model: Model Selection
lr=LinearRegression()
lr.fit(X_train2,y_train2)

#predictions 
pred2=lr.predict(X_test2)

#Accuary
def custom_error2(
    y_true,
    y_pred
 ):
    """A custom metric that is related to the business, the lower the better."""
    diff = (y_true - y_pred)/y_true  # negative if predicted value is greater than true value
   
    return max(np.abs(diff).iloc[:,0]*100)

print(custom_error2(y_test2,pred2))

#Registro
with open('./outputs/proyecc_model.pkl', 'wb') as model_pkl:
    pickle.dump(lr, model_pkl)
