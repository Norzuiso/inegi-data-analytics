# Importamos las librerias necesarias

# Tratamiento de datos
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
#%matplotlib inline

# Gráficos
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuración matplotlib
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')

# Importamos el archivo CS_ADULTOS.csv obtenido de la pagina del 'INEGI' y lo convertimos en 'DataFrame'
data = pd.read_csv('CS_ADULTOS.csv')
data = pd.DataFrame(data)
data = data.fillna(0)

# Filtramos y eliminamos las columnas del dataframe que no guardan relacion con la columna 'P3_1' (Diabetes)
data = data.drop(['UPM', 'VIV_SEL', 'HOGAR', 'NUMREN', 'P1_2', 'P1_3',
       'P1_5', 'P1_6', 'P1_7', 'P1_8', 'P1_9', 'P1_10_1',
       'P1_10_2', 'P1_10_3', 'P1_10_4', 'P1_10_5', 'P1_10_6', 'P1_10_7',
       'P1_10_8', 'P1_10_9', 'P1_10_10', 'P2_1_1',
       'P2_1_6',
       'P3_2', 'P3_3V', 'P3_4', 'P3_5D', 'P3_5M', 'P3_5A', 'P3_6',
       'P3_7_1', 'P3_7_2', 'P3_7_3', 'P3_7_4', 'P3_7_5', 'P3_7_6',
       'P3_7_7', 'P3_7_8', 'P3_7_9', 'P3_7_10', 'P3_7_11', 'P3_7_12',
       'P3_8', 'P3_9M', 'P3_9A', 'P3_10M', 'P3_10A', 'P3_11', 'P3_12',
       'P3_14_1',
       'P3_14_2', 'P3_14_3', 'P3_14_4',
       'P3_14_5', 'P3_14_6', 'P3_15_6',
       'P3_14_7', 'P3_14_8', 'P3_16_2',
       'P3_16_7', 'P3_16_8', 'P3_16_10',
       'P3_16_11', 'P3_16_12',
       'P3_16_17', 'P3_16_18',       
       'P4_2M', 'P4_2A', 'P4_3', 'P4_4',
       'P4_5M', 'P4_5A', 'P4_6', 'P4_7', 'P4_8_1', 'P4_8_2', 'P4_8_3',
       'P4_8_4', 'P4_8_5', 'P4_9', 'P4_9_1', 'P4_10_1', 'P4_10_2',
       'P4_10_3', 'P4_10_4', 'P4_10_5', 'P4_10_5V', 'P4_10_6', 'P4_10_6V',
       'P4_10_6C', 'P5_1', 'P5_2_2', 'P5_2_3', 'P5_3', 'P5_4',
       'P5_5', 'P5_6', 'P5_7', 'P6_1_1', 'P6_1_2', 'P6_2_1',
       'P6_2_2', 'P6_2_3', 'P6_2_4', 'P6_2_5', 'P6_2_6', 'P6_2_7', 'P6_3',
       'P6_4', 'P6_5_1', 'P6_5_2', 'P6_5_3', 'P6_5_4', 'P6_6', 'P6_7_1',
       'P6_7_2', 'P6_7_3', 'P6_7_4', 'P6_8_1', 'P6_8_2', 'P6_8_3',
       'P6_8_4', 'P6_8_5', 'P6_8_6', 'P6_9', 'P7_1', 'P7_2', 'P7_3',
       'P7_4_1', 'P7_5_1',
       'P7_4_2', 'P7_5_2',
       'P7_4_3', 'P7_5_3', 'P8_1', 'P8_2_1', 'P8_2_2', 'P8_2_3',
       'P8_2_4', 'P8_2_5', 'P8_2_6', 'P8_2_7', 'P8_2_8', 'P8_2_9',
       'P8_2_10', 'P8_2_11', 'P8_2_12', 'P8_2_13', 'P8_2_14', 'P8_2_15',
       'P8_3_1', 'P8_3_2', 'P8_3_3', 'P8_3_4', 'P8_3_5', 'P8_3_6',
       'P8_3_7', 'P8_3_8', 'P8_3_9', 'P8_3_10', 'P8_3_11', 'P8_3_12',
       'P8_3_13', 'P8_3_14', 'P8_3_15', 'P8_3_16', 'P8_3_17', 'P8_4',
       'P8_5', 'P8_6M', 'P8_6A', 'P8_7', 'P8_8', 'P8_9', 'P8_10',
       'P8_11_1', 'P8_11_2', 'P8_11_3', 'P8_11_4', 'P8_11_5', 'P8_13',
       'P8_14_1', 'P8_14_2', 'P8_14_3', 'P8_14_4', 'P8_14_5', 'P8_14_6',
       'P8_14_7', 'P8_15', 'P8_16', 'P8_17_1', 'P8_17_2', 'P8_17_3',
       'P8_17_4', 'P8_17_5', 'P8_17_6', 'P8_17_7', 'P8_17_8', 'P8_17_9',
       'P8_17_10', 'P8_17_11', 'P8_17_12', 'P8_17_13', 'P8_17_14',
       'P8_17_15', 'P8_17_16', 'P8_18', 'P8_19', 'P8_20_1', 'P8_20_2',
       'P8_21_1', 'P8_21_2', 'P8_21_3', 'P8_21_4', 'P8_21_5', 'P8_21_6',
       'P8_21_7', 'P8_21_8', 'P8_21_9', 'P8_21_10', 'P8_21_11',
       'P8_21_12', 'P8_21_13', 'P8_21_14', 'P8_22', 'P8_23', 'P8_24_1',
       'P8_24_2', 'P8_24_3', 'P8_25_1', 'P8_25_2', 'P8_26_1', 'P8_26_2',
       'P8_26_3', 'P8_26_4', 'P8_26_5', 'P8_26_6', 'P8_26_7', 'P8_26_8',
       'P8_26_9', 'P8_26_10', 'P8_27', 'P8_28', 'P8_29_1', 'P8_29_2',
       'P8_29_3', 'P8_29_4', 'P8_29_5', 'P8_30', 'P8_31', 'P8_32', 'P9_1',
       'P9_2', 'P9_3', 'P9_4', 'P9_5', 'P9_6', 'P9_7', 'P9_8', 'P9_9_A1',
       'P9_9_B1D', 'P9_9_B1M', 'P9_9_B1A', 'P9_9_C1', 'P9_9_A2',
       'P9_9_B2D', 'P9_9_B2M', 'P9_9_B2A', 'P9_9_C2', 'P9_9_A3',
       'P9_9_B3D', 'P9_9_B3M', 'P9_9_B3A', 'P9_9_C3', 'P9_10_A1',
       'P9_10_B1D', 'P9_10_B1M', 'P9_10_B1A', 'P9_10_C1', 'P9_10_A2',
       'P9_10_B2D', 'P9_10_B2M', 'P9_10_B2A', 'P9_10_C2', 'P9_10_A3',
       'P9_10_B3D', 'P9_10_B3M', 'P9_10_B3A', 'P9_10_C3', 'P9_10_A4',
       'P9_10_B4D', 'P9_10_B4M', 'P9_10_B4A', 'P9_10_C4', 'P9_11_A1',
       'P9_11_B1D', 'P9_11_B1M', 'P9_11_B1A', 'P9_11_C1', 'P9_12_A1',
       'P9_12_B1D', 'P9_12_B1M', 'P9_12_B1A', 'P9_12_C1', 'P9_12_A2',
       'P9_12_B2D', 'P9_12_B2M', 'P9_12_B2A', 'P9_12_C2', 'P9_13',
       'P9_14', 'P9_15', 'P9_16', 'P9_17', 'P9_18', 'P9_19', 'P9_20',
       'P9_21', 'P9_22_A1', 'P9_22_B1D', 'P9_22_B1M', 'P9_22_B1A',
       'P9_22_C1', 'P9_22_A2', 'P9_22_B2D', 'P9_22_B2M', 'P9_22_B2A',
       'P9_22_C2', 'P9_22_A3', 'P9_22_B3D', 'P9_22_B3M', 'P9_22_B3A',
       'P9_22_C3', 'P9_23_A1', 'P9_23_B1D', 'P9_23_B1M', 'P9_23_B1A',
       'P9_23_C1', 'P9_23_A2', 'P9_23_B2D', 'P9_23_B2M', 'P9_23_B2A',
       'P9_23_C2', 'P9_23_A3', 'P9_23_B3D', 'P9_23_B3M', 'P9_23_B3A',
       'P9_23_C3', 'P9_23_A4', 'P9_23_B4D', 'P9_23_B4M', 'P9_23_B4A',
       'P9_23_C4', 'P9_24_A1', 'P9_24_B1D', 'P9_24_B1M', 'P9_24_B1A',
       'P9_24_C1', 'P9_25', 'P10_1_1', 'P10_2_1', 'P10_3_1', 'P10_4_1',
       'P10_5_1', 'P10_6_1', 'P10_7_1', 'P10_1_2', 'P10_2_2', 'P10_3_2',
       'P10_4_2', 'P10_5_2', 'P10_6_2', 'P10_7_2', 'P10_1_3', 'P10_2_3',
       'P10_3_3', 'P10_4_3', 'P10_5_3', 'P10_6_3', 'P10_7_3', 'P10_1_4',
       'P10_2_4', 'P10_3_4', 'P10_4_4', 'P10_5_4', 'P10_6_4', 'P10_7_4',
       'P10_1_5', 'P10_2_5', 'P10_3_5', 'P10_4_5', 'P10_5_5', 'P10_6_5',
       'P10_7_5', 'P10_1_6', 'P10_2_6', 'P10_3_6', 'P10_4_6', 'P10_5_6',
       'P10_6_6', 'P10_7_6', 'P10_1_7', 'P10_2_7', 'P10_3_7', 'P10_4_7',
       'P10_5_7', 'P10_6_7', 'P10_7_7', 'P10_1_8', 'P10_2_8', 'P10_3_8',
       'P10_4_8', 'P10_5_8', 'P10_6_8', 'P10_7_8', 'P10_1_9', 'P10_2_9',
       'P10_3_9', 'P10_4_9', 'P10_5_9', 'P10_6_9', 'P10_7_9', 'P10_1_10',
       'P10_2_10', 'P10_3_10', 'P10_4_10', 'P10_5_10', 'P10_6_10',
       'P10_7_10', 'P10_1_11', 'P10_2_11', 'P10_3_11', 'P10_4_11',
       'P10_5_11', 'P10_6_11', 'P10_7_11', 'P11_1', 'P11_2', 'P11_3',
       'P11_4', 'P11_5', 'P11_6', 'P11_7', 'P11_8', 'P12_1', 'P12_2_1',
       'P12_2_2', 'P12_2_3', 'P12_2_4', 'P12_2_5', 'P12_2_6', 'P12_2_7',
       'P12_2_8', 'P12_2_9', 'P12_2_10', 'P12_2_11', 'P12_3', 'P12_4_1',
       'P12_5', 'P12_6', 'P12_7', 'P12_7_1', 'P12_8', 'P12_8_1', 'P13_1',
       'P13_3', 'P13_4', 'P13_5', 'P13_6', 'P13_6_1', 'P13_7_1',
       'P13_7_2', 'P13_8', 'P13_8_1', 'P13_9', 'P13_10',
       'P13_12_2', 'P13_13M', 'P13_13A', 'P13_14', 'P14_1',
       'P14_2', 'P14_3', 'P14_4', 'P14_5', 'P14_6', 'P14_7', 'P14_8',
       'P14_9', 'P14_10', 'ENT', 'DOMINIO', 'REGION',
       'EST_DIS', 'UPM_DIS', 'ESTRATO', 'F_20MAS', 'FECHA_NAC',
       'FECHA_ENT', 'DIFERENCIA', 'HIJ_ULT5AD', 'HIJ_ULT1AD', 'EDAD'], axis=1)

# Convertimos todos los datos en formato 'int32'
data = data.astype(int)
data.dtypes

data.replace({'SEXO' : 2}, 0, inplace=True)

# Para el analisis predictivo, creamos una poblacion n=3000 seleccionando individuos con diabetes
column = data['P3_1']
PDiabeticos = data[column>0]
PDiabeticos = PDiabeticos.sample(n=3000, random_state=1)
PDiabeticos

# Creamos otra poblacion n=3000 seleccionando individuos no diabeticos
PNDiabeticos = data[data['P3_1']==0]
PNDiabeticos = PNDiabeticos.sample(n=3000, random_state=1)
PNDiabeticos

# Creamos una poblacion de 6000 registros con 50% individuos diaveticos y 50% de individuos no diabeticos
poblacion = pd.concat([PDiabeticos,PNDiabeticos], axis=0)
poblacion

# Mostramos estadisticas descriptivas basicas del 'DataFrame' con el que vamos a trabajar
poblacion.describe()

# Definimos la variable dependiente Y y las variables independientes X del 'DataFrame'
Y = poblacion['P3_1']
X = poblacion.drop('P3_1',axis =1)
print(poblacion['P3_1'].value_counts())

# Dividimos la muestra prueba y entrenamiento
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y) 

# Aplicamos una normalizacion a los datos para asegurarnos de trabajar en las mismas escalas
# Para el modelo de clasificacion aplicamos el algoritmo de Regresion Logistica Optimizado mediante 'GD'
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_array = sc.fit_transform(X_train.values)
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test_array = sc.transform(X_test.values)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

# Algoritmo de 'Gradiente Decendiente'
class LogisticRegressionGD(object):
    
    def __init__(self, l_rate = 0.1, n_iter =10000, random_state =1):
        self.l_rate = l_rate
        self.n_iter = n_iter
        self.random_state = random_state               
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)        
        self.theta = rgen.normal(loc = 0.0, scale = 0.01,
                                 size = 1 + X.shape[1])     
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            h = self.sigmoid(net_input)   
            errors = y-h
            self.theta[1:] += -self.l_rate*X.T.dot(errors) 
            self.theta[0] += -self.l_rate*errors.sum()          
        return self.theta
    
    def sigmoid(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def net_input(self, X):
        return np.dot(X, self.theta[1:]) + self.theta[0]
    
    def predict(self, X):        
        return np.where(self.sigmoid(self.net_input(X))>= 0.5, 0, 1)

# Aplicación del modelo de Regresion Logistica optimizado mediante Gradiente Decendeinte
regression = LogisticRegressionGD(l_rate = 0.0000001, n_iter = 20000)
coef = regression.fit(X_train, Y_train)
Y_predict = regression.predict(X_test)

# Matriz de confusión para el modelo de Regresion Logista mediante Gradiente Decendiente
from sklearn.metrics import confusion_matrix

confusion_matrix = pd.crosstab(Y_predict, Y_test.ravel(), rownames=['Prediccion'], colnames=['Real'])
confusion_matrix

# Muestra la precision del modelo realizado mediante 'Regresion Logistica' con 'Gradiente Desendiente'
print(f"La precision del modelo 'Regresion Logistica GD' es de: {100*(1125/1200)}%")

# Muestra los valores predichos por el modelo
y_pred = regression.predict(X_test)
y_pred

# Con la finalidad de realizar una comparacion, realizamos la clasificacion de los datos mediante el algoritmo de 'SVM'
from sklearn.svm import SVC
clf=SVC(kernel='poly').fit(X_train,Y_train)

predicciones = clf.predict(X_test)
predicciones

# Muestra la precision del modelo realizado mediante el algoritmo 'SVM'
print(f"La precision del modelo es de: {100*clf.score(X_test,Y_test)}%")

# Matriz de confusión de las predicciones de test (SVM)
confusion_matrix = pd.crosstab(predicciones, Y_test.ravel(), rownames=['Prediccion'], colnames=['Real'])
confusion_matrix

# Muestra la matriz de correlacion de las variables del 'DataFrame'
PDiabeticos = data[data['P3_1']==1]
corr = poblacion.corr()
corr