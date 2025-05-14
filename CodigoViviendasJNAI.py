# Se necesitan las siguientes p1_1 a la P1_3
import os
import pandas as pd

# Preprocesado y modelado
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
import statsmodels.formula.api as smf

sc = StandardScaler()


# Columnas y valores del archivo vivienda que necesitas
segments_vivienda = {
                    "P1_1": [1, 2, 3, 4], # Estos valores son los valores significativos y que tendran True 
                    "P1_2": [1, 2, 3, 4], # Los valores que no agregues tendran false
                    "P1_3": []
                    }
# Columnas y valores del archivo adultos que vayas a probar. 
segments_adultos = {
                 "P4_1": [1],
                 "P4_2M": [],
                 "P4_2A": [],
                 "P4_3": [],
                 "P4_4": [],
                 "P4_5M": [],
                 "P4_5A": [],
                 "P4_6": [],
                 "P4_7": [],
                 "P4_8_1": [],
                 "P4_8_2": [],
                 "P4_8_3": [],
                 "P4_8_4": [],
                 "P4_8_5": [],
                 "P4_9": [],
                 "P4_9_1": [],
                 "P4_10_1": [],
                 "P4_10_2": [],
                 "P4_10_3": [],
                 "P4_10_4": [],
                 "P4_10_5": [],
                 "P4_10_5V": [],
                 "P4_10_6": [],
                 "P4_10_6V": [],
                 "P4_10_6C": [],
                 "P13_1": [],
                 "P13_2": [],
                 "P13_3": [],
                 "P13_4": [],
                 "P13_5": [],
                 "P13_6": [],
                 "P13_6_1": [],
                 "P13_7_1": [],
                 "P13_7_2": [],
                 "P13_8": [],
                 "P13_8_1": [],
                 "P13_9": [],
                 "P13_10": [],
                 "P13_11": [],
                 "P13_12_1": [],
                 "P13_12_2": [],
                 "P13_13M": [],
                 "P13_13A": [],
                 "P13_14": []
                 }

# Nombre de los archivos que van a anlizarse
file_viviendas = "CN_VIVIENDAS.csv"
file_adultos = "CS_ADULTOS.csv"

# Columnas que relacionan a los archivos entre si para que no se rompa todo
merge_columns = ["DOMINIO",
                 "ENT",
                 "EST_DIS",
                 "ESTRATO",
                 "REGION",
                 "UPM",
                 "UPM_DIS",
                 "VIV_SEL"
]
# Verbose permite ver las tablas o las cosas que se estan modificando
VERBOSE = False

# La columna que pongas aquí sera la la variable dependiente
target = {
            "file": file_adultos, # Este es el archivo en el que se encuentra esa variable
            "column": "P4_1" # Columna de la variable indepentendiente
          }

# La condición que se va a usar para separar los dos grupos.
# Nota: Para este punto los valores de la columna target son binarios (1 o 0)
target_segment_condition_positive = lambda column : column > 0 
target_segment_condition_negative = lambda column : column == 0 

# Este es el tamaño de la población
sample_size = 3000

def main():
    data_viviendas = regular_file_process(file_viviendas, segments_vivienda)
    data_adultos = regular_file_process(file_adultos, segments_adultos)

    # Hace merge de lambos datasets
    data = pd.merge( data_adultos, 
                    data_viviendas, 
                    on=merge_columns, # Son las columnas que coinciden
                    how='inner') #Conserva solo coincidencias en ambos

    col_target = get_col_target_name(target)
    poblacion = generate_poblacion(data, col_target)
    print(poblacion.describe())
    
    training_process(poblacion, col_target)

def training_process(poblacion: pd.DataFrame, col_target: str):
    # Definimos la variable dependiente Y y las variables independientes X del 'DataFrame'
    Y = poblacion[col_target]
    X = poblacion.drop(col_target, axis=1)
    print(poblacion[col_target].value_counts())
    
    # Dividimos la muestra prueba y entrenamiento
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y) 

    # Aplicamos una normalizacion a los datos para asegurarnos de trabajar en las mismas escalas
    # Para el modelo de clasificacion aplicamos el algoritmo de Regresion Logistica Optimizado mediante 'GD'
    X_train_array = sc.fit_transform(X_train.values)
    X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
    X_test_array = sc.transform(X_test.values)
    X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)



def generate_poblacion(data: pd.DataFrame, col_target: str):
    poblacion_positiva = data[data[col_target].apply(target_segment_condition_positive)]
    poblacion_positiva = poblacion_positiva.sample(n=sample_size, random_state=1, replace=True)

    poblacion_negativa = data[data[col_target].apply(target_segment_condition_negative)]
    poblacion_negativa = poblacion_negativa.sample(n=sample_size, random_state=1, replace=True)

    poblacion = pd.concat([poblacion_positiva, poblacion_negativa], axis=0)

    return poblacion


def get_col_target_name(target: dict[str, str]): 
    file = target.get("file")
    base_file_name = get_base_file_name(file)
    col_target = f"{base_file_name}_{target.get("column")}"
    return col_target

def get_base_file_name(file):
    base_file_name = os.path.splitext(os.path.basename(file))[0]
    return base_file_name


def regular_file_process(file_name: str, segments: dict[str, list]):
    data = read_csv_order(file_name)
    validate_columns_in_file(data, list(segments.keys()), file_name) 
    validate_merge_columns(data, list(segments.keys()), merge_columns, file_name)
    data = filter_columns(segments, data, merge_columns)
    data = replace_values(segments, data)
    data, segments = rename_columns(file_name, segments, data)
    return data

def validate_merge_columns(data: pd.DataFrame, segments: list, merge_columns: list, file_name: str):
    validate_columns_in_file(data, merge_columns, file_name) # Validamos que merge_columns esten en archivo
    overlapping = set(segments).intersection(set(merge_columns)) # Hacemos interesencción entre merge y segment
    if overlapping: # si hay intersección, arrojamos error
        raise Exception(f"Columnas conflictivas entre segments y merge_columns en {file_name}: {list(overlapping)}")


def validate_columns_in_file(data: pd.DataFrame, segments_keys: list, file_name: str):
    # 1.- Primero sacamos las columnas que estan declaradas en el segments
    expected_cols = segments_keys
    # 2. Ahora vemos cuales estan en el DataFrame
    available_cols = data.columns
    # 3. Cuales son las que no estan en el DataFrame pero si en el segments
    missing_cols = [col for col in expected_cols if col not in available_cols]
    if missing_cols: # Si en segments existen columnas que no existan en archivo, arroja error :p
        raise Exception(f"Columnas no encontradas en el archivo: {file_name}.\nColumnas no encontradas: {missing_cols}")

def rename_columns(file_names: str, segmets: dict[str, list], data: pd.DataFrame):
    old_cols = list(segmets.keys())
    new_cols = [get_base_file_name(file_names) + "_" + col for col in old_cols] # 1.- Construir nuevos nombres de columnas
    data.rename(columns=dict(zip(old_cols, new_cols)), inplace=True)   # 2.- Rename columnas en el DataFrame
    new_segments = dict(zip(new_cols, segmets.values()))    # 3.- Reconstruir el nuevo segmets
    return data, new_segments

def read_csv_order(csv_file_name: str):
    data = pd.read_csv(csv_file_name) # Busca el archivo
    data = pd.DataFrame(data) # Carga el DataFrame
    data = data.fillna(0) # Remplaza los valores NA con 0
    data = data.astype(int) # Convierte todo a enteros
    return data

def filter_columns(segments: dict[str, list], data: pd.DataFrame, merge_cols = None):
    cols = list(segments.keys()) # Tomamos la lista de columnas dentro del segmento enviado
    if merge_cols:
        cols += merge_cols
    if VERBOSE:
        print(f"Filtrando columnas: {cols}") # Mostramos cuales columnas se estan filtrando
    return data[cols]    # Realiza el filtrado de las columnas

def replace_values (segments: dict[str, list], data: pd.DataFrame):
    for col, vals in segments.items(): # Recorre segmentos
        if vals:
            data[col] = data[col].isin(vals).astype(int) 
            # Si el valor de la col coincide con alguno dentro de la lista de vals, guarda como True/False
            # Cuando realiza el astype(int) combierte el Boolean en 0 o 1
    return data


main()