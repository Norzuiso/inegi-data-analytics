import logging
import pandas as pd
import numpy as np
from typing import List

from config import parameters
from config_class import AnalysisConfig, GeneralConfig, PreprocessingConfig, SampleConfig, VisualizationConfig,ModelsConfig,  SampleConfig, PreprocessingConfig, Column, MergeConfig
from data_utils import generate_sample, merge_files, file_processing, read_file, binary_replace
from visualization import explore_corr
from modeling import train

# -*- coding: utf-8 -*-

def testing_code( config: GeneralConfig):        
    general: GeneralConfig = config.general
    preprocessing: List[PreprocessingConfig] = config.preprocessing
    sample_config: SampleConfig = config.sample_config
    visualization: VisualizationConfig = config.visualization_config
    models: ModelsConfig = config.models
    
    pre: PreprocessingConfig = preprocessing[0]
    target_column = pre.target_column

    data = read_file(preprocessing[0].input_file)

    columns_cfg = pre.columns
    # si columnas esta vacio agarramos todas las columnas del dataframe
    if len(pre.columns) == 0: 
        for col in data.columns.tolist():
            if col not in pre.remove_columns:
                columns_cfg.append(Column(column_name=col, condition=None))

    # Si hay columnas a eliminar, las eliminamos
    if pre.remove_columns is not None:
        # Si target esta n remove, le decimos que esta pendejo
        if target_column.column_name in pre.remove_columns:
            raise ValueError(f"Target column '{target_column.column_name}' cannot be in remove_columns list.")
        data = data.drop(columns=pre.remove_columns, errors='ignore')
    
    # Solo checamos que las columnas si existen en el dataframe
    if pre.columns is not None:
        expected_columns = [col.column_name for col in columns_cfg]
        available_columns = data.columns.tolist()
        missing_columns = [col for col in expected_columns if col not in available_columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the data: {', '.join(missing_columns)}")

    # limpieza de columnas que nos dan asco
    if pre.remove_columns is not None:
        data = data.drop(columns=pre.remove_columns, errors='ignore')
    data.to_csv(f"{general.results_directory}data.csv")
    column_names = [col.column_name for col in pre.columns]

    #data = binary_replace(data, columns_cfg)
    data.fillna(0, inplace=True)  # Rellenar NaN con 0 o un valor más adecuado según el contexto
    data.replace(" ", 0, inplace=True)  # Reemplazar espacios en blanco con 0
    data[[col.column_name for col in pre.columns if col.column_name in data.columns]].copy()
    
    data[target_column.column_name] = pd.to_numeric(data[target_column.column_name])  
    filtered_data = data.query("a0401 != 3")
    filtered_data = filtered_data[column_names]

    filtered_data.to_csv(f"{general.results_directory}filtered_data.csv", index=False)
    
    # Ahora hacemos el sample
    positive_group = data[(data[target_column.column_name] == 1)]
