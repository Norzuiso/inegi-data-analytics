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
    
    # Si en las columnas no existe el target_column, lo agregamos
    if target_column not in columns_cfg:
        columns_cfg.append(target_column)

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

    data = binary_replace(data, columns_cfg)
    data.fillna(0, inplace=True)  # Rellenar NaN con 0 o un valor más adecuado según el contexto
    data.replace(" ", 0, inplace=True)  # Reemplazar espacios en blanco con 0

    
    # Ahora hacemos el sample
    positive_group = data[(data[target_column.column_name] != 0)]
    negative_group = data[(data[target_column.column_name] != 1)]

    if len(positive_group) < sample_config.sample_size or len(negative_group) < sample_config.sample_size:
        raise ValueError(f"Not enough data to sample {sample_config.sample_size} from both groups. "
                         f"Positive group size: {len(positive_group)}, Negative group size: {len(negative_group)}")
    
    sample = pd.concat([positive_group.sample(n=sample_config.sample_size, random_state=sample_config.random_seed),
                negative_group.sample(n=sample_config.sample_size, random_state=sample_config.random_seed)],
               axis=0).reset_index(drop=True)
    
    sample = sample.apply(pd.to_numeric, errors='coerce')
    sample.to_csv(sample_config.output_sample_file, index=False)
    
    corr_matrix = sample.corr(method=sample_config.correlation_method)
    cor_target = corr_matrix[target_column.column_name].abs()
    cor_target = cor_target.drop(target_column.column_name)
    correlated_features = cor_target[cor_target > sample_config.correlation_threshold].sort_values(ascending=False)
    correlated_features.to_csv(general.results_directory + "correlated_features.csv")
