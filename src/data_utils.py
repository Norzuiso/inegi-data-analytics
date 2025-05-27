import pandas as pd
import os
import numexpr as ne
import numpy as np
from typing import List
from functools import reduce

from config_class import SampleConfig, PreprocessingConfig, Column, MergeConfig

def get_groups(data: pd.DataFrame, col, discriminator):
    data[col] = data[col].astype(type(discriminator))

    pos_group = data[data[col] == discriminator]
    neg_group = data[data[col] != discriminator]
    return pos_group, neg_group


def generate_sample(data: pd.DataFrame,
                    target_column: str,
                    sample_config: SampleConfig):

    pos_group, neg_group = get_groups(data,
                                       target_column,
                                       sample_config.discriminator)
    if len(pos_group) < sample_config.sample_size or len(neg_group) < sample_config.sample_size:
        raise ValueError(f"Not enough data to sample {sample_config.sample_size} from both groups. "
                         f"Positive group size: {len(pos_group)}, Negative group size: {len(neg_group)}")

    pos_group = pos_group.sample(n=sample_config.sample_size, random_state=sample_config.random_seed)
    neg_group = neg_group.sample(n=sample_config.sample_size, random_state=sample_config.random_seed)

    return pd.concat([pos_group, neg_group], axis=0).reset_index(drop=True)


def merge_files(preprocessing: List[PreprocessingConfig], merge_config: MergeConfig):
    merge_columns = merge_config.merge_columns
    list_of_dataframes = []
    for prep in preprocessing:
        data = file_processing(prep.input_file)
        data, prep.columns = rename_columns(data, prep.columns)
        list_of_dataframes.append(data)

    base_file_name = get_base_file_name(merge_config.file_for_target)
    merge_config.target_column.column_name = f"{base_file_name}_{merge_config.target_column.column_name}"
    
    for df in list_of_dataframes:
        validate_str_columns(df, merge_columns)
    merged_df = reduce(lambda left, right: left.merge(right, on=merge_columns, how="inner"), list_of_dataframes)
    if merge_config.target_column.column_name not in merged_df.columns:
        raise ValueError(f"Target column '{merge_config.target_column.column_name}' not found in merged data.")
    merge_prepro = PreprocessingConfig(input_file="Files merged", feature_conditions=[], target_column=merge_config.target_column)
    for prep in preprocessing:
        merge_prepro.columns.extend(prep.columns)
    return merged_df, merge_prepro

def validate_str_columns(data: pd.DataFrame, columns: list[str]):
    available_cols = data.columns.tolist()
    missing_cols = [col for col in columns if col not in available_cols]
    if missing_cols:
        raise ValueError(f"Missing columns in the data: {', '.join(missing_cols)}")
    
def validate_columns(data: pd.DataFrame, columns: list[Column]):
    expected_cols = [col.column_name for col in columns]
    available_cols = data.columns.tolist()
    missing_cols = [col for col in expected_cols if col not in available_cols]
    if missing_cols: # Si en segments existen columnas que no existan en archivo, arroja error :p
        raise ValueError(f"Missing columns in the data: {', '.join(missing_cols)}")

def clean_data_columns(data: pd.DataFrame, columns: list[Column]):
    return data[[col.column_name for col in columns if col.column_name in data.columns]].copy()

def generate_condition_str(condition: str):
    if "(" in condition and ")" in condition:
        close = condition.split(")")
        for cond_split in close:
            cond_split 
    return True
def should_use_numexpr(column: Column):
    if column.value_type == "str":
        return False
    if "'" in column.condition or '"' in column.condition:
        return False
    return True

def binary_replace(data: pd.DataFrame, columns: list[Column]):
    for col in columns:
        name = col.column_name
        condition = col.condition
        val_true = col.true_value
        val_false = col.false_value
        if condition is None or condition == "":
            continue
        if should_use_numexpr(col):
            evaluated = ne.evaluate(condition, local_dict={"x": data[name].to_numpy()})
            data[name] = np.where(evaluated, val_true, val_false)
        else:
            serie_evaluada = data[name].apply(lambda x: eval(condition, {}, {"x": x}))
            data[name] = np.where(serie_evaluada, val_true, val_false)
    return data

def file_processing(preprocessing: PreprocessingConfig):
    file_path = preprocessing.input_file
    data = read_file(file_path)
    if preprocessing.columns:
        columns_cfg = preprocessing.columns
    else:
        columns_cfg = [
            Column(column_name = col)
            for col in data.columns.tolist()
        ]
    if preprocessing.target_column is not None and preprocessing.target_column.column_name not in data.columns:
        columns_cfg.append(preprocessing.target_column)

    validate_columns(data, columns_cfg)
    data = clean_data_columns(data, columns_cfg)
    data = binary_replace(data, columns_cfg)
    return data

def rename_columns(data: pd.DataFrame, columns: list[Column], file_path: str):
    base_file_name = get_base_file_name(file_path)
    rename_map = {
        col.column_name : f"{base_file_name}_{col.column_name}"
        for col in columns if col.column_name in data.columns
    }
    data.rename(columns=rename_map, inplace=True)
    return data, columns

def get_base_file_name(file):
    base_file_name = os.path.splitext(os.path.basename(file))[0]
    return base_file_name

def read_file(file_name: str):
    data = pd.read_csv(file_name)
    data = pd.DataFrame(data)
    data = data.fillna(0)
    return data