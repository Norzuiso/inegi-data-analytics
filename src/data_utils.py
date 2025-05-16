import pandas as pd

from config_class import Sample_config


def get_groups(data: pd.DataFrame, col, pos_condition, neg_condition):
    data = data[data[col].isin([pos_condition, neg_condition])]
    pos_group = data[data[col] == pos_condition]
    neg_group = data[data[col] == neg_condition]
    return pos_group, neg_group


def generate_sample(data: pd.DataFrame,
                    target_column: str,
                    sample_config: Sample_config):

    pos_group, neg_group = get_groups(data,
                                       target_column,
                                       sample_config.pos_condition,
                                       sample_config.neg_condition)
    if len(pos_group) < sample_config.sample_size or len(neg_group) < sample_config.sample_size:
        raise ValueError("No hay suficientes datos en alguno de los grupos")

    pos_group = pos_group.sample(n=sample_config.sample_size, random_state=sample_config.random_state)
    neg_group = neg_group.sample(n=sample_config.sample_size, random_state=sample_config.random_state)

    return pd.concat([pos_group, neg_group], axis=0).reset_index(drop=True)


def read_file(file_name: str):
    data = pd.read_csv(file_name)
    data = pd.DataFrame(data)
    data = data.fillna(0)
    data = data.astype(int)
    return data