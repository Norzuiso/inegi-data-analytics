import logging

from config import parameters
from config_class import Config
from data_utils import generate_sample
from data_utils import read_file
from modeling import train
from visualization import explore_corr

# Configuración del logger
log = logging.getLogger(__name__)

def main():
    config: Config = parameters()
    target = config.target_column
    columns = config.columns
    file = config.file
    sample_config = config.sample_config
    graph_config = config.graph_config
    model_config = config.model_config
    
    data = read_file(file)

    if len(columns) != 0:
        columns.append(target)
        data = data[columns]
    sample = generate_sample(data,
                             target,
                             sample_config)
    print(sample.describe())
    
    sample.to_csv(config.sample_config.sample_file, index=False)
    corr_matrix = sample.corr()
    cor_target = corr_matrix[target].abs()
    cor_target = cor_target.drop(target)
    correlated_features = cor_target[cor_target > sample_config.percentage_per_corr].sort_values(ascending=False)
    correlated_features.to_csv(config.corr_store_file)

    explore_corr(sample,
                 correlated_features,
                 target,
                 graph_config)

    train(sample,
          correlated_features,
          model_config,
          config,
          columns,
          target,
          sample_config.test_size,
          random_state=sample_config.random_state,)

if __name__ == "__main__":
    main()