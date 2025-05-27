import logging
import pandas as pd

from typing import List

from config import parameters
from config_class import AnalysisConfig, GeneralConfig, PreprocessingConfig, SampleConfig, VisualizationConfig,ModelsConfig
from data_utils import generate_sample, read_file, merge_files, file_processing
from modeling import train
from visualization import explore_corr



# Configuración del logger
log = logging.getLogger(__name__)


def main():
    config: AnalysisConfig = parameters()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log.info("Starting the analysis with the provided configuration.")
    log.info(f"Configuration: {config}")
    
    general: GeneralConfig = config.general
    preprocessing: List[PreprocessingConfig] = config.preprocessing
    sample_config: SampleConfig = config.sample_config
    visualization: VisualizationConfig = config.visualization_config
    models: ModelsConfig = config.models
    
    processing_mode = general.processing_mode
    data = []    
    if processing_mode == "single":
        log.info("Only one preprocessing configuration provided. Using it for file processing.")
        data.append(file_processing(preprocessing[0].input_file))

    if processing_mode == "merge":
        log.info("Processing mode is set to 'merge'. Merging files.")
        data = [file_processing(prep.input_file) for prep in preprocessing]
        data = merge_files(data, general.merge_columns)

    target = general.target_column
    if not target:
        raise ValueError("Target column must be specified in the configuration.")


    
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