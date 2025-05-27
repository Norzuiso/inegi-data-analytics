import logging
import pandas as pd
import numpy as np
from typing import List

from config import parameters
from config_class import AnalysisConfig, GeneralConfig, PreprocessingConfig, SampleConfig, VisualizationConfig,ModelsConfig
from data_utils import generate_sample, merge_files, file_processing
from visualization import explore_corr
from modeling import train

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
    data = pd.DataFrame
    if processing_mode == "single":
        log.info("Only one preprocessing configuration provided. Using it for file processing.")
        preprocessing: PreprocessingConfig = preprocessing[0]
        data = file_processing(preprocessing)
        data.to_csv(f"{general.results_directory}data.csv")

    if processing_mode == "merge":
        log.info("Processing mode is set to 'merge'. Merging files.")
        data, preprocessing = merge_files(preprocessing, general.merge_config)
        data.to_csv(f"{general.results_directory}data.csv")

    
    target = preprocessing.target_column.column_name
    columns = preprocessing.columns
    

    sample = generate_sample(data, target, sample_config)
    print(sample.describe())
    sample.to_csv(config.sample_config.output_sample_file, index=False)
    sample = sample.select_dtypes(include=[np.number])
    sample = sample.fillna(0)  # o usa un valor más sensato, según contexto
    corr_matrix = sample.corr(method=sample_config.correlation_method)
    cor_target = corr_matrix[target].abs()
    cor_target = cor_target.drop(target)
    correlated_features = cor_target[cor_target > sample_config.correlation_threshold].sort_values(ascending=False)
    correlated_features.to_csv(general.results_directory + "correlated_features.csv")

    explore_corr(sample,
                 correlated_features,
                 target,
                 visualization)
    train(sample,
          correlated_features,
          models,
          columns,
          preprocessing.target_column,
          sample_config.test_size,
          sample_config.random_seed,)

if __name__ == "__main__":
    main()