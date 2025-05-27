import logging
import pandas as pd

from typing import List

from config import parameters
from config_class import AnalysisConfig, GeneralConfig, PreprocessingConfig, SampleConfig, VisualizationConfig,ModelsConfig
from data_utils import generate_sample, merge_files, file_processing

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

    
    target = preprocessing.target_column
    columns = preprocessing.columns
    
#    graph_config = config.graph_config
#    model_config = config.model_config
    
    sample = generate_sample(data, target.column_name, sample_config)   
    print(sample.describe())
    sample.to_csv(config.sample_config.output_sample_file, index=False)
#    
#    corr_matrix = sample.corr()
#    cor_target = corr_matrix[target].abs()
#    cor_target = cor_target.drop(target)
#    correlated_features = cor_target[cor_target > sample_config.percentage_per_corr].sort_values(ascending=False)
#    correlated_features.to_csv(config.corr_store_file)
#
#    explore_corr(sample,
#                 correlated_features,
#                 target,
#                 graph_config)
#
#    train(sample,
#          correlated_features,
#          model_config,
#          config,
#          columns,
#          target,
#          sample_config.test_size,
#          random_state=sample_config.random_state,)

if __name__ == "__main__":
    main()