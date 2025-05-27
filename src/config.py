import json
import os
import datetime
import argparse
import config_class as cs
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parameters():
    config = cs.AnalysisConfig()
    args = parse_args()
    # check args for config file
    if not args.config:
        args.config = "config.json"
        logger.warning("No configuration file provided. Please provide a JSON configuration file. Using default configuration.")
    if not os.path.exists(args.config):
        logger.warning("Configuration file does not exist. Using default configuration.")

    config_dict = load_config(args.config)
    try:
        config = cs.AnalysisConfig(**config_dict)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.error("Please check the configuration file format and content.")
        raise
    processing_mode =config.general.processing_mode 
    if processing_mode == "shap" and config.preprocessing[0].input_file is None:
        logger.error("Processing mode is set to 'shap', but no input file is provided in the preprocessing configuration.")
        raise ValueError("Input file must be specified when processing mode is 'shap'.")
        
    if config.general.verbose:
        logger.info("Verbose mode is enabled. Detailed logs will be printed.")
        logger.info(f"Using configuration file: {args.config}")
        logger.info(f"Configuration version: {config.general.config_version}")
        logger.info(f"Configuration: {config}")
    # Generar carpetas para guardar resultados
    if not config.general.results_directory:
        config.general.results_directory = "results"
    config.general.results_directory = add_trailing_if_missing(config.general.results_directory)
    # Paths
    default_save_path = create_timestamped_folders(config.general.results_directory)

    config.models.models_directory = add_trailing_if_missing(config.models.models_directory)
    config.models.models_directory = f"{default_save_path}{config.models.models_directory}"
    create_folder(config.models.models_directory)

    config.visualization_config.figures_directory = add_trailing_if_missing(config.visualization_config.figures_directory)
    config.visualization_config.figures_directory = f"{default_save_path}{config.visualization_config.figures_directory}"
    create_folder(config.visualization_config.figures_directory)

    config.sample_config.output_sample_file = add_trailing_if_missing(config.sample_config.output_sample_file, char=".csv")
    config.sample_config.output_sample_file = f"{default_save_path}{config.sample_config.output_sample_file}"

    config.general.correlation_results_file = f"{default_save_path}{config.general.correlation_results_file}.csv"
    config.general.results_directory = add_trailing_if_missing(default_save_path)
    config.general.report_title = f"{config.general.report_title} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return config

def create_folder(directory_path: str):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created figures directory: {directory_path}")

def add_trailing_if_missing(default_save_path, char: str = "/"):
    if not default_save_path.endswith(char):
        default_save_path += char
    return default_save_path

def parse_args():
    parser = argparse.ArgumentParser(description="Parámetros de configuración del modelo")
    parser.add_argument("--config", type=str, default="config.json", help="Archivo de configuración JSON")
    return parser.parse_args()

def create_timestamped_folders(default_save_path: str):
    timestamp_time = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    default_save_path = f"{default_save_path}"
    create_folder(default_save_path)
    default_save_path = f"{default_save_path}/{timestamp_time}"
    create_folder(default_save_path)
    default_save_path = f"{default_save_path}/"
    return default_save_path


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)
