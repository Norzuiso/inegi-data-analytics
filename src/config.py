import json
import os
import datetime
import argparse

from config_class import Config

def parameters():
# -------------------------------------
#    Construct all the parameters
# -------------------------------------
    # File CSV
    config = Config()
    
    args = parse_args()
    config.file = args.file if args.file else config.file
    config.target_column = args.target if args.target else config.target_column
    config.columns = args.columns if args.columns else config.columns

    config.graph_config = graph_config
        
    if not args.config:
        raise ValueError("No se ha proporcionado un archivo de configuración. Por favor, proporciona el archivo de configuración JSON.")
    
    # Cargar y aplicar configuración desde el JSON
    config_dict = load_config(args.config)
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    # Generar carpetas para guardar resultados
    graph_config = config.graph_config

    # Paths
    default_save_path = config.default_save_path
    default_save_path = generate_folders(default_save_path)
    graph_config.save_fig_path = f"{default_save_path}{graph_config.figname}.{graph_config.img_format}"

    config.corr_store_file = f"{default_save_path}{config.corr_store_file}.csv"
    config.results_path = f"{default_save_path}/"
    
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Modelo de predicción de hipertensión basado en ENSANUT.")
    parser.add_argument("--file", type=str, help="Ruta del archivo CSV.")
    parser.add_argument("--target", type=str, help="Columna objetivo (target).")
    parser.add_argument("--columns", nargs="+", help="Lista de columnas específicas a usar.")
    parser.add_argument("--config", type=str, default="config.json", help="Archivo de configuración JSON")
    return parser.parse_args()

def generate_folders(default_save_path: str):
    timestamp_time = f"{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/"
    default_save_path = f"{default_save_path}"
    if not os.path.exists(default_save_path):
        os.mkdir(default_save_path)
    default_save_path = f"{default_save_path}/{timestamp_time}"
    if not os.path.exists(default_save_path):
        os.mkdir(default_save_path)
    default_save_path = f"{default_save_path}/"
    return default_save_path


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)
