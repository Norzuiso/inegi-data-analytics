import logging
import pandas as pd
import numpy as np
from typing import List

from config import parameters
from config_class import AnalysisConfig, GeneralConfig, PreprocessingConfig, SampleConfig, VisualizationConfig,ModelsConfig
from data_utils import generate_sample, merge_files, file_processing, read_file
from visualization import explore_corr
from modeling import train, train_linear
from test import testing_code
# -*- coding: utf-8 -*-

# Configuración del logger
log = logging.getLogger(__name__)

from scipy.stats import fisher_exact

def epidemiological_table(df, target, variables, general):
    if(target in variables):
        variables.remove(target)
    
    resultados = []
    for var in variables:
        df[var] = pd.to_numeric(df[var], errors='coerce')
        df[target] = pd.to_numeric(df[target], errors='coerce')
        # Tabla de contingencia 2x2
        tab = pd.crosstab(df[var], df[target])
        # Odds ratio (fisher exact test para odds ratio univariado)
        if tab.shape == (2, 2):
            tab.to_csv()
            oddsratio, p = fisher_exact(tab)
            # Probabilidad del evento cuando variable=1
            prob_1 = tab.loc[1, 1] / tab.loc[1].sum() if 1 in tab.index else np.nan
            # Probabilidad del evento cuando variable=0
            prob_0 = tab.loc[0, 1] / tab.loc[0].sum() if 0 in tab.index else np.nan
            # Tasa de probabilidad (porcentaje)
            tasa_1 = prob_1 * 100 if not np.isnan(prob_1) else np.nan
            tasa_0 = prob_0 * 100 if not np.isnan(prob_0) else np.nan
            resultados.append({
                'variable': var,
                'odds_ratio': oddsratio,
                'p_value': p,
                'prob_evento_si_1': prob_1,
                'prob_evento_si_0': prob_0,
                'tasa_1': tasa_1,
                'tasa_0': tasa_0
            })
    return pd.DataFrame(resultados)

def epidemiological_table_v1(df, target, variables, general):
    if(target in variables):
        variables.remove(target)
    
    resultados = []
    for var in variables:
        # Tabla de contingencia 2x2
        print(f"var: {var} target:{target}\n")
        tab = pd.crosstab(df[var], df[target])
        # Odds ratio (fisher exact test para odds ratio univariado)
        if tab.shape == (2, 2):
            oddsratio, p = fisher_exact(tab)
            # Probabilidad del evento cuando variable=1
            prob_1 = tab.loc[1, 1] / tab.loc[1].sum() if 1 in tab.index else np.nan
            # Probabilidad del evento cuando variable=0
            prob_0 = tab.loc[0, 1] / tab.loc[0].sum() if 0 in tab.index else np.nan
            # Tasa de probabilidad (porcentaje)
            tasa_1 = prob_1 * 100 if not np.isnan(prob_1) else np.nan
            tasa_0 = prob_0 * 100 if not np.isnan(prob_0) else np.nan
            resultados.append({
                'variable': var,
                'odds_ratio': oddsratio,
                'p_value': p,
                'prob_evento_si_1': prob_1,
                'prob_evento_si_0': prob_0,
                'tasa_1': tasa_1,
                'tasa_0': tasa_0
            })
    return pd.DataFrame(resultados)

def prob_ajustada_por_habito(model, X, habits):
    base = X.median().to_dict()
    res = []
    for h in habits:
        base0 = base.copy()
        base1 = base.copy()
        base0[h] = 0
        base1[h] = 1
        p0 = model.predict_proba(pd.DataFrame([base0]))[:,1][0]
        p1 = model.predict_proba(pd.DataFrame([base1]))[:,1][0]
        res.append({
            'habito': h,
            'p_evento_si_habito_0': p0,
            'p_evento_si_habito_1': p1,
            'diferencia_probabilidad': p1 - p0
        })
    return pd.DataFrame(res)

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
    if processing_mode == "test":
        log.info("Processing mode is set to 'test'.")
        testing_code(config)
        return
    data = pd.DataFrame
    
    if processing_mode == "single":
        log.info("Only one preprocessing configuration provided. Using it for file processing.")
        preprocessing: PreprocessingConfig = preprocessing[0]
        data = file_processing(preprocessing)
        data.to_csv(f"{general.results_directory}data.csv")
    
    target = preprocessing.target_column.column_name
    columns = preprocessing.columns
        
    vars_binarias = [col.column_name for col in preprocessing.columns]

    data['A0408E'] = ((data['A0408A']== 5)).astype(int)
    data['A0408D'] = ((data['A0408A']== 4) | (data['A0408B']==4) | (data['A0408C']==4)| (data['A0408D']==4)).astype(int)
    data['A0408C'] = ((data['A0408A']== 3) | (data['A0408B']==3) | (data['A0408C']==3)).astype(int)
    data['A0408B'] = ((data['A0408A']== 2) | (data['A0408B']==2)).astype(int)
    data['A0408A'] = ((data['A0408A']== 1)).astype(int)

    sample = data
    sample.to_csv(config.sample_config.output_sample_file, index=False)

    resultados_univariados = epidemiological_table(sample, target, vars_binarias, general)
    resultados_univariados.to_csv(f"{general.results_directory}resultados_univariados.csv", index=False)

    corr_matrix = sample.corr(method=sample_config.correlation_method)
    cor_target = corr_matrix[target].abs()
    cor_target = cor_target.drop(target)
    correlated_features = cor_target[cor_target > sample_config.correlation_threshold].sort_values(ascending=False)
    correlated_features.to_csv(f"{general.results_directory}correlated_features.csv")


    modelo = train_linear(sample,
          correlated_features,
          models,
          columns,
          preprocessing.target_column,
          sample_config.test_size,
          sample_config.random_seed,)
    features = [col for col in correlated_features.index if col != target]
    X = sample[features]

    # Coeficientes (como log-odds)
    log_odds = modelo.coef_[0]
    # Odds Ratios (e^coef)
    odds_ratios = np.exp(log_odds)
    or_df = pd.DataFrame({
        'feature': X.columns,
        'log_odds': log_odds,
        'odds_ratio': odds_ratios
    })
    or_df.to_csv(f"{general.results_directory}resultados_logisticos.csv", index=False)

    print(or_df)

    tabla_prob = prob_ajustada_por_habito(modelo, X, features)
    tabla_prob.to_csv(f"{general.results_directory}resultados_probabilidades.csv", index=False)

if __name__ == "__main__":
    main()