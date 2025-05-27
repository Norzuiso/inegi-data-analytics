import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from config_class import VisualizationConfig


def explore_corr(sample: pd.DataFrame, 
                 correlated_features: pd.Series, 
                 target_column: str,
                 visualization_config: VisualizationConfig,
                 ):
    selected_cols = correlated_features.index.tolist()
    corr_matrix = sample[selected_cols].corr()
    if target_column in corr_matrix.columns:
        corr_matrix = corr_matrix.drop(target_column, axis=1)
    corr_matrix = corr_matrix[correlated_features.index.tolist()].corr()
    mask = None
    if visualization_config.hide_upper_half:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(visualization_config.width, visualization_config.height))
    sns.heatmap(corr_matrix, 
                mask=mask,
                fmt=visualization_config.number_format,
                cbar=visualization_config.show_colorbar,
                annot=visualization_config.show_annotations, 
                cmap=visualization_config.color_map)
    plt.title(visualization_config.title)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(visualization_config.figures_directory + visualization_config.figure_name + "." + visualization_config.image_format, format=visualization_config.image_format)
    if visualization_config.display_plot: 
        plt.show()