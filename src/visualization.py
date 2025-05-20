import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from config_class import Graph_config


def explore_corr(sample: pd.DataFrame, 
                 correlated_features: pd.Series, 
                 target_column: str,
                 graph_config: Graph_config,
                 ):
    selected_cols = correlated_features.index.tolist()
    corr_matrix = sample[selected_cols].corr()
    corr_matrix = corr_matrix[correlated_features.index.tolist() + [target_column]].corr()
    mask = None
    if graph_config.hide_halfh_graph:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(graph_config.size_w, graph_config.size_h))
    sns.heatmap(corr_matrix, 
                mask=mask,
                fmt=graph_config.fmt,
                cbar=graph_config.color_bar,
                annot=graph_config.annot, 
                cmap=graph_config.color_map)
    plt.title(graph_config.graph_tittle)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(graph_config.save_fig_path, format=graph_config.img_format)
    if graph_config.show_plot: 
        plt.show()