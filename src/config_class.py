from dataclasses import dataclass, field
    

@dataclass
class Sample_config:
    sample_size: int = 3000
    pos_condition: int = 1
    neg_condition: int = 2
    test_size: float = 0.2
    percentage_per_corr: float = 0.1
    random_state: int = 42
    sample_file: str = "sample.csv"
    CorrelationMethod: str = "pearson"

@dataclass
class Graph_config:
    graph_tittle: str = "Correlaciones"
    annot: bool = False
    color_bar: bool = True
    color_map: str = "coolwarm"
    fmt: str = ".2f"
    hide_halfh_graph: bool = True
    figname: str = "correlaciones_heatmap"
    img_format: str = "svg"
    show_plot: bool = False
    size_w: int = 18
    size_h: int = 12
    save_fig_path: str = "figures/"

@dataclass
class Logistic_regression_config:
    max_iter: int = 1000
    solver: str = "liblinear"
    penalty: str = "l2"
    C: float = 1.0
    class_weight: str = "balanced"
    multiclass: str = "auto"
    random_state: int = 42
    intercept_scaling: int = 1
    tol: float = 0.0001
    fit_intercept: bool = True
    warm_start: bool = False

@dataclass
class Random_forest_config:
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    random_state: int = 42
    bootstrap: bool = True
    criterion: str = "gini"
    max_leaf_nodes: None = None
    min_weight_fraction_leaf: float = 0.0
    oob_score: bool = False
    verbose: int = 0
    class_weight: None = None
    max_samples: None = None

@dataclass
class SVC_config:
    kernel: str = "linear"
    C: float = 1.0
    gamma: str = "scale"
    degree: int = 3
    random_state: int = 42
    coef0: float = 0.0
    probability: bool = False
    shrinking: bool = True
    tol: float = 0.001
    class_weight: str = None
    max_iter: int = -1
    decision_function_shape: str = "ovr"
    break_ties: bool = False
    cache_size: int = 200
    verbose: bool = False

@dataclass
class Model_config:
    logistic_regression_config: Logistic_regression_config = field(default_factory=Logistic_regression_config)
    random_forest_config: Random_forest_config = field(default_factory=Random_forest_config)
    svc_config: SVC_config = field(default_factory=SVC_config)

@dataclass
class Config:
    file: str = "CS_ADULTOS.csv"
    target_column: str = "P4_1"
    columns: list = field(default_factory=list)
    default_save_path: str = "./results"
    results_path: str = "correlaciones_heatmap.png"
    report_tittle: str = "Modelo de predicción de hipertensión basado en ENSANUT"
    corr_store_file: str = "features_correlacionadas"
    sample_config: Sample_config = field(default_factory=Sample_config)
    graph_config: Graph_config = field(default_factory=Graph_config)
    model_config: Model_config = field(default_factory=Model_config)
