from typing import List, Optional, Literal, Union, Annotated
from pydantic import BaseModel, Field, ValidationError, conint, confloat, model_validator
import logging

# Configuración del logger
log = logging.getLogger(__name__)

# ==== PREPROCESAMIENTO ====
class Column(BaseModel):
    column_name: str = ""
    condition: Optional[str] = ""
    true_value: Optional[str] = "1"
    false_value: Optional[str] = "0"
    value_type: Literal["int", "str", "float"] = "int"

class PreprocessingConfig(BaseModel):
    input_file: str = "data.csv"
    target_column: Optional[Column] = None
    columns: List[Column] = Field(default_factory=list)

    @model_validator(mode='before')
    def validate_conditions(cls, values):
        if not values.get('input_file'):
            raise ValueError("input_file must be specified")
        if not values.get('columns'):
            log.warning("columns is empty, preprocessing all columns without conditions")
        return values

# ==== DATOS Y MUESTREO ====
class SampleConfig(BaseModel):
    sample_size: int = 1000
    discriminator: int = 1
    test_size: float = 0.2
    correlation_threshold: float = 0.1 # percentage_per_corr
    random_seed: int = 42
    output_sample_file: str = "sample.csv"
    correlation_method: str = "pearson"

# ==== VISUALIZACIÓN ====
class VisualizationConfig(BaseModel):
    title: str = "Correlaciones" 
    show_annotations: bool = False
    show_colorbar: bool = True
    color_map: str = "coolwarm"
    number_format: str = ".2f"
    hide_upper_half: bool = True
    figure_name: str = "default_graph"
    image_format: str = "svg"
    display_plot: bool = False
    width: int = 18
    height: int = 12
    figures_directory: str = "figures/"

# ==== MODELOS ====
class LogisticRegressionConfig(BaseModel):
    C: Annotated[float, Field(strict=False, gt=0)] = 1.0
    class_weight: Literal[None, "balanced"] = None
    fit_intercept: bool = True
    intercept_scaling: int = 1
    max_iter: int = 1000
    multiclass: str = "auto"
    penalty: Union[Literal['l1', 'l2', 'elasticnet', 'none'], None] = 'l2'
    random_seed: Annotated[int, Field(strict=False, ge=0)] = 42
    solver: Literal['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'] = 'lbfgs'
    tol: Annotated[float, Field(strict=False, gt=0)] = 1e-4
    warm_start: bool = False
    
    @model_validator(mode='before')
    def check_penalty_for_solver(cls, values):
        solver = values.get('solver', 'lbfgs')
        penalty = values.get('penalty', 'l2')
        if solver == 'liblinear' and penalty not in ['l1', 'l2']:
            raise ValueError("Penalty must be one of ['l1', 'l2'] when solver is 'liblinear'")
        if solver == 'saga' and penalty not in ['l1', 'l2', 'elasticnet', 'none']:
            raise ValueError("Penalty must be one of ['l1', 'l2', 'elasticnet', 'none'] when solver is 'saga'")
        if solver in ['lbfgs', 'newton-cg', 'sag'] and penalty not in ['l2', 'none']:
            raise ValueError("Penalty must be one of ['l2', 'none'] when solver is 'lbfgs', 'newton-cg' or 'sag'")
        if solver == 'liblinear' and values.get('intercept_scaling', 1) != 1:
            raise ValueError("Intercept scaling must be 1 when solver is 'liblinear'")
        return values

class RandomForestConfig(BaseModel):
    bootstrap: bool = True
    ccp_alpha: Annotated[float, Field(strict=False, ge=0)] = 0.0
    class_weight: Optional[Literal[None, "balanced"]] = None 
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    max_depth: Optional[Annotated[int, Field(strict=False, ge=1)]] = 10 
    max_features: Union[Literal["sqrt", "log2"], float, int, None] = "sqrt" 
    max_leaf_nodes: Optional[Annotated[int, Field(strict=False, ge=1)]] = None
    max_samples: Optional[Annotated[int, Field(strict=False, ge=1)]] = None
    min_samples_leaf: Annotated[int, Field(strict=False, ge=1)] = 1
    min_samples_split: Annotated[int, Field(strict=False, ge=2)] = 2
    min_weight_fraction_leaf: Optional[Annotated[float, Field(strict=False, ge=0.0, le=0.9)]] = 0.0
    n_estimators: Annotated[int, Field(strict=False, ge=1)] = 100
    oob_score: bool = False
    random_seed: Optional[Annotated[int, Field(strict=False, ge=0)]] = 42
    verbose: Annotated[int, Field(strict=False, ge=0)] = 0

    @model_validator(mode='before')
    def check_oob_requires_bootstrap(cls, values):
        bootstrap = values.get('bootstrap', True)
        oob_score = values.get('oob_score', False)
        if oob_score and not bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True")
        return values

class SVCConfig(BaseModel):
    break_ties: bool = False
    C: Annotated[float, Field(strict=False, gt=0)] = 1.0
    cache_size: Annotated[int, Field(strict=False, ge=1)] = 200
    class_weight: Literal[None, "balanced"] = None
    coef0: Annotated[float, Field(strict=False, ge=0)] = 0.0
    decision_function_shape: Literal["ovr", "ovo"] = "ovr"
    degree: Annotated[int, Field(strict=False, ge=1)] = 3
    gamma: Annotated[Union[float, int, Literal["scale", "auto"]], Field()] = "scale"
    kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "linear"
    max_iter: Annotated[int, Field(strict=False, ge=-1)] = -1
    probability: bool = False
    random_seed: Optional[Annotated[int, Field(strict=False, ge=0)]] = 42
    shrinking: bool = True
    tol: Annotated[float, Field(strict=False, gt=0)] = 1e-4
    verbose: bool = False

    @model_validator(mode='before')
    def validate_kernel_params(cls, values):
        kernel = values.get('kernel', 'linear')
        coef0 = values.get('coef0', 0.0)
        gamma = values.get('gamma', 'scale')
        if kernel in ["linear", "rbf"] and coef0 != 0.0:
            raise ValueError(f"coef0 must be 0.0 for kernel '{kernel}'")
        if kernel == "linear" and gamma not in ["scale", "auto"]:
            raise ValueError("gamma must be 'scale' or 'auto' for linear kernel")
        if kernel == "sigmoid" and coef0 < 0:
            raise ValueError("coef0 must be non-negative for sigmoid kernel")
        if kernel == "poly" and coef0 < 0:
            raise ValueError("coef0 must be non-negative for polynomial kernel")
        if kernel == "precomputed":
            if gamma not in ["scale", "auto"]:
                raise ValueError("gamma must be 'scale' or 'auto' for precomputed kernel")
            if coef0 != 0.0:
                raise ValueError("coef0 must be 0.0 for precomputed kernel")

        return values

class XGBoostConfig(BaseModel):
    n_estimators: Annotated[int, Field(strict=False, ge=1)] = 100
    learning_rate: Annotated[float, Field(strict=False, gt=0, le=1)] = 0.1
    max_depth: Annotated[int, Field(strict=False, ge=1)] = 6
    min_child_weight: Annotated[int, Field(strict=False, ge=0)] = 1
    subsample: Annotated[float, Field(strict=False, gt=0, le=1)] = 1.0
    colsample_bytree: Annotated[float, Field(strict=False, gt=0, le=1)] = 1.0
    gamma: Annotated[float, Field(strict=False, ge=0)] = 0.0
    reg_alpha: Annotated[float, Field(strict=False, ge=0)] = 0.0
    reg_lambda: Annotated[float, Field(strict=False, ge=0)] = 1.0
    objective: Literal[
        "binary:logistic", "binary:hinge",
        "multi:softmax", "multi:softprob",
        "reg:squarederror", "reg:squaredlogerror"
    ] = "binary:logistic"
    booster: Literal["gbtree", "gblinear", "dart"] = "gbtree"
    random_seed: Annotated[int, Field(strict=False, ge=0)] = 42
    verbosity: Annotated[int, Field(strict=False, ge=0, le=3)] = 1

    @model_validator(mode='before')
    def check_combo(cls, values):
        obj = values.get('objective', 'objective')
        bst = values.get('booster', 'gbtree')
        if obj.startswith('multi') and bst == 'gblinear':
            raise ValueError("gblinear no soporta objetivos multi*")
        return values

class RidgeConfig(BaseModel):
    alpha: Annotated[float, Field(strict=False, ge=0)] = 1.0
    fit_intercept: bool = True
    normalize: bool = False
    solver: Literal[
        "auto", "svd", "cholesky", "lsqr",
        "sparse_cg", "sag", "saga"
    ] = "auto"
    tol: Annotated[float, Field(strict=False, gt=0)] = 1e-4
    random_seed: Annotated[int, Field(strict=False, ge=0)] = 42

class ModelsConfig(BaseModel):
    models_directory: str = "models/"
    logistic_regression: LogisticRegressionConfig = Field(default_factory=LogisticRegressionConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    svc: SVCConfig = Field(default_factory=SVCConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    ridge: RidgeConfig = Field(default_factory=RidgeConfig)

# ==== CONFIGURACIÓN DEL SISTEMA ====
class MergeConfig(BaseModel):
    target_column: Optional[Column] = None
    file_for_target: str = "data.csv"
    merge_columns: Optional[List[str]] = None

class GeneralConfig(BaseModel):
    merge_config: MergeConfig = Field(default_factory=MergeConfig)
    results_directory: str = "./results"
    correlation_results_file: str = "correlated_features.csv"
    report_title: str = "Modelo de predicción de hipertensión basado en ENSANUT"
    processing_mode: Literal["merge", "single", "exploration"] = "single"
    verbose: bool = False  # Modo detallado
    config_version: str = "1.0.0"

    @model_validator(mode='before')
    def validate_merge_columns(cls, values):
        mode = values.get('processing_mode', 'single')
        merge_columns = values.get('merge_columns', None)
        target_column = values.get('target_column', None)
        if mode == 'merge' and not merge_columns:
            raise ValueError("merge_columns must be specified when processing_mode is 'merge'")
        if mode == 'single' and merge_columns:
            raise ValueError("merge_columns should not be specified when processing_mode is 'single'")
        if mode == 'merge' and target_column is None:
            raise ValueError("target_column must be specified when processing_mode is 'merge'")
        if mode == 'single' and target_column is not None:
            raise ValueError("target_column should not be specified when processing_mode is 'single'")
        
        return values

class AnalysisConfig(BaseModel):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    preprocessing: List[PreprocessingConfig] = Field(default_factory=list)

    sample_config: SampleConfig = Field(default_factory=SampleConfig)
    visualization_config: VisualizationConfig = Field(default_factory=VisualizationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    