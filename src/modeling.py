from generate_pdf_results import save_results_to_pdf
from config_class import Model_config, Config, Logistic_regression_config, Random_forest_config, SVC_config

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def raise_error_logistic_regression_params(params: Logistic_regression_config):
    if params is None:
        return None
    if params.solver not in ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']:
        raise ValueError("Solver must be one of ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']")
    if params.solver == 'liblinear' and params.penalty not in ['l1', 'l2']:
        raise ValueError("Penalty must be one of ['l1', 'l2'] when solver is 'liblinear'")
    if params.solver == 'saga' and params.penalty not in ['l1', 'l2', 'elasticnet', 'none']:
        raise ValueError("Penalty must be one of ['l1', 'l2', 'elasticnet', 'none'] when solver is 'saga'")
    if params.solver in ['lbfgs', 'newton-cg', 'sag'] and params.penalty not in ['l2', 'none']:
        raise ValueError("Penalty must be one of ['l2', 'none'] when solver is 'lbfgs', 'newton-cg' or 'sag'")
    if params.solver == 'liblinear' and params.intercept_scaling != 1:
        raise ValueError("Intercept scaling must be 1 when solver is 'liblinear'")
    if params.class_weight not in [None, 'balanced']:
        raise ValueError("Class weight must be None or 'balanced'")
    if params.C <= 0:
        raise ValueError("C must be greater than 0")
    if params.tol <= 0:
        raise ValueError("Tolerance must be greater than 0")
    if params.fit_intercept not in [True, False]:
        raise ValueError("Fit intercept must be True or False")
    if params.warm_start not in [True, False]:
        raise ValueError("Warm start must be True or False")

def train_logistic_regression(X_train, Y_train, X_test, Y_test, params: Logistic_regression_config, results_path: str):
    raise_error_logistic_regression_params(params)

    modelo = LogisticRegression(
        max_iter=params.max_iter,
        solver=params.solver,
        penalty=params.penalty,
        C=params.C,
        class_weight=params.class_weight,
        random_state=params.random_state,
        intercept_scaling=params.intercept_scaling,
        tol=params.tol,
        fit_intercept=params.fit_intercept,
        warm_start=params.warm_start
    )

    modelo.fit(X_train, Y_train)
    Y_pred = modelo.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    cl_report = classification_report(Y_test, Y_pred)
    conf_matrx = confusion_matrix(Y_test, Y_pred)

    joblib.dump(modelo, f"{results_path}LogisticRegression_{params.solver}_Penalty{params.penalty}_C{params.C}_MaxIter{params.max_iter}.pkl")
    results = {
        "Model": "LogisticRegression",
        "Accuracy": acc,
        "F1-Score": f1,
        "Classification Report": cl_report,
        "Confusion Matrix": conf_matrx
    }
    return results

def raise_error_random_forest_params(params: Random_forest_config):
    if params.criterion not in ["gini", "entropy", "log_loss"]:
        raise ValueError("Criterion must be 'gini', 'entropy' or 'log_loss'")
    valid_max_features = ["sqrt", "log2"]
    if not (
        params.max_features in valid_max_features
        or isinstance(params.max_features, (float, int))
        or params.max_features is None
    ):
        raise ValueError("Max features must be 'sqrt', 'log2', a float, an int, or None")
    if params.min_samples_split < 2:
        raise ValueError("Min samples split must be at least 2")
    if params.min_samples_leaf < 1:
        raise ValueError("Min samples leaf must be at least 1")
    if not (0 <= params.min_weight_fraction_leaf <= 0.9):
        raise ValueError("Min weight fraction leaf must be between 0 and 0.9")
    if params.max_depth is not None and params.max_depth < 1:
        raise ValueError("Max depth must be at least 1 or None")
    if params.max_leaf_nodes is not None and params.max_leaf_nodes < 1:
        raise ValueError("Max leaf nodes must be at least 1 or None")
    if params.max_samples is not None and params.max_samples < 1:
        raise ValueError("Max samples must be at least 1 or None")
    if params.oob_score not in [True, False]:
        raise ValueError("OOB score must be True or False")
    if params.bootstrap not in [True, False]:
        raise ValueError("Bootstrap must be True or False")
    if params.oob_score and not params.bootstrap:
        raise ValueError("Bootstrap must be True if OOB score is True")
    if params.n_estimators < 1:
        raise ValueError("Number of estimators must be at least 1")
    if params.random_state is not None and params.random_state < 0:
        raise ValueError("Random state must be a non-negative integer or None")
    if params.verbose < 0:
        raise ValueError("Verbose must be a non-negative integer")
    if params.class_weight not in [None, "balanced"]:
        raise ValueError("Class weight must be None or 'balanced'")


def train_random_forest(X_train, Y_train, X_test, Y_test, params: Random_forest_config, results_path: str):
    raise_error_random_forest_params(params)

    modelo = RandomForestClassifier(
        n_estimators=params.n_estimators,
        criterion=params.criterion,
        max_depth=params.max_depth,
        min_samples_split=params.min_samples_split,
        min_samples_leaf=params.min_samples_leaf,
        min_weight_fraction_leaf=params.min_weight_fraction_leaf,
        max_features=params.max_features,
        max_leaf_nodes=params.max_leaf_nodes,
        max_samples=params.max_samples,
        oob_score=params.oob_score,
        bootstrap=params.bootstrap,
        random_state=params.random_state,
        verbose=params.verbose,
        class_weight=params.class_weight
    )

    modelo.fit(X_train, Y_train)
    Y_pred = modelo.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    cl_report = classification_report(Y_test, Y_pred)
    conf_matrx = confusion_matrix(Y_test, Y_pred)

    joblib.dump(modelo, f"{results_path}RandomForest_{params.criterion}_n{params.n_estimators}_depth{params.max_depth}_rs{params.random_state}.pkl")
    results = {
        "Model": "RandomForest",
        "Accuracy": acc,
        "F1-Score": f1,
        "Classification Report": cl_report,
        "Confusion Matrix": conf_matrx
    }
    return results
def raise_error_svc_params(params: SVC_config):
    if params.kernel not in ["linear", "poly", "rbf", "sigmoid", "precomputed"]:
        raise ValueError("Kernel must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'")
    if params.C <= 0:
        raise ValueError("C must be greater than 0")
    if params.gamma not in ["scale", "auto"] and not isinstance(params.gamma, (float, int)):
        raise ValueError("Gamma must be 'scale', 'auto', float or int")
    if params.degree < 1:
        raise ValueError("Degree must be at least 1")
    if params.tol <= 0:
        raise ValueError("Tolerance must be greater than 0")
    if params.class_weight not in [None, "balanced"]:
        raise ValueError("Class weight must be None or 'balanced'")
    if params.max_iter != -1 and params.max_iter < 1:
        raise ValueError("Max iter must be -1 (no limit) or a positive integer")
    if params.decision_function_shape not in ["ovr", "ovo"]:
        raise ValueError("Decision function shape must be 'ovr' or 'ovo'")
    if params.cache_size < 1:
        raise ValueError("Cache size must be at least 1")

def train_svc(X_train, Y_train, X_test, Y_test, params: SVC_config, results_path: str):
    raise_error_svc_params(params)
    svc_kwargs = dict(
        kernel=params.kernel,
        C=params.C,
        gamma=params.gamma,
        degree=params.degree,
        random_state=params.random_state,
        coef0=params.coef0,
        probability=params.probability,
        shrinking=params.shrinking,
        tol=params.tol,
        class_weight=params.class_weight,
        max_iter=params.max_iter,
        decision_function_shape=params.decision_function_shape,
        break_ties=params.break_ties,
        cache_size=params.cache_size,
        verbose=params.verbose
    )
    # NO incluyas intercept_scaling
    modelo = SVC(**svc_kwargs)

    modelo.fit(X_train, Y_train)
    Y_pred = modelo.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    cl_report = classification_report(Y_test, Y_pred)
    conf_matrx = confusion_matrix(Y_test, Y_pred)

    joblib.dump(modelo, f"{results_path}SVC_{params.kernel}_C{params.C}_Gamma{params.gamma}_Degree{params.degree}.pkl")
    results = {
        "Model": "SVC",
        "Accuracy": acc,
        "F1-Score": f1,
        "Classification Report": cl_report,
        "Confusion Matrix": conf_matrx
    }
    return results

def train(sample: pd.DataFrame,
          correlated_features: pd.Series,
          model_config: Model_config,
          config: Config,
          columns: list = [],
          target: str = "target",
          test_size: float = 0.2,
          random_state: int = 42):
    # Models
    logistic_regression_params = model_config.logistic_regression_config
    random_forest_params = model_config.random_forest_config
    svc_params = model_config.svc_config

    # Ensure the directory exists
    import os
    os.makedirs(config.results_path, exist_ok=True)

    # DATASET
    if not isinstance(correlated_features, pd.Series):
        raise ValueError("correlated_features must be a pandas Series.")
    if not hasattr(correlated_features.index, '__iter__'):
        raise ValueError("correlated_features.index must be iterable.")

    # Initialize results list
    results = []
   
    features = [col for col in correlated_features.index if col != target]
    if not features:
        raise ValueError("No features found for training.")
    X = sample[features]
    Y = sample[target]
    if target in X.columns:
        raise ValueError("Target column was found in X — this will invalidate your training.")

    # Split de los datos
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=random_state
    )

    # Normalización
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize results list
    results = []

    # Entrenamiento de los modelos
    if logistic_regression_params is not None:
        try:
            logistic = train_logistic_regression(X_train, Y_train, X_test, Y_test, logistic_regression_params, config.results_path)
            results.append(logistic)
        except Exception as e:
            results.append({"Model": "LogisticRegression", "Error": str(e)})
    else:
        results.append({"Model": "LogisticRegression", "Error": "No configuration provided"})
    
    if random_forest_params is not None:
        try:
            random_forest = train_random_forest(X_train, Y_train, X_test, Y_test, random_forest_params, config.results_path)
            results.append(random_forest)
        except Exception as e:
            results.append({"Model": "RandomForest", "Error": str(e)})
    else:
        results.append({"Model": "RandomForest", "Error": "No configuration provided"})
    if svc_params is not None:
        try:
            svc = train_svc(X_train, Y_train, X_test, Y_test, svc_params, config.results_path)
            results.append(svc)
        except Exception as e:
            results.append({"Model": "SVC", "Error": str(e)})
    else:
        results.append({"Model": "SVC", "Error": "No configuration provided"})
    
    # Guardar el scaler
    joblib.dump(scaler, f"{config.results_path}scaler.pkl")
    # Guardar los resultados en un PDF
    save_results_to_pdf(results, f"{config.results_path}results.pdf", columns, config.report_tittle)

