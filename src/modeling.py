from generate_pdf_results import save_results_to_pdf
from config_class import ModelConfig, Config, Logistic_regression_config, Random_forest_config, SVC_config

import joblib
import pandas as pd
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def train_logistic_regression(X_train, Y_train, X_test, Y_test, params: Logistic_regression_config, results_path: str):
    print(f"Training Logistic Regression with parameters: {params}")

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
    coef = modelo.coef_  # Reshape to 2D array for consistency
    modelo.intercept_ = modelo.intercept_.reshape(1, -1)  # Reshape to 2D array for consistency
    # Save the array to a CSV file
    with open(f"{results_path}logistic_regression_coefficients.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(coef)

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
    # Guardar los coeficientes del modelo
    
    store_coefficients(modelo, X_train, f"{results_path}logistic_regression_coefficients.csv")
    return results


def train_random_forest(X_train, Y_train, X_test, Y_test, params: Random_forest_config, results_path: str):
    print(f"Training Random Forest with parameters: {params}")
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

def train_svc(X_train, Y_train, X_test, Y_test, params: SVC_config, results_path: str):
    print(f"Training SVC with parameters: {params}")
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
          model_config: ModelConfig,
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
            coeficientes = pd.Series(random_forest.feature_importances_, index=X_train.columns)
            print(coeficientes)
            coeficientes.to_csv(f"{config.results_path}random_forest_feature_importances.csv")
        except Exception as e:
            results.append({"Model": "RandomForest", "Error": str(e)})
    else:
        results.append({"Model": "RandomForest", "Error": "No configuration provided"})
    if svc_params is not None:
        try:
            svc = train_svc(X_train, Y_train, X_test, Y_test, svc_params, config.results_path)
            results.append(svc)
            coeficientes = pd.Series(svc.feature_importances_, index=X_train.columns)
            print(coeficientes)
            coeficientes.to_csv(f"{config.results_path}svc_feature_importances.csv")
        except Exception as e:
            results.append({"Model": "SVC", "Error": str(e)})
    else:
        results.append({"Model": "SVC", "Error": "No configuration provided"})
    
    # Guardar el scaler
    joblib.dump(scaler, f"{config.results_path}scaler.pkl")
    # Guardar los resultados en un PDF
    save_results_to_pdf(results, f"{config.results_path}results.pdf", columns, config.report_tittle)

