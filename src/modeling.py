from generate_pdf_results import save_results_to_pdf
from config_class import Column, ModelsConfig, LogisticRegressionConfig, RandomForestConfig, SVCConfig, XGBoostConfig, RidgeConfig

import joblib
import pandas as pd
import csv
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
import shap
from pathlib import Path
import matplotlib.pyplot as plt

def train_logistic_regression(X_train, Y_train, X_test, Y_test, params: LogisticRegressionConfig, results_path: str):
    print(f"Training Logistic Regression with parameters: {params}")
    modelo = LogisticRegression(
        max_iter=params.max_iter,
        solver=params.solver,
        penalty=params.penalty,
        C=params.C,
        class_weight=params.class_weight,
        random_state=params.random_seed,
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

    return results


def train_random_forest(X_train, Y_train, X_test, Y_test, params: RandomForestConfig, results_path: str):
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
        random_state=params.random_seed,
        verbose=params.verbose,
        class_weight=params.class_weight
    )

    modelo.fit(X_train, Y_train)
    Y_pred = modelo.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    cl_report = classification_report(Y_test, Y_pred)
    conf_matrx = confusion_matrix(Y_test, Y_pred)

    joblib.dump(modelo, f"{results_path}RandomForest_{params.criterion}_n{params.n_estimators}_depth{params.max_depth}_rs{params.random_seed}.pkl")
    results = {
        "Model": "RandomForest",
        "Accuracy": acc,
        "F1-Score": f1,
        "Classification Report": cl_report,
        "Confusion Matrix": conf_matrx
    }
    explain_with_shap(modelo, X_train, X_test, results_path, "RandomForest")
    return results

def train_svc(X_train, Y_train, X_test, Y_test, params: SVCConfig, results_path: str):
    print(f"Training SVC with parameters: {params}")
    svc_kwargs = dict(
        kernel=params.kernel,
        C=params.C,
        gamma=params.gamma,
        degree=params.degree,
        random_state=params.random_seed,
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


def train_xgboost(X_train, Y_train, X_test, Y_test, params: XGBoostConfig, results_path: str):
    print(f"Training XGBoost with parameters: {params}")
    modelo = XGBClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        gamma=params.gamma,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        random_state=params.random_seed,
        use_label_encoder=False,  # Para evitar el warning de XGBoost
        eval_metric='logloss'  # Puedes cambiar esto según tu métrica preferida
    )

    modelo.fit(X_train, Y_train)
    Y_pred = modelo.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    cl_report = classification_report(Y_test, Y_pred)
    conf_matrx = confusion_matrix(Y_test, Y_pred)

    joblib.dump(modelo, f"{results_path}XGBoost_n{params.n_estimators}_depth{params.max_depth}_lr{params.learning_rate}.pkl")
    results = {
        "Model": "XGBoost",
        "Accuracy": acc,
        "F1-Score": f1,
        "Classification Report": cl_report,
        "Confusion Matrix": conf_matrx
    }
    return results

def train_ridge(X_train, Y_train, X_test, Y_test, params: RidgeConfig, results_path: str):
    print(f"Training Ridge Regression with parameters: {params}")
    modelo = RidgeClassifier(
        alpha=params.alpha,
        fit_intercept=params.fit_intercept,
        tol=params.tol,
        solver=params.solver,
        random_state=params.random_seed
    )

    modelo.fit(X_train, Y_train)
    Y_pred = modelo.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    cl_report = classification_report(Y_test, Y_pred)
    conf_matrx = confusion_matrix(Y_test, Y_pred)

    joblib.dump(modelo, f"{results_path}RidgeClassifier_alpha{params.alpha}_solver{params.solver}.pkl")
    results = {
        "Model": "RidgeClassifier",
        "Accuracy": acc,
        "F1-Score": f1,
        "Classification Report": cl_report,
        "Confusion Matrix": conf_matrx
    }
    return results


def train(sample: pd.DataFrame,
          correlated_features: pd.Series,
          models: ModelsConfig,
          columns: list = [],
          target: Column = Column(column_name="target"),
          test_size: float = 0.2,
          random_seed: int = 42):
    # Models
    logistic_regression_params = models.logistic_regression
    random_forest_params = models.random_forest
    svc_params = models.svc
    xgboost_params = models.xgboost
    ridge_params = models.ridge
    target_name = target.column_name
    results_path = models.models_directory

    # Ensure the directory exists
    import os
    os.makedirs(models.models_directory, exist_ok=True)

    # DATASET
    if not isinstance(correlated_features, pd.Series):
        raise ValueError("correlated_features must be a pandas Series.")
    if not hasattr(correlated_features.index, '__iter__'):
        raise ValueError("correlated_features.index must be iterable.")

    # Initialize results list
    results = []
   
    features = [col for col in correlated_features.index if col != target_name]
    if not features:
        raise ValueError("No features found for training.")
    X = sample[features]
    Y = sample[target_name]
    if target_name in X.columns:
        raise ValueError("Target column was found in X — this will invalidate your training.")

    # Split de los datos
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=random_seed
    )

    # Normalización
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize results list
    results = []

    # Entrenamiento de los modelos
    if logistic_regression_params is not None:
            logistic = train_logistic_regression(X_train, Y_train, X_test, Y_test, logistic_regression_params, results_path)
            results.append(logistic)
    else:
        results.append({"Model": "LogisticRegression", "Error": "No configuration provided"})
    
    if random_forest_params is not None:
        random_forest = train_random_forest(X_train, Y_train, X_test, Y_test, random_forest_params, results_path)
        results.append(random_forest)
    else:
        results.append({"Model": "RandomForest", "Error": "No configuration provided"})

    if svc_params is not None:
        svc = train_svc(X_train, Y_train, X_test, Y_test, svc_params, results_path)
        results.append(svc)
    else:
        results.append({"Model": "SVC", "Error": "No configuration provided"})
    

    if ridge_params is not None:
        ridge = train_ridge(X_train, Y_train, X_test, Y_test, ridge_params, results_path)
        results.append(ridge)
    else:
        results.append({"Model": "RidgeClassifier", "Error": "No configuration provided"})
    # Guardar los resultados en un CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{results_path}results.csv", index=False)

    # Guardar el scaler
    joblib.dump(scaler, f"{results_path}scaler.pkl")
    
    # Guardar los resultados en un PDF
    save_results_to_pdf(results, f"{results_path}results.pdf", columns, target_name)
    print("Training completed. Results saved to CSV and PDF.")

def explain_with_shap(modelo, X_train, X_test, results_path, model_name):
    """
    Genera valores SHAP y guarda un summary_plot en PNG.
    """
    out = Path(results_path)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Elige el explainer según el modelo
    if hasattr(modelo, "estimators_") or hasattr(modelo, "feature_importances_"):
        # RandomForest, XGBoost, etc.
        explainer = shap.TreeExplainer(modelo)
    elif isinstance(modelo, LogisticRegression):
        explainer = shap.LinearExplainer(modelo, X_train, feature_dependence="independent")
    else:
        # Caerá a un método universal (más lento)
        explainer = shap.KernelExplainer(modelo.predict_proba, shap.sample(X_train, 100))

    # 2) Calcula los valores SHAP para el set de prueba
    shap_values = explainer(X_test)

    print(f"SHAP values calculated for {model_name} model.")
    print(shap_values)
    
    # 3) Genera y guarda el summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(out / f"{model_name}_shap_summary.svg", dpi=150)
    plt.close()
