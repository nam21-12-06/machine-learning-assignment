import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocessing import get_preprocessor


def build_linear_svm_pipeline(
    numerical_features,
    categorical_features,
    scaler_type="standard",
    pca_variance=None,
    C=1.0,
    class_weight=None,
    random_state=42,
    min_category_frequency=20,
):
    """Build a leakage-safe preprocessing and LinearSVC pipeline."""
    if pca_variance not in (None, 0.90, 0.95):
        raise ValueError("pca_variance must be None, 0.90, or 0.95")
    if C <= 0:
        raise ValueError("C must be greater than zero")

    preprocessor = get_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        scaler_type=scaler_type,
        log_numeric=False,
        min_category_frequency=min_category_frequency,
    )

    steps = [("preprocessor", preprocessor)]
    if pca_variance is not None:
        steps.append(
            (
                "pca",
                PCA(n_components=pca_variance, svd_solver="full"),
            )
        )

    steps.append(
        (
            "svm",
            LinearSVC(
                C=C,
                class_weight=class_weight,
                dual="auto",
                max_iter=20000,
                random_state=random_state,
            ),
        )
    )
    return Pipeline(steps)


def run_svm_cross_validation(
    X_train,
    y_train,
    groups_train,
    group_cv,
    configs,
    numerical_features,
    categorical_features,
    scoring,
    random_state=42,
    n_jobs=2,
    min_category_frequency=20,
):
    """Evaluate SVM configurations using grouped cross-validation."""
    if not configs:
        raise ValueError("configs must contain at least one SVM configuration")

    required_keys = {"name", "scaler_type", "pca_variance", "C", "class_weight"}
    rows = []

    for config in configs:
        missing_keys = required_keys.difference(config)
        if missing_keys:
            raise ValueError(
                f"Configuration is missing required keys: {sorted(missing_keys)}"
            )

        model = build_linear_svm_pipeline(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            scaler_type=config["scaler_type"],
            pca_variance=config["pca_variance"],
            C=config["C"],
            class_weight=config["class_weight"],
            random_state=random_state,
            min_category_frequency=min_category_frequency,
        )

        result = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            groups=groups_train,
            cv=group_cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=n_jobs,
            error_score="raise",
        )

        rows.append(
            {
                "configuration": config["name"],
                "scaler": config["scaler_type"],
                "pca": (
                    "None"
                    if config["pca_variance"] is None
                    else f"{int(config['pca_variance'] * 100)}%"
                ),
                "C": config["C"],
                "class_weight": str(config["class_weight"]),
                "cv_accuracy_mean": result["test_accuracy"].mean(),
                "cv_balanced_accuracy_mean": result[
                    "test_balanced_accuracy"
                ].mean(),
                "cv_precision_macro_mean": result["test_precision_macro"].mean(),
                "cv_recall_macro_mean": result["test_recall_macro"].mean(),
                "cv_f1_macro_mean": result["test_f1_macro"].mean(),
                "cv_f1_macro_std": result["test_f1_macro"].std(),
                "cv_f1_weighted_mean": result["test_f1_weighted"].mean(),
                "fit_time_mean": result["fit_time"].mean(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(by="cv_f1_macro_mean", ascending=False)
        .reset_index(drop=True)
    )
