import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from .preprocessing import get_preprocessor


def build_rf_pipeline(
    numerical_features,
    categorical_features,
    scaler_type="standard",
    feature_range=(0, 1),
    pca_variance=None,
    n_estimators=100,
    max_depth=14,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    model_n_jobs=-1,
    min_category_frequency=20,
):
    """Build the project's preprocessing and Random Forest pipeline."""
    if pca_variance not in (None, 0.90, 0.95):
        raise ValueError("pca_variance must be None, 0.90, or 0.95")

    steps = [
        (
            "preprocessor",
            get_preprocessor(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                scaler_type=scaler_type,
                feature_range=feature_range,
                log_numeric=False,
                min_category_frequency=min_category_frequency,
            ),
        )
    ]

    if pca_variance is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_variance,
                    svd_solver="full",
                    random_state=random_state,
                ),
            )
        )

    steps.append(
        (
            "rf",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=model_n_jobs,
            ),
        )
    )
    return Pipeline(steps)


def run_rf_cross_validation(
    X_train,
    y_train,
    groups_train,
    group_cv,
    configs,
    numerical_features,
    categorical_features,
    scoring,
    random_state=42,
    cv_n_jobs=1,
    model_n_jobs=-1,
    min_category_frequency=20,
):
    """Evaluate Random Forest configurations using grouped cross-validation."""
    if not configs:
        raise ValueError("configs must contain at least one RF configuration")

    required_keys = {
        "name",
        "scaler_type",
        "pca_variance",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "class_weight",
    }
    rows = []

    for config in configs:
        missing_keys = required_keys.difference(config)
        if missing_keys:
            raise ValueError(
                f"Configuration is missing required keys: {sorted(missing_keys)}"
            )

        model = build_rf_pipeline(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            scaler_type=config["scaler_type"],
            feature_range=config.get("feature_range", (0, 1)),
            pca_variance=config["pca_variance"],
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            class_weight=config["class_weight"],
            random_state=random_state,
            model_n_jobs=model_n_jobs,
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
            n_jobs=cv_n_jobs,
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
                "trees": config["n_estimators"],
                "max_depth": config["max_depth"],
                "min_leaf": config["min_samples_leaf"],
                "class_weight": str(config["class_weight"]),
                "cv_accuracy_mean": result["test_accuracy"].mean(),
                "cv_balanced_accuracy_mean": result[
                    "test_balanced_accuracy"
                ].mean(),
                "cv_precision_macro_mean": result[
                    "test_precision_macro"
                ].mean(),
                "cv_recall_macro_mean": result[
                    "test_recall_macro"
                ].mean(),
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

