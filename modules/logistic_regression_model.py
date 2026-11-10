import time
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from .preprocessing import get_preprocessor


def train_logistic_regression_pipeline(
    X_train,
    y_train,
    numerical_features,
    categorical_features,
    epochs=100,
    random_state=42,
    class_weight="balanced",
    min_category_frequency=20,
    print_steps=None,
):
    """Fit the project's warm-start Logistic Regression pipeline.

    The model intentionally performs one SAGA iteration per loop so that the
    optimization loss can be inspected at selected iterations.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    if print_steps is None:
        selected_steps = {1, *range(10, epochs + 1, 10)}
    else:
        selected_steps = {int(step) for step in print_steps}
        invalid_steps = {
            step for step in selected_steps if step < 1 or step > epochs
        }
        if invalid_steps:
            raise ValueError(
                "print_steps must contain values between 1 and epochs"
            )

    preprocessor = get_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        scaler_type="standard",
        log_numeric=False,
        min_category_frequency=min_category_frequency,
    )
    X_train_transformed = preprocessor.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1,
        warm_start=True,
        solver="saga",
        random_state=random_state,
        class_weight=class_weight,
    )

    history_rows = []
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for iteration in range(1, epochs + 1):
            model.fit(X_train_transformed, y_train)

            if iteration in selected_steps:
                predicted_probabilities = model.predict_proba(
                    X_train_transformed
                )
                history_rows.append(
                    {
                        "iteration": iteration,
                        "log_loss": log_loss(
                            y_train,
                            predicted_probabilities,
                        ),
                    }
                )

    elapsed_seconds = time.time() - start_time
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    history = pd.DataFrame(
        history_rows,
        columns=["iteration", "log_loss"],
    )
    return pipeline, history, elapsed_seconds



def build_logistic_regression_pipeline(
    numerical_features,
    categorical_features,
    scaler_type="standard",
    pca_variance=None,
    C=1.0,
    class_weight="balanced",
    random_state=42,
    solver="lbfgs",
    max_iter=200,
    min_category_frequency=20,
):
    """Build a leakage-safe preprocessing and Logistic Regression pipeline."""
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
                PCA(
                    n_components=pca_variance,
                    svd_solver="full",
                    random_state=random_state,
                ),
            )
        )

    steps.append(
        (
            "model",
            LogisticRegression(
                C=C,
                solver=solver,
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=random_state,
            ),
        )
    )
    return Pipeline(steps)


def run_logistic_regression_cross_validation(
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
    """Evaluate Logistic Regression configurations using grouped cross-validation."""
    if not configs:
        raise ValueError("configs must contain at least one LR configuration")

    required_keys = {"name", "scaler_type", "pca_variance", "C", "class_weight"}
    rows = []

    for config in configs:
        missing_keys = required_keys.difference(config)
        if missing_keys:
            raise ValueError(
                f"Configuration is missing required keys: {sorted(missing_keys)}"
            )

        model = build_logistic_regression_pipeline(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            scaler_type=config["scaler_type"],
            pca_variance=config["pca_variance"],
            C=config["C"],
            class_weight=config["class_weight"],
            random_state=random_state,
            min_category_frequency=min_category_frequency,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
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
