import time
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
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

