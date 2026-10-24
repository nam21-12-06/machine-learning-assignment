import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


def get_preprocessor(
    numerical_features,
    categorical_features,
    scaler_type="standard",
    feature_range=(0, 1),
    log_numeric=False,
    min_category_frequency=20,
):
    """Build a train-fitted preprocessor for numeric and categorical columns."""
    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(feature_range=feature_range),
        "none": "passthrough",
    }

    if scaler_type not in scalers:
        raise ValueError(
            "scaler_type must be 'standard', 'robust', 'minmax', or 'none'"
        )
    if min_category_frequency is not None and min_category_frequency < 1:
        raise ValueError("min_category_frequency must be at least 1 or None")

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if log_numeric:
        numeric_steps.append(
            (
                "log1p",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            )
        )
    numeric_steps.append(("scaler", scalers[scaler_type]))

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Missing"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_category_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), list(numerical_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
