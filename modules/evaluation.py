"""Classification evaluation utilities."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(y_true, y_pred, labels, class_names):
    """Compute overall metrics and a class-level classification report."""
    if len(labels) != len(class_names):
        raise ValueError("labels and class_names must have the same length")

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

    summary = pd.DataFrame(
        {"Metric": list(metrics.keys()), "Score": list(metrics.values())}
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    return metrics, summary, report_df
