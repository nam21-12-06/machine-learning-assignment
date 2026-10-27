"""PyTorch utilities for the diabetes readmission MLP pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import Dataset


LABELS = (0, 1, 2)
CLASS_NAMES = ("NO", ">30", "<30")


class DiabetesTabularDataset(Dataset):
    """Convert preprocessed feature matrices and integer labels to tensors."""

    def __init__(self, features, labels):
        features_array = np.asarray(features, dtype=np.float32)
        labels_array = np.asarray(labels, dtype=np.int64)

        if features_array.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if labels_array.ndim != 1:
            raise ValueError("labels must be a one-dimensional array")
        if features_array.shape[0] != labels_array.shape[0]:
            raise ValueError("features and labels must have the same number of rows")
        if features_array.shape[0] == 0:
            raise ValueError("features and labels must not be empty")
        if not np.isfinite(features_array).all():
            raise ValueError("features must contain only finite values")
        if not np.isin(labels_array, LABELS).all():
            raise ValueError(f"labels must be class identifiers in {LABELS}")

        self.features = torch.from_numpy(features_array.copy())
        self.labels = torch.from_numpy(labels_array.copy())

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class DiabetesMLP(nn.Module):
    """Funnel-shaped MLP with batch normalization and dropout."""

    def __init__(
        self,
        input_dim=226,
        hidden_dims=(128, 64, 32),
        dropout_rates=(0.3, 0.2, 0.1),
        num_classes=3,
    ):
        super().__init__()
        if input_dim < 1 or num_classes < 2:
            raise ValueError("input_dim must be positive and num_classes at least 2")
        if len(hidden_dims) != len(dropout_rates) or not hidden_dims:
            raise ValueError("hidden_dims and dropout_rates must have equal nonzero length")
        if any(width < 1 for width in hidden_dims):
            raise ValueError("hidden layer widths must be positive")
        if any(not 0 <= rate < 1 for rate in dropout_rates):
            raise ValueError("dropout rates must be in the range [0, 1)")

        layers = []
        previous_dim = input_dim
        for width, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.extend(
                [
                    nn.Linear(previous_dim, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            previous_dim = width
        layers.append(nn.Linear(previous_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)


def _collect_predictions(model, data_loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            if criterion is not None:
                total_loss += criterion(logits, labels).item() * labels.size(0)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())
            total_samples += labels.size(0)

    return (
        total_loss / total_samples if criterion is not None else None,
        np.concatenate(targets),
        np.concatenate(predictions),
    )


def train_mlp_pipeline(
    model,
    train_loader,
    validation_loader,
    class_weights,
    device,
    checkpoint_path=None,
    learning_rate=1e-3,
    weight_decay=1e-4,
    epochs=100,
    early_stopping_patience=12,
    scheduler_patience=5,
):
    """Train with validation Macro-F1 scheduling and best-checkpoint restore."""
    if epochs < 1 or early_stopping_patience < 1 or scheduler_patience < 0:
        raise ValueError("epochs and early-stopping patience must be positive")

    device = torch.device(device)
    model = model.to(device)
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    if (
        weights.numel() != len(LABELS)
        or not torch.isfinite(weights).all()
        or torch.any(weights <= 0)
    ):
        raise ValueError("class_weights must contain one positive finite value per class")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=scheduler_patience,
    )

    history = []
    best_macro_f1 = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_targets = []
        train_predictions = []
        train_samples = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_total += loss.item() * batch_size
            train_samples += batch_size
            train_targets.append(labels.detach().cpu().numpy())
            train_predictions.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_macro_f1 = f1_score(
            np.concatenate(train_targets),
            np.concatenate(train_predictions),
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
        validation_loss, validation_targets, validation_predictions = (
            _collect_predictions(
                model, validation_loader, device, criterion=criterion
            )
        )
        validation_macro_f1 = f1_score(
            validation_targets,
            validation_predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
        scheduler.step(validation_macro_f1)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_total / train_samples,
                "validation_loss": validation_loss,
                "train_macro_f1": train_macro_f1,
                "validation_macro_f1": validation_macro_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if validation_macro_f1 > best_macro_f1:
            best_macro_f1 = validation_macro_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                checkpoint = Path(checkpoint_path)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "input_dim": model.network[0].in_features,
                        "num_classes": model.network[-1].out_features,
                        "best_epoch": best_epoch,
                        "best_validation_macro_f1": best_macro_f1,
                    },
                    checkpoint,
                )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss={history[-1]['train_loss']:.4f} | "
            f"val loss={validation_loss:.4f} | "
            f"val Macro-F1={validation_macro_f1:.4f}"
        )
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch, best_macro_f1


def evaluate_mlp(model, data_loader, device):
    """Return test metrics, class report, confusion matrix, and probabilities."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    targets = []
    predictions = []
    probabilities = []
    with torch.no_grad():
        for features, labels in data_loader:
            logits = model(features.to(device))
            targets.append(labels.detach().cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    y_probability = np.concatenate(probabilities)

    y_true_binary = np.eye(len(LABELS), dtype=np.int64)[y_true]
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        "Weighted F1": f1_score(
            y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
        ),
        "<30 Precision": precision_score(
            y_true, y_pred, labels=LABELS, average=None, zero_division=0
        )[2],
        "<30 Recall": recall_score(
            y_true, y_pred, labels=LABELS, average=None, zero_division=0
        )[2],
        "<30 F1": f1_score(
            y_true, y_pred, labels=LABELS, average=None, zero_division=0
        )[2],
        "Macro ROC-AUC OvR": roc_auc_score(
            y_true, y_probability, labels=LABELS, multi_class="ovr", average="macro"
        ),
        "Macro Average Precision OvR": average_precision_score(
            y_true_binary, y_probability, average="macro"
        ),
    }
    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).T
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    return metrics, report, matrix, y_pred, y_probability
