# Diabetes Readmission Prediction

An end-to-end tabular machine-learning project that predicts hospital readmission status for patients with diabetes. The canonical workflow is implemented in [`ML_Assignment_Diabetes_Readmission_models.ipynb`](ML_Assignment_Diabetes_Readmission_models.ipynb), covering data acquisition, exploratory analysis, leakage-aware preprocessing, model selection, evaluation, and artifact export.

The prediction target has three classes:

- `NO`: no recorded readmission;
- `>30`: readmission after more than 30 days;
- `<30`: readmission within 30 days.

## Dataset

The project uses the [Diabetes 130-US Hospitals for Years 1999-2008 dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008) from the UCI Machine Learning Repository. It contains 101,766 hospital encounters and 50 attributes describing patient demographics, diagnoses, medications, laboratory results, and hospital utilization.

After excluding encounters ending in death or discharge to hospice, 99,343 encounters remain. Repeated encounters are retained, while all records belonging to the same patient are kept within a single train, validation, or test partition to prevent patient-level leakage.

| Item | Value |
|---|---|
| Eligible encounters after death/hospice exclusion | 99,343 |
| Patient-disjoint training set | 79,512 encounters |
| Patient-disjoint test set | 19,831 encounters |
| Raw model features | 44 |
| Encoded features | 226 |
| Primary selection metric | Macro-F1 |
| Minority outcome monitored separately | `<30` recall and F1 |

## Current Results

The checked-in reports contain the following held-out test results:

| Model | Accuracy | Balanced Accuracy | Macro-F1 | `<30` Recall | `<30` F1 |
|---|---:|---:|---:|---:|---:|
| MLP | 0.5129 | **0.4693** | **0.4526** | **0.3752** | **0.2762** |
| Random Forest | 0.5170 | 0.4649 | 0.4494 | 0.3623 | 0.2642 |
| Linear SVM | **0.5594** | 0.4372 | 0.4393 | 0.1599 | 0.1939 |
| Logistic Regression | 0.4854 | 0.4477 | 0.4310 | 0.3153 | 0.2421 |

The MLP achieves the highest held-out Macro-F1. Random Forest is the strongest traditional model, while Linear SVM has the highest accuracy but substantially lower recall for the `<30` class.

## Pipeline

1. Download and extract the UCI dataset into `data/raw/`.
2. Parse identifiers and diagnosis codes as categorical values and distinguish missing data from the laboratory state `NotMeasured`.
3. Exclude encounters ending in death or discharge to hospice.
4. Create a patient-disjoint train/test split and perform detailed EDA on the training partition only.
5. Standardize hidden missing codes, group ICD-9 diagnoses, and encode age as a categorical feature.
6. Fit imputation, scaling, rare-category handling, and one-hot encoding on training data only.
7. Compare optional PCA representations and run grouped cross-validation for Linear SVM, Logistic Regression, and Random Forest.
8. Select the MLP epoch count with patient-grouped validation, refit on the complete training set, and evaluate every model once on the held-out test set.

## Repository Structure

```text
.
├── ML_Assignment_Diabetes_Readmission_models.ipynb  # Canonical notebook
├── modules/                                         # Reusable ML utilities
├── data/raw/                                        # Downloaded UCI data
├── features/                                        # Exported arrays and feature names
├── models/                                          # Trained model checkpoints
├── reports/                                         # Metrics, tables, and figures
├── requirements.txt
└── README.md
```

## Run in Google Colab

Open [Google Colab](https://colab.research.google.com/), select **File -> Upload notebook**, and upload `ML_Assignment_Diabetes_Readmission_models.ipynb` from this repository.

1. Upload the notebook
2. Select **Runtime -> Change runtime type** and set the hardware accelerator to **None (CPU)**.
3. Click **Connect**.
4. Select **Runtime -> Run all**.

## Generated Outputs

Running the notebook produces:

- `features/`: processed train/test matrices, PCA projections, labels, patient groups, and feature names;
- `models/best_linear_svm.joblib`;
- `models/logistic_regression_pipeline.joblib`;
- `models/best_random_forest.joblib`;
- `models/mlp_preprocessor.joblib` and `models/best_mlp.pt`;
- model-specific results under `reports/svm/`, `reports/lr/`, `reports/rf/`, and `reports/mlp/`;
- `reports/master_cv_configurations_comparison.csv`;
- `reports/master_model_comparison.csv`.

The generated features, models, and reports are currently tracked in this repository.

## Reproducibility

- Random seeds are fixed to `42` where supported.
- Train/test and cross-validation partitions are grouped by `patient_nbr`.
- Imputers, encoders, scalers, and PCA are fitted without using the outer test set.
- Macro-F1 is the primary model-selection metric because the classes are imbalanced.
- The `<30` class is reported separately to expose clinically important false negatives.

## Data Citation and Usage

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999-2008* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5230J](https://doi.org/10.24432/C5230J)


