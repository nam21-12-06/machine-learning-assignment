# Diabetes Readmission Prediction

An end-to-end tabular machine-learning project for predicting hospital readmission among patients with diabetes using the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008).

The prediction target has three classes:

- `NO`: no recorded readmission;
- `>30`: readmission after more than 30 days;
- `<30`: readmission within 30 days.

The canonical entry point is [`ML_Assignment_Diabetes_Readmission_models.ipynb`](ML_Assignment_Diabetes_Readmission_models.ipynb). It contains data acquisition, exploratory analysis, preprocessing, model experiments, evaluation, and artifact export.

## Key Facts

| Item | Value |
|---|---|
| Original encounters | 101,766 |
| Eligible encounters after death/hospice exclusion | 99,343 |
| Patient-disjoint training set | 79,512 encounters |
| Patient-disjoint test set | 19,831 encounters |
| Raw model features | 44 |
| Encoded features | 226 |
| Primary selection metric | Macro-F1 |
| Minority outcome monitored separately | `<30` recall and F1 |

Repeated encounters are retained. Leakage is controlled by keeping every patient entirely within one train, validation, or test partition.

## Current Results

The checked-in reports contain the following held-out test results:

| Model | Accuracy | Balanced Accuracy | Macro-F1 | `<30` Recall | `<30` F1 |
|---|---:|---:|---:|---:|---:|
| MLP | 0.5129 | **0.4693** | **0.4526** | **0.3752** | **0.2762** |
| Random Forest | 0.5170 | 0.4649 | 0.4494 | 0.3623 | 0.2642 |
| Linear SVM | **0.5594** | 0.4372 | 0.4393 | 0.1599 | 0.1939 |
| Logistic Regression | 0.4854 | 0.4477 | 0.4310 | 0.3153 | 0.2421 |

The MLP has the highest held-out Macro-F1, Random Forest is the strongest traditional model, and Linear SVM has the highest accuracy but substantially lower recall for the `<30` class.

## Pipeline

1. Download and extract the UCI dataset into `data/raw/`.
2. Parse identifiers and diagnosis codes as categorical values and distinguish missing data from the laboratory state `NotMeasured`.
3. Exclude encounters ending in death or discharge to hospice.
4. Create a patient-disjoint train/test split and perform detailed EDA on the training partition only.
5. Standardize hidden missing codes, group ICD-9 diagnoses, and keep age as a one-hot categorical feature by default.
6. Fit imputation, scaling, rare-category handling, and one-hot encoding on training data only.
7. Compare optional PCA representations and run grouped cross-validation for Linear SVM, Logistic Regression, and Random Forest.
8. Select the MLP epoch count with leakage-safe validation, refit on the complete outer training set, and evaluate all models once on the held-out patient-disjoint test set.

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

Model-specific working notebooks may also exist in the repository, but the combined notebook above is the source of truth.

## Setup and Run

### Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File -> Upload notebook**.
3. Upload `ML_Assignment_Diabetes_Readmission_models.ipynb`.
4. Select **Runtime -> Change runtime type**.
5. Set the hardware accelerator to **None (CPU)**.
6. Click **Connect**.
7. Select **Runtime -> Run all**.

The setup cell detects Colab, clones the repository into `/content/machine-learning-assignment`, and imports the supporting modules from the branch configured in `REPO_BRANCH`. That branch must contain the module versions matching the uploaded notebook. The notebook then downloads the UCI dataset and creates the feature, model, and report directories automatically.

Internet access is required for the GitHub clone and dataset download. Colab storage is temporary, so download any models, reports, or feature artifacts that must be retained before disconnecting the runtime.

### Local execution

Python 3.10 or newer is recommended. Create and activate a virtual environment.

```bash
python -m venv .env
```

Windows PowerShell:

```powershell
.\.env\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .env/bin/activate
```

Install dependencies and start Jupyter from the repository root:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Open `ML_Assignment_Diabetes_Readmission_models.ipynb` and run the cells from top to bottom. The dataset is downloaded automatically when it is not present. Local execution uses CUDA when available and otherwise falls back to CPU.

## Generated Outputs

Running the notebook produces:

- `features/`: processed train/test matrices, PCA projections, labels, patient groups, and feature names;
- `models/best_linear_svm.joblib`;
- `models/logistic_regression_pipeline.joblib`;
- `models/best_random_forest.joblib`;
- `models/mlp_preprocessor.joblib`;
- `models/best_mlp.pt`;
- `reports/svm/`, `reports/lr/`, `reports/rf/`, and `reports/mlp/`;
- `reports/master_cv_configurations_comparison.csv`;
- `reports/master_model_comparison.csv`.

The generated features, models, and reports are currently tracked in this repository.

## Reproducibility

- Random seeds are fixed to `42` where supported.
- Train/test and cross-validation partitions are grouped by `patient_nbr`.
- Imputers, encoders, scalers, and PCA are fitted without using the outer test set.
- Macro-F1 is the primary model-selection metric because the classes are imbalanced.
- The `<30` class is reported separately to expose clinically important false negatives.

## Known Limitations

- The final Logistic Regression report uses the warm-start SAGA experiment rather than refitting the highest-ranked grouped-CV configuration.
- The checked-in MLP metrics, figures, and checkpoint were generated before the leakage-safe validation and full-train refit workflow was introduced. Regenerate them by running the updated notebook.
- Dependency versions are not pinned in `requirements.txt`.
- Some saved notebook outputs contain absolute paths from the environment in which the notebook was executed.
- The current results are internal held-out estimates; there is no external validation, probability calibration, or threshold optimization.

## Data Usage and License

The dataset is provided by the UCI Machine Learning Repository. Follow its attribution and usage requirements when redistributing the data or derived artifacts.

This is a coursework repository. No separate software license has been provided; add a `LICENSE` file before reusing or redistributing the project code.
