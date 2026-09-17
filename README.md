# Diabetes Readmission Prediction

Traditional machine learning pipeline for predicting hospital readmission among diabetic patients using the UCI Diabetes 130-US Hospitals dataset.

The target variable has three classes:

- `NO`: no readmission
- `>30`: readmission after 30 days
- `<30`: readmission within 30 days

## Project Status

The notebook currently implements:

- Exploratory data analysis (EDA)
- Missing-value and duplicate analysis
- Patient-level leakage prevention
- Data cleaning and feature selection
- ICD-9 diagnosis grouping
- Age feature engineering
- Configurable imputation, scaling, and one-hot encoding
- PCA explained-variance analysis

Model training, hyperparameter experiments, and final model comparison are planned next.

## Repository Structure

```text
Assignment/
├── ML_Assignment_Diabetes_Readmission.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── ml_assignment/
	├── data/
	│   └── raw/
	├── modules/
	├── features/
	└── reports/
```

Downloaded datasets, local virtual environments, generated feature files, and notebook checkpoints are excluded by `.gitignore`.

## Dataset

The notebook downloads the dataset automatically from the public UCI repository:

<https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip>

The raw files are extracted into `ml_assignment/data/raw/` when the notebook is executed. The dataset is not committed to this repository.

## Setup

Python 3.10 or newer is recommended.

Create and activate a virtual environment:

```bash
python -m venv .env
```

On Windows PowerShell:

```powershell
.\.env\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .env/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Open `ML_Assignment_Diabetes_Readmission.ipynb` and run all cells from top to bottom.

## Pipeline Overview

### 1. Exploratory Data Analysis

The notebook examines:

- Dataset dimensions and data types
- Missing-value distributions
- Duplicate rows and repeated patient encounters
- Target-class imbalance
- Categorical feature cardinality and dominance
- Numerical distributions and outliers
- Correlations and feature-target relationships

### 2. Preprocessing

The preprocessing pipeline:

1. Removes hospice/death discharge records that cannot represent future readmission.
2. Sorts encounters numerically and keeps the first encounter per patient.
3. Removes identifier columns and the heavily missing `weight` column.
4. Converts invalid mapped IDs to missing values.
5. Groups ICD-9 diagnosis codes into broad clinical categories.
6. Converts age ranges to numerical midpoints.
7. Splits the data using stratification.
8. Selects near-constant features using training data only.
9. Imputes missing values and applies configurable scaling and one-hot encoding.

The preprocessor supports both `StandardScaler` and `MinMaxScaler`, including a configurable `feature_range` for Min-Max scaling.

### 3. PCA

PCA is fitted only on the processed training data. The notebook reports the number of components required to preserve 90% and 95% of the variance.

## Reproducibility Notes

- The random seed is fixed at `42`.
- No Google Drive or personal cloud storage is required.
- Dataset downloading and extraction are handled by the notebook.
- Imputers, scalers, and encoders are fitted on training data and only applied to test data afterward.
- Patient-level duplicate encounters are removed before the train/test split to reduce patient leakage.

## Assignment Deliverables

The final submission should include:

- This notebook with successful `Run All` execution
- A PDF report containing EDA, pipeline design, experiments, and analysis
- Model training and evaluation results
- Comparison of tested configurations
- Any generated feature files required by the assignment

## License and Data Usage

This repository contains coursework code. Please follow the terms and attribution requirements of the UCI dataset when redistributing the data or derived materials.
