# Diabetes Readmission Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nam21-12-06/machine-learning-assignment/blob/main/ML_Assignment_Diabetes_Readmission_models.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HCMUT](https://img.shields.io/badge/Institution-HCMUT--CSE-0052CC.svg)](#)
[![Course](https://img.shields.io/badge/Course-CO3117--Machine%20Learning-008080.svg)](#)
[![Semester](https://img.shields.io/badge/Semester-261%20(2026--2027)-orange.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Diabetes%20130--US%20Hospitals-blueviolet.svg)](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)

[![Linear Support Vector Machine](https://img.shields.io/static/v1?label=Model&message=Linear%20Support%20Vector%20Machine%20(Linear%20SVM)&color=2b5c8f)](#)
[![Logistic Regression](https://img.shields.io/static/v1?label=Model&message=Logistic%20Regression%20(SAGA)&color=d95f02)](#)
[![Random Forest Classifier](https://img.shields.io/static/v1?label=Model&message=Random%20Forest%20Classifier&color=2ca02c)](#)
[![Multilayer Perceptron](https://img.shields.io/static/v1?label=Model&message=Multilayer%20Perceptron%20(PyTorch%20MLP)&color=7570b3)](#)

| Parameter | Specification |
|:---|:---|
| **Institution** | Ho Chi Minh City University of Technology (HCMUT - VNU-HCM) |
| **Faculty** | Faculty of Computer Science and Engineering (CSE) |
| **Course Name** | Machine Learning |
| **Course ID** | CO3117 |
| **Semester** | Semester 261 (Academic Year 2026–2027) |
| **Instructor** | Dr. Truong Vinh Lan |
| **Dataset** | Diabetes 130-US Hospitals for Years 1999–2008 (UCI Machine Learning Repository) |
| **Evaluated Models** | Linear Support Vector Machine (Linear SVM), Logistic Regression, Random Forest, Multilayer Perceptron (PyTorch MLP) |

---

## 1. General Information

- **Course Name**: Machine Learning
- **Course ID**: CO3117
- **Semester**: Semester 261, Academic Year 2026–2027
- **Faculty**: Faculty of Computer Science and Engineering (CSE)
- **University**: Ho Chi Minh City University of Technology, VNU-HCM
- **Instructor**: Dr. Truong Vinh Lan

### Group Members

| No. | Full Name | Student ID | Email | Key Roles & Responsibilities | Contribution |
|:---:|:---|:---:|:---|:---|:---:|
| 1 | Nguyen Thai Hoai Nam | 2452794 | `nam.nguyenthaihoai06@hcmut.edu.vn` | Exploratory Data Analysis (EDA), Data Preprocessing & Feature Engineering Pipeline, Principal Component Analysis (PCA), Random Forest & PyTorch MLP Model Construction, Tuning & Training | 100% |
| 2 | Duong Quoc Huy | 2452371 | `huy.duongkhmtclc@hcmut.edu.vn` | Data Leakage Prevention & Patient-Level Group Split Design, Linear Support Vector Machine (Linear SVM) Construction & Hyperparameter Tuning, Software Modularization (`modules/`), Repository Architecture & Workflow Optimization | 100% |
| 3 | Nguyen Thanh Luan | 2452711 | `luan.nguyenbktechpro@hcmut.edu.vn` | Logistic Regression Modeling (SAGA Solver & Warm-Start Optimization), Literature Review & Theoretical Foundations, Multi-Class Metric Design & Clinical Imbalance Auditing, Diagnostic Visualization & Error Analysis, Technical Report Writing & Comprehensive Documentation | 100% |

---

## 2. Assignment Objectives

The primary goal of this assignment (Task 4.1: Machine Learning with Tabular Data) is to construct, evaluate, and benchmark a complete machine learning pipeline on clinical tabular data:
1. **Exploratory Data Analysis (EDA)**: Conduct thorough statistical analysis, examine feature distributions, assess high-cardinality categorical features, detect hidden missing values, and audit class imbalances.
2. **Leakage-Free Preprocessing**: Implement strict patient-level grouping (`GroupShuffleSplit`, `GroupKFold`) based on `patient_nbr` so that encounters from the same patient never appear in both training and test splits.
3. **Configurable Feature Pipeline**: Build reusable, modular preprocessors supporting numerical scaling (`StandardScaler`, `MinMaxScaler`), rare category handling, one-hot encoding, and dimensionality reduction via PCA (90% and 95% variance).
4. **Traditional Machine Learning Benchmarking**: Train and tune three distinct model families using 5-fold patient-grouped cross-validation:
   - **Linear Support Vector Machine (Linear SVM)**
   - **Logistic Regression (SAGA solver)**
   - **Random Forest Classifier**
5. **Deep Learning Extension (+5% Bonus)**: Implement a deep Multilayer Perceptron (MLP) in PyTorch featuring batch normalization, dropout, and learning rate scheduling (`ReduceLROnPlateau`), benchmarking its performance against traditional models.
6. **Reproducibility & Software Engineering**: Structure code into reusable Python modules under `modules/`, export feature arrays (`.npy`), model checkpoints, and comprehensive CSV reports and diagnostic plots.

---

## 3. Dataset Description

The project uses the [Diabetes 130-US Hospitals for Years 1999-2008 dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008) from the UCI Machine Learning Repository. It contains 101,766 clinical encounters across 130 hospitals with 50 attributes describing patient demographics, admission details, diagnoses (ICD-9 codes), medications, and laboratory tests.

The prediction target is hospital readmission within a 3-tier classification:
- `NO`: No recorded readmission.
- `>30`: Readmission after more than 30 days.
- `<30`: Unplanned readmission within 30 days (highest clinical priority).

| Item | Value / Description |
|---|---|
| Initial Encounters | 101,766 encounters, 50 attributes |
| Eligible Encounters (Death/Hospice Excluded) | 99,343 encounters |
| Patient-Disjoint Training Set ($N_{train}$) | 79,512 encounters (55,992 unique patients) |
| Patient-Disjoint Held-Out Test Set ($N_{test}$) | 19,831 encounters (13,998 unique patients) |
| Raw Features Retained | 44 clinical features |
| Processed / One-Hot Features | 226 dense features |
| Primary Selection Metric | **Macro-F1** (robust to severe class imbalance) |
| Clinical Objective | Maximize `<30` Recall and F1 while maintaining discrimination |

---

## 4. Key Experimental Results

All four model paradigms were evaluated on the exact same held-out test partition ($N=19,831$ encounters):

| Rank | Model | Best Configuration | Test Accuracy | Balanced Accuracy | Test Macro-F1 | `<30` Recall | `<30` F1 | Macro ROC-AUC |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **MLP (Deep Learning)** | 226-128-64-32-3 (BN, ReLU, Dropout) | 0.5129 | **0.4693** | **0.4526** | **0.3752** | **0.2762** | 0.6669 |
| **2** | **Random Forest** | RF-8 (Trees=100, Depth=18, Leaf=5, Balanced) | 0.5170 | 0.4649 | 0.4494 | 0.3623 | 0.2642 | **0.6691** |
| **3** | **Linear SVM** | SVM-2 (Standard, No PCA, C=1, Balanced) | **0.5594** | 0.4372 | 0.4393 | 0.1599 | 0.1939 | N/A |
| **4** | **Logistic Regression** | LR-1 (Standard, No PCA, SAGA, Balanced) | 0.4854 | 0.4477 | 0.4310 | 0.3153 | 0.2421 | 0.6533 |

### Findings & Insights:
1. **Overall Champion**: The **Multilayer Perceptron (MLP)** achieved the highest test Macro-F1 (**0.4526**) and Balanced Accuracy (**46.93%**), confirming that deep funnel architectures effectively capture non-linear clinical interactions.
2. **Clinical Sensitivity**: The MLP caught **37.52%** of early readmissions (`<30`), higher than Random Forest (36.23%) and Linear SVM (15.99%).
3. **Traditional ML Champion**: **Random Forest** is the leading traditional model, providing the highest probability discrimination (ROC-AUC 0.6691) and interpretable feature rankings.
4. **Effect of PCA**: PCA consistently degraded performance across all models (e.g., Random Forest dropped from 0.4494 to 0.4074 with PCA 90%), demonstrating that sparse clinical signals are lost under linear compression.

---

## 5. Repository Structure

```text
.
├── ML_Assignment_Diabetes_Readmission_models.ipynb  # Canonical runnable notebook
├── README.md                                        # Project documentation and metadata
├── requirements.txt                                 # Environment dependencies
├── data/
│   └── raw/                                         # Automated extraction folder for UCI data
├── features/                                        # Exported canonical .npy arrays
│   ├── X_train_processed.npy                        # Processed training features (79,512 x 226)
│   ├── X_test_processed.npy                         # Processed test features (19,831 x 226)
│   ├── y_train.npy, y_test.npy                      # Target class arrays
│   ├── groups_train.npy, groups_test.npy            # Patient group arrays (int64)
│   ├── X_train_pca90.npy, X_test_pca90.npy          # PCA 90% variance projections
│   ├── X_train_pca95.npy, X_test_pca95.npy          # PCA 95% variance projections
│   └── feature_names.json                           # Names of all 226 encoded features
├── models/                                          # Exported champion models
│   ├── best_linear_svm.joblib
│   ├── logistic_regression_pipeline.joblib
│   ├── best_random_forest.joblib
│   ├── best_mlp.pt                                  # PyTorch champion checkpoint
│   └── mlp_preprocessor.joblib
├── modules/                                         # Modular Python source files
│   ├── __init__.py
│   ├── preprocessing.py                             # Missing code standardization, encoding, pipeline
│   ├── evaluation.py                                # Standard classification reports & metrics
│   ├── svm_model.py                                 # Linear SVM CV & tuning utilities
│   ├── logistic_regression_model.py                 # Warm-start SAGA LR & CV functions
│   ├── random_forest_model.py                       # RF hyperparameter search & feature importance
│   └── mlp_model.py                                 # PyTorch DiabetesMLP architecture, dataset, training
└── reports/                                         # Experimental metrics, figures, and artifacts
    ├── master_model_comparison.csv                  # 4-model held-out test benchmark
    ├── master_cv_configurations_comparison.csv      # All 21 tested configurations
    ├── svm/                                         # SVM metrics and CV logs
    ├── lr/                                          # Logistic Regression metrics and CV logs
    ├── rf/                                          # Random Forest metrics, top features, and CV logs
    └── mlp/                                         # MLP training history, confusion matrix, loss curves
```

---

## 6. Execution Guidelines

### Option A: Run in Google Colab (Recommended)

1. Click the badge to open the notebook directly in Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nam21-12-06/machine-learning-assignment/blob/main/ML_Assignment_Diabetes_Readmission_models.ipynb)
2. Select **Runtime -> Change runtime type** and choose **T4 GPU** (or CPU).
3. Select **Runtime -> Run all** (`Ctrl + F9`).
4. The notebook will automatically:
   - Clone the supporting modules from GitHub.
   - Download and extract the raw dataset from the public UCI repository.
   - Execute all preprocessing, cross-validation, and testing stages without personal cloud mounting.

### Option B: Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nam21-12-06/machine-learning-assignment.git
   cd machine-learning-assignment
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook ML_Assignment_Diabetes_Readmission_models.ipynb
   ```
5. Select **Kernel -> Restart & Run All**.

---

## 7. Submission Package & Deliverables

As required by Section 5 of the course specification:
- **Colab Notebook**: `ML_Assignment_Diabetes_Readmission_models.ipynb` (runs end-to-end via `Run all`).
- **PDF Report**: `[Link or path to PDF report when completed]`.
- **Extracted Feature Files**: Saved in `features/` in `.npy` format.
- **GitHub Repository**: [https://github.com/nam21-12-06/machine-learning-assignment](https://github.com/nam21-12-06/machine-learning-assignment).
- **Submission `.zip` Structure**: Formatted with `notebooks/`, `modules/`, `reports/`, and `features/`.

---

## 8. Data Citation and Usage

This project is submitted for academic evaluation in the Machine Learning course at HCMUT. The dataset is provided by the UCI Machine Learning Repository under standard academic fair-use terms:

> Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999-2008* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J
