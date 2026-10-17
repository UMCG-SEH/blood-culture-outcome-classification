# Machine learning pipelines for predicting positive blood cultures. 

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
DOI:

## Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Key Features](#key-features)
4. [Installation & Requirements](#installation--requirements)
5. [Usage / Quick Start](#usage--quick-start)
6. [Interactive Pipeline Options](#interactive-pipeline-options)
7. [Limitations & Notes](#limitations--notes)
8. [Citation](#citation)

---

## Overview

This repository contains the official machine learning pipelines developed for the prediction of bloodstream infections, as described in the accompanying scientific article: *Machine learning pipeline for predicting positive blood cultures using Sysmex hematology data in the emergency department.*. 

The codebase is designed with a  emphasis on **research reproducibility**, extensive metric evaluation, and advanced handling of imbalanced datasets. It consists of **5 standalone Jupyter Notebook pipelines**:

- **4 Tree-based Models:** Random Forest, XGBoost, LightGBM, and CatBoost.
- **1 Deep Learning Model:** A ResNet-based neural network architecture. (Note: This pipeline shares the same robust evaluation structure but excludes feature selection/importance scoring and utilizes a streamlined hyperparameter tuning approach).

All pipelines evaluate performance using rigorous cross-validation, hyperparameter tuning via Bayesian optimization, threshold analysis, and probability calibration.

---

## Repository Structure

```text
├── models/
│   ├── randomforest_pipeline.ipynb
│   ├── xgboost_pipeline.ipynb
│   ├── lightgbm_pipeline.ipynb
│   ├── catboost_pipeline.ipynb
│   └── resnet_pipeline.ipynb
├── requirements.txt
└── README.md

```

*(Note: Please ensure your data is placed in the root or a designated `data/` folder as expected by the notebooks).*

---

## Key Features

### 1. Feature Selection (Tree-based Models Only)

To identify the most predictive clinical variables, the tree-based pipelines offer 5 interchangeable feature selection methodologies:

* **RFE (Recursive Feature Elimination):** Iteratively removes features that contribute least to predictions (minimum retention of 2 features).
* **Boruta:** Generates randomized "shadow" features and iteratively removes original features that perform statistically worse than their shadow counterparts.
* **Lasso (L1 Regularization):** Shrinks less important feature coefficients to exactly zero, effectively removing them (ideal for linear feature relationships).
* **Tree Importance:** Sequentially removes the top 10 least important features based on inherent node-impurity metrics.
* **NONE:** Retains the complete feature set.

> *Note: The ResNet model bypasses explicit feature selection, as neural networks inherently perform representation learning across large feature spaces.*

### 2. Reproducibility

All pipelines enforce a **fixed random state (42)** across data splitting, model initialization, and resampling algorithms to support reproducible results across multiple execution runs.

### 3. Imbalanced Data Handling

Bloodstream infection datasets are typically highly imbalanced. The pipelines natively support multiple intervention strategies:

* **Algorithmic Weighting:** Adjusts the cost function to heavily penalize minority class misclassifications (Highly recommended for tree-based models).
* **SMOTE:** Synthetic Minority Oversampling Technique.
* **Borderline-SMOTE:** Synthesizes new minority instances near the decision boundary.
* **ADASYN:** Adaptive Synthetic Sampling, dynamically shifting focus to harder-to-learn minority samples.
* **SMOTENC:** SMOTE variant designed for datasets with categorical features.

> *Note on Imbalance Strategies: The optimal strategy for handling class imbalance is highly dataset-dependent and should be empirically validated. However, as a starting point, explicit oversampling techniques (e.g., SMOTE) are often beneficial for the ResNet model, while native algorithmic class weights frequently provide a strong baseline for tree-based models without distorting the underlying data distribution.*

### 4. Hyperparameter Tuning

All pipelines utilize **Optuna** for state-of-the-art Bayesian optimization, allowing the models to efficiently search the hyperparameter space and avoid local minima.

### 5. Training Metrics & Evaluation

Models can be optimized and evaluated across a comprehensive suite of metrics:

* **Discrimination:** ROC AUC, Average Precision (AP), Accuracy, Balanced Accuracy.
* **Diagnostic:** Sensitivity (Recall), Specificity, Precision, F1-Score, F2-Score (prioritizing recall).
* **Statistical:** Matthews Correlation Coefficient (MCC).
* **Probabilistic:** Log Loss (Cross-Entropy).

The final test set evaluation systematically generates **ROC curves**, **Precision-Recall curves**, **SHAP summary plots** (tree-based models), confusion matrices, and threshold optimization analysis (maximizing Youden's J statistic).

### 6. Model Calibration

To ensure the outputted probabilities are clinically interpretable (e.g., a prediction of 0.8 accurately reflects an 80% true likelihood of infection), all pipelines implement probability calibration using **Platt Scaling (Sigmoid)** by default.

---

## Installation & ments

This pipeline was developed and validated using **Python 3.11**.

1. Clone this repository:
```bash
git clone [https://github.com/UMCG-SEH/blood-culture-outcome-classification.git](https://github.com/UMCG-SEH/blood-culture-outcome-classification.git)
cd blood-culture-outcome-classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. Install the required dependencies:
```bash
pip install -r requirements.txt
pip install torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu

```


**Core Dependencies:**
`numpy` (2.3.5), `pandas` (2.3.3), `scikit-learn` (1.7.2), `imbalanced-learn` (0.14.0), `xgboost` (3.0.2), `lightgbm` (4.6.0), `catboost` (1.2.8), `boruta` (0.4.3), `optuna` (4.6.0), `torch` (2.9.1 - CPU), `matplotlib` (3.10.8), `shap` (0.50.0), `plotly` (6.5.0).

---

## Usage / Quick Start

1. Open any of the provided Jupyter Notebooks (e.g., `xgboost_pipeline.ipynb`).
2. Scroll to the data loading section near the bottom of the notebook:
```python
import pandas as pd

# Update this path to your local dataset
df = pd.read_csv('training.csv')

# Define your target column
target_col = 'status'
feature_columns = [c for c in df.columns if c != target_col]

```


3. **Data Prerequisites:**
* All features must be numerical (categorical data must be one-hot encoded or explicitly handled prior to running).
* The target column must be binary (`0` for negative, `1` for positive).
* Missing values (NaNs) must be imputed beforehand.


4. Run all cells (`Cell -> Run All`). The notebook will prompt you interactively to configure the pipeline (see section below).

---

## Interactive Pipeline Options

Upon execution, the notebook will present several interactive prompts, allowing you to dynamically configure the experiment without altering the code:

1. **Device:**
* `CPU` (Default) or `GPU` (Accelerates training, particularly for ResNet, XGBoost, and CatBoost).


2. **Train/Test/Validation Split:**
* `none`: Standard 80/20 train/test split.
* `validation` (Default): 70/10/20 train/validation/test split for stricter early stopping and evaluation.


3. **Feature Weights (Tree-based models):**
* Option to apply algorithmic weights directly to the loss function to counter imbalance.


4. **Feature Selection:**
* `NONE` (Default), `RFE`, `BORUTA`, `LASSO`, or `TREE_IMPORTANCE`. Multiple inputs are supported.


5. **Plots / Results:**
* Choose from `roc`, `pr`, `confusion`, `importance`, `shap`, or `all`.


6. **Scoring Metric:**
* `roc_auc` (Default), `average_precision`, `f2`, `balanced_accuracy`, `mcc`, `f1`, `accuracy`, `logloss`, or `recall`.


7. **SMOTE / Imbalance Handling:**
* `NONE` (Default), `SMOTE`, `BorderlineSMOTE`, `ADASYN`, or `SMOTENC`.


---

## Limitations & Notes

* **Binary Classification Only:** This pipeline is strictly engineered for binary classification tasks (e.g., predicting the presence vs. absence of an infection). It does not natively support multi-class target variables.
* **Pre-processing Requirement:** The pipeline expects structured, pre-processed numerical data. Feature scaling (Standardization) is handled internally by the pipeline.

---

## Citation

If you use this code or pipeline in your research, please cite our corresponding paper:

```bibtex
@article{your_surname2024bloodstream,
  title={Your Article Title About Bloodstream Infections},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024},
  volume={xx},
  pages={xxx-xxx},
  doi={10.xxxx/xxxxx}
}

```
