# ML Pipelines for Predicting Bloodstream Infections

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
    - [Feature Selection (Tree-based Models Only)](#1-feature-selection-tree-based-models-only)
    - [Reproducibility](#2-reproducibility)
    - [Imbalanced Data Handling](#3-imbalanced-data-handling)
    - [Hyperparameter Tuning](#4-hyperparameter-tuning)
    - [Training Metrics](#5-training-metrics)
    - [Model Calibration](#6-model-calibration)
    - [Testing & Evaluation](#7-testing--evaluation)
3. [Usage / Quick Start](#usage--quick-start)
4. [Package Requirements](#package-requirements)
5. [Pipeline Options](#pipeline-options)
    - [Device](#1-device)
    - [Train/Test/Validation Split](#2-traintestvalidation-split)
    - [Feature Weights (Tree-based Models)](#3-feature-weights-some-tree-based-models)
    - [Feature Selection](#4-feature-selection)
    - [Plots / Results](#5-plots--results)
    - [Scoring Metric](#6-scoring-metric)
    - [SMOTE / Imbalance Handling](#7-smote--imbalance-handling)
    - [Recommendations & Notes](#8-recommendations--notes)

---

## Overview

**5 machine learning pipelines** for predicting bloodstream infections are developed. These pipelines are extensive and designed for research purposes. The pipelines are **5 documents**:

- **4 tree-based models:** Random Forest, XGBoost, LightGBM, and CatBoost.
 
- **1 ResNet neural network model:** It has a similar pipeline structure but does not use feature selection/feature importance scores and has only 1 hyperparameter tuning option.

All pipelines provide options for **train/test splits**:  
- **Train/Test:** 80/20 split  
- **Train/Test/Validation:** Includes a 12.5% validation set, which can improve model performance.  

Each pipeline includes:  
- Saving the final trained model  
- Extensive metric evaluation on the test set using bootstrapping and no bootstrapping
- Thresholding analysis on the test set  
- 5-fold cross-validation metrics during training

#### Key Features of All Models
- Hyperparameter options are chosen based on research and if they match the current pipeline.  
- All models work with **CPU and GPU**.  
- Random state is fixed to 42.  

## Key Features

### 1. Feature Selection (Tree-based Models Only)

The pipelines provide **5 options for feature selection**:

1. **RFE (Recursive Feature Elimination)**  
   - Removes features that contribute least to predictions  
   - Always deletes a minimum of 2 features  

2. **Boruta**  
   - Generates shadow features by shuffling values  
   - Compares feature performance against shadows and deletes underperforming features  

3. **Lasso**  
   - Best for linear data  
   - Shrinks feature importance toward 0; features reaching 0 are removed  

4. **Tree Importance**  
   - Sequentially removes the 10 least important features  

5. **NONE**  
   - Keeps all features  

> The ResNet model **does not use feature selection** because neural networks handle large feature sets well.

---
### 2. Reproducibility

All models use a **fixed random state of 42**.  
- This makes sure that results are **consistent across multiple runs**.  
- Training the same model multiple times with the same data and configuration will have identical performance metrics.  

### 3. Imbalanced Data Handling

All pipelines address class imbalance. Options include:

- **Weights:** Increase importance of the minority class (recommended for tree-based models)  
- **SMOTE (Synthetic Minority Oversampling Technique):** Creates synthetic minority samples  
- **Borderline SMOTE:** Adds new samples along feature borders  
- **ADASYN (Adaptive Synthetic Sampling):** Focuses on hard-to-predict minority samples  
- **None:** No balancing applied  

> For ResNet, using oversampling techniques is important to improve minority class predictions.
> For tree based models use the _weight files and no smote.

---

### 4. Hyperparameter Tuning

**Optuna:** Bayesian optimization package used by all pipelines including ResNet; understands patterns in parameter selection.  

### 5. Training Metrics

During training, pipelines allow optimization using various metrics:

- **Recall** – Measures how many true positives the model detects.
- **Precision** – Measures how many predicted positives are actually positive.  
- **F1 Score** – Harmonic mean of recall and precision.  
- **F2 Score** – Like F1 but gives more weight to recall. Use when detecting positives is more important than false positives.  
- **Accuracy** – Percentage of all correct predictions. Useful for balanced datasets; can be misleading in imbalanced datasets.  
- **Balanced Accuracy** – Average recall per class; good for imbalanced datasets.  
- **ROC AUC** – Measures model’s ability to distinguish positives from negatives across all thresholds.  
- **Average Precision (AP)** – Average precision over all recall levels. Good for imbalanced datasets.  
- **MCC (Matthews Correlation Coefficient)** – Measures the quality of binary classifications including all true/false positives and negatives. Works well with imbalanced datasets.  
- **Log Loss (Cross-Entropy Loss)** – Measures prediction confidence. Important for **probabilistic outputs**.

---

### 6. Model Calibration

All pipelines include **probability calibration**, which improves how well the predicted probabilities match the true likelihood of a positive outcome.  

- Why it matters:  
  Some models are overconfident or underconfident in their predictions. Calibration adjusts the probabilities so that, for example, a prediction of 0.8 really corresponds to an ~80% chance of being positive.  

- **Calibration Methods:**  
  1. **Sigmoid (Platt Scaling)**  
     - Fits a logistic function to map predicted probabilities to true probabilities.  
     - Works well for small to medium datasets.  
     - Less likely to overfit.  
  2. **Isotonic Regression**  
     - Fits a flexible, non-linear mapping to predicted probabilities.  
     - Works best for larger datasets.  
     - Can capture complex probability patterns but may overfit on small datasets.  

> In these pipelines, **sigmoid calibration** is used by default.

---

### 7. Testing & Evaluation

The test set evaluation includes:

- ROC AUC curve  
- Precision-Recall curve  
- Feature importance (tree-based models)  
- SHAP plots (tree-based models; not available for ResNet)  
- Predictions for the test set (allowing custom metric calculations)  
- Confusion matrices  
- Threshold optimization for maximizing positives/negatives  

> Calibration makes that these probabilities are interpretable for further analyses.


## Usage / Quick Start

To run the pipelines, follow these steps:

1. **Open the Notebook**  
   Open the Jupyter notebook provided with the pipeline.

2. **Locate the Data Loading Block**  
   Go to the bottom code blocks in the notebook. You will find something like this:

   ```python
   import pandas as pd

   df = pd.read_csv('final_training.csv')
   target_col = 'status'
   feature_columns = [c for c in df.columns if c != target_col]

### Adjust to Your Data

- Replace `'final_training.csv'` with the location of your dataset.  
- Update `target_col` to match the column name of your target variable.  
- Ensure `feature_columns` contains all feature columns you want to use.
- Make sure that the dataset contains only numerical features (categorical features must be preprocessed). 
- The target column must be binary (0/1 or equivalent).
- Missing values should be changed before running the pipeline.


### Run All Cells

Execute the entire notebook. It will load the data, preprocess it, and run the pipelines with all configurations.
Just fill in the options you would like to use in the pop ups.

---

### Package Requirements

This pipeline was developed and tested using **Python 3.11** with the following packages:

**Data processing & ML:**
- `numpy` v2.3.5
- `pandas` v2.3.3
- `scikit-learn` v1.7.2
- `imbalanced-learn` v0.14.0
- `XGBoost` v3.0.2
- `LightGBM` v4.6.0
- `CatBoost` v1.2.8
- `Boruta` v0.4.3
- `Optuna` v4.6.0
- `torch` v2.9.1(cpu)

**Data visualization:**
- `matplotlib` v3.10.8
- `shap` v0.50.0
- `plotly` v6.5.0

Note: This pipeline was tested with these specific versions. Other combinations may not work correctly.

---

## Pipeline Options

The pipelines has some interactive options for quick settings. When you run the notebook, you will see prompts allowing you to change things in the pipeline to your preferences.

### 1. Device
Choose whether to run the model on CPU or GPU:

**Prompt:** Which device do you want to use? (CPU / GPU) [default: CPU]

- Default: `CPU`  
- GPU training can speed up ResNet and other models if available.

---

### 2. Train/Test/Validation Split
Choose whether to include a validation set:

**Prompt:** Do you want to use a validation set? (none / validation) [default: validation]

- `none`: Train/test split only (80/20)  
- `validation`: Train/test/validation split with ~12.5% for validation

---

### 4. Feature Selection
Select which feature selection methods to use (tree-based models only):

**Prompt:** Which feature selection method? (NONE) [default: NONE]

- Multiple methods can be selected depending on the model used.  
- Available options: `NONE`, `RFE`, `BORUTA`, `LASSO`, `TREE_IMPORTANCE`  `  
- ResNet does **not use feature selection**.

---

### 5. Plots / Results
Choose which evaluation plots to generate:

**Prompt:** Which results/plots do you want? (roc, pr, confusion, all) [default: roc pr confusion]

- Options: `roc` (ROC curve), `pr` (Precision-Recall curve), `confusion` (Confusion matrix)  
- `'all'` will include all plots

---

### 6. Scoring Metric
Choose the metric to optimize during training:

**Prompt:** Which scoring metric do you want to use? (roc_auc / average_precision / f2 / balanced_accuracy / mcc / f1 / accuracy / logloss / recall) [default: roc_auc]

- Default: `roc_auc`  
- Available metrics:  
  - `roc_auc`
  - `average_precision`
  - `f2`
  - `balanced_accuracy`  
  - `mcc`
  - `f1'
  - `accuracy`
  - `logloss`
  - `recall`

---

### 7. SMOTE / Imbalance Handling
Choose which oversampling method(s) to use:

**Prompt:** Which SMOTE version do you want to use? (NONE SMOTE BorderlineSMOTE ADASYN SMOTENC) [default: NONE]

- Default: `NONE` (no oversampling)  
- Options:  
  - `SMOTE` synthetic minority oversampling technique
  - `BorderlineSMOTE` generates samples near class boundaries  
  - `ADASYN` adaptive oversampling for difficult samples  
  - `SMOTENC` SMOTE for categorical features  

---

> All options are prompted in the notebook. Default values are there, but you can customize them to fit your dataset.

### 8. Notes

**The pipeline only has been test on binary numerical datasets and won't work with multiple classification problems**

