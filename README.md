# Tourism Package Prediction — MLOps Project

## 📌 Project Overview

This project develops a machine learning solution to predict whether a customer is likely to purchase a tourism package.

The project implements an end-to-end machine learning and MLOps workflow covering:

- Data registration
- Data preparation and preprocessing
- Exploratory analysis
- Machine learning model experimentation
- Model evaluation
- MLflow experiment tracking
- Model pipeline creation
- Hugging Face model registration
- Streamlit deployment
- Dockerization
- GitHub Actions CI/CD automation

The project is designed as an end-to-end deployment pipeline rather than only a standalone machine learning model.

---

## 🎯 Project Objective

The objective is to build a classification model that predicts the `ProdTaken` variable, which indicates whether a customer purchased the tourism package.

### Target Variable

`ProdTaken`

- `0` — Customer did not purchase the package
- `1` — Customer purchased the package

The dataset contains customer demographic, travel, contact, and behavioural characteristics that are used to predict package purchase.

---

## 📊 Dataset

The project uses a tourism package prediction dataset containing:

- **4,128 observations**
- **21 columns** in the original dataset
- Customer demographic and behavioural features
- Target variable: `ProdTaken`

The data was divided into:

- Training set: **3,302 observations**
- Test set: **826 observations**

### Data Sources

The dataset and train/test splits are registered on Hugging Face.

**Hugging Face Dataset:**

https://huggingface.co/datasets/rashmikportfolio/tourism-package-prediction

---

## 🔍 Data Preparation

The data preparation workflow included:

- Checking dataset dimensions
- Checking missing values
- Checking duplicate records
- Checking unique customer IDs
- Examining the target variable distribution
- Standardizing inconsistent categorical values
- Separating features and target
- Identifying numerical and categorical variables
- Encoding categorical variables using `OneHotEncoder`
- Preparing training and testing datasets

The original dataset contained an inconsistent gender value:

`Fe Male`

This was standardized to:

`Female`

### Feature Processing

The final preprocessing pipeline consisted of:

- Numerical feature handling
- Categorical feature encoding
- One-hot encoding of categorical variables

After preprocessing:

- Training data: **3,302 × 34**
- Test data: **826 × 34**

---

## 🤖 Machine Learning Models

Several classification algorithms were experimented with and evaluated.

The models included:

1. Decision Tree
2. Bagging Classifier
3. Random Forest
4. AdaBoost
5. Gradient Boosting
6. XGBoost

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 89.71% | 73.42% | 72.96% | 73.19% | 83.33% |
| Bagging | 92.62% | 90.83% | 68.55% | 78.14% | 97.54% |
| Random Forest | 91.53% | 93.20% | 60.38% | 73.28% | 96.71% |
| AdaBoost | 84.62% | 74.24% | 30.82% | 43.56% | 82.63% |
| Gradient Boosting | **92.74%** | **91.60%** | **68.55%** | **78.42%** | **95.36%** |
| XGBoost | 91.04% | 88.99% | 61.01% | 72.39% | 94.68% |

---

## 🏆 Final Model

The final model used in the deployment pipeline is the:

### Gradient Boosting Classifier

The trained model was combined with the preprocessing steps into a single machine learning pipeline.

### Final Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 92.74% |
| Precision | 91.60% |
| Recall | 68.55% |
| F1 Score | 78.42% |
| ROC-AUC | 95.36% |

The complete preprocessing and model pipeline is saved as:

```text
tourism_package_pipeline.joblib
