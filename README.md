# Loan Approval Prediction System

Built an end-to-end supervised ML pipeline using KNN, Logistic Regression and Naive BAyes to predict loan approval.
Implemented Binary classification along with EDA,feature engineering & model evaluation(Precision, Recall,F1)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)

---

## Table of Contents

- [Overview](#-overview)
- [Dataset Description](#-dataset-description)
- [Features](#-features)
- [Machine Learning Models](#-machine-learning-models)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights)
- [Technologies Used](#️-technologies-used)
- [Future Improvements](#-future-improvements)

---

## Overview

This project implements and compares multiple machine learning algorithms to predict loan approval decisions based on applicant financial and demographic information. The system analyzes various features including income, credit score, employment status, and debt ratios to make accurate predictions.

### Business Problem

Financial institutions need an automated, data-driven approach to:
- Evaluate loan applications efficiently
- Reduce human bias in decision-making
- Minimize default risk
- Improve approval process speed
- Maintain fair lending practices

### Solution

This machine learning system:
-  Analyzes 20+ applicant features
-  Compares 3+ classification algorithms
-  Offers interpretable predictions

---

## Dataset Description

### Dataset Size
- **Total Applications:** 1000 records
- **Features:** 20 columns
- **Target Variable:** Loan_Approved (Binary: 0/1))|

---

## Features

### Data Preprocessing
- **Missing Value Handling:** Imputation and removal strategies
- **Encoding Categorical Variables:** Label encoding for ML models
- **Feature Scaling:** StandardScaler for normalization
- **Train-Test Split:** 80-20 split for validation
- **Data Cleaning:** Outlier detection and handling

### Exploratory Data Analysis (EDA)
- Distribution analysis of all features
- Correlation analysis between variables
- Target variable class distribution
- Feature importance analysis
- Relationship between approval and key factors

---

##  Machine Learning Models

### Classification Models Implemented

| # | Model | Type | Description |
|---|-------|------|-------------|
| 1 | **Logistic Regression** | Linear | Baseline model for binary classification |
| 2 | **K-Nearest Neighbors (KNN)** | Instance-based | Non-parametric classification |
| 3 | **Naive Bayes** | Probabilistic | Efficient for high-dimensional data|


---

## Project Structure

```
loan-approval-prediction/
│
├── data/
│   ├── raw/
│   │   └── loan_applications.csv
│   └── processed/
│       └── cleaned_data.csv
│
├── notebooks/
│   └── ML.ipynb                    # Main analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data cleaning functions
│   ├── feature_engineering.py      # Feature creation
│   ├── model_training.py           # Model training scripts
│   └── evaluation.py               # Evaluation metrics 
│
├── models/
│   ├── best_model.pkl              # Saved best model
│   ├── scaler.pkl                  # Fitted scaler
│   └── encoder.pkl                 # Label encoder
│
├── images/
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Model Performance

### Evaluation Metrics

All models are evaluated using:
- **Accuracy:** Overall correctness
- **Precision:** Positive prediction accuracy
- **Recall (Sensitivity):** True positive rate
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Detailed classification results

### Results Summary

> **Note:** Fill in after running analysis

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 88% | 78.46% | 83.60% | 80.95% |
| KNN | 78.5% | 67.30% | 57.37% | 61.94% |
| Naive Bayes | 86% | 81.13% | 70.49% | 75.43% |


**Best Model:** Naive Bayes with Precision of 81.13%

### Confusion Matrix (Best Model)

```
                Predicted
                No    Yes
Actual  No     TN     FP
        Yes    FN     TP
```

### Feature Importance

**Top 10 Most Important Features:**
1. Credit_Score
2. Applicant_Income
3. Debt-to-Income Ratio
4. Loan_Amount
5. Loan_Term 
6. Employer_Category
7. Loan_Purpose
8. Gender
9. Education_Level
10. Employment_Status

---

## Key Insights

### Factors Favoring Loan Approval

**High Credit Score** (>650)  
**Stable Employment** 
**Low Debt-to-Income Ratio** (<40%) 
**Clean Payment History**  
**Education_Level** -Graduate   

### Factors Leading to Rejection

**Low Credit Score** (<650)  
**High Debt-to-Income Ratio** (>40%)  
**Previous Loan Defaults**  
**Unstable Employment History**  
**Poor Payment History**  
**Education_Level**- Not Graduate  


---

## Technologies Used

### Core
- **Python 3.12** - Programming language
- **Jupyter Notebook** - Interactive development

### Data Science Libraries
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Basic visualization
- **Seaborn** - Statistical visualization

### Machine Learning
- **Scikit-learn** - ML algorithms and tools
  - Preprocessing (StandardScaler, LabelEncoder)
  - Model Selection (train_test_split, GridSearchCV)
  - Metrics (accuracy, precision, recall, F1, confusion matrix)
  - Models (LR, KNN, NB)

### Development Tools
- **Git** - Version control
- **GitHub** - Repository hosting

---

## Future Improvements

### Model Enhancements
- [ ] Add more ML algorithms and Models
- [ ] Implement deep learning models (Neural Networks)
- [ ] Add ensemble methods (Stacking, Voting)
- [ ] Perform advanced feature engineering
- [ ] Use SHAP values for model interpretability
- [ ] Implement AutoML for automated model selection

### Data Improvements
- [ ] Collect more data to improve model robustness
- [ ] Include time-series data for applicant tracking
- [ ] Add macroeconomic indicators
- [ ] Incorporate alternative data sources (social media, utility bills)

### Deployment
- [ ] Create Flask/FastAPI REST API
- [ ] Build web interface for predictions
- [ ] Deploy on cloud platform (AWS, Azure, GCP)
- [ ] Implement real-time prediction pipeline
- [ ] Add model monitoring and retraining automation

### Business Features
- [ ] Add explainability for loan officers
- [ ] Create risk assessment reports
- [ ] Implement A/B testing framework
- [ ] Add fairness and bias detection
- [ ] Develop dashboard for analytics

---
