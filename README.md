# Customer Churn Prediction

## Project Overview

Customer churn is one of the most critical business problems in subscription-based industries such as telecommunications, banking, SaaS, and e-commerce. Retaining existing customers is often more cost-effective than acquiring new ones.

This project aims to build an end-to-end Machine Learning system that predicts whether a customer is likely to churn based on customer demographics, account information, and service usage patterns.

The project focuses not only on model building but also on:
- Business understanding
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Evaluation
- Flask Web Application
- End-to-End ML Pipeline
- Deployment-ready structure

---

# Business Problem

Telecom companies face significant revenue loss due to customer churn. The objective of this project is to identify customers who are likely to leave the service so that the company can take proactive retention actions.

Using machine learning, we aim to:
- Predict churn probability
- Identify major churn-driving factors
- Help businesses improve customer retention strategies

---

# Project Objectives

- Perform comprehensive Exploratory Data Analysis
- Understand customer churn behavior
- Handle data preprocessing and feature engineering
- Build and compare multiple classification models
- Optimize model performance
- Build an end-to-end ML pipeline
- Deploy the model using Flask
- Create a user-friendly web interface for prediction

---

# Dataset Features

The dataset contains customer demographic information, subscription details, billing information, and service usage features.

### Important Features:
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Online Security
- Tech Support
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- Churn (Target Variable)

---

# Machine Learning Workflow

1. Problem Understanding
2. Data Collection
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Engineering
6. Data Preprocessing
7. Model Building
8. Model Evaluation
9. Hyperparameter Tuning
10. Pipeline Creation
11. Flask Application Development
12. Deployment

---

# Models Planned

The following classification models will be explored:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost
- AdaBoost
- K-Nearest Neighbors
- Support Vector Machine

---

# Evaluation Metrics

Since churn prediction is a classification problem, the following metrics will be used:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

# Tech Stack

## Programming Language
- Python

## Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- CatBoost
- Flask
- Imbalanced-learn
- Joblib

## Tools
- VS Code
- Git & GitHub
- Jupyter Notebook

---

# Project Structure

```bash
Customer-Churn-Prediction/
│
├── notebook/                # Jupyter notebooks for EDA and experimentation
├── src/                     # Source code files
├── templates/               # HTML templates for Flask
├── static/                  # CSS and static assets
├── artifacts/               # Saved models and preprocessing objects
│
├── app.py                   # Flask application
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── .gitignore               # Ignored files for Git