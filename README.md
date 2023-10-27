# Predicting Heart Disease Using Machine Learning

## Introduction

This repository contains a Jupyter Notebook that demonstrates the use of various Python-based machine learning and data science libraries to build a machine learning model for predicting whether or not someone has heart disease based on their medical attributes.

## Approach

### Problem Definition

The project aims to predict whether a patient has heart disease or not based on clinical parameters.

### Data

The original data is sourced from the Cleveland dataset from the UCI Machine Learning Repository. There is also a version available on Kaggle.

### Evaluation

The project will be considered successful if it achieves at least 90% accuracy in predicting the presence or absence of heart disease during the proof of concept.

### Features

The dataset includes various features, such as age, sex, chest pain type, resting blood pressure, serum cholesterol levels, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, the slope of the peak exercise ST segment, the number of major vessels colored by fluoroscopy, thalassemia, and the predicted attribute.

## Installation

To set up the required dependencies, you can run the following command:
```bash
pip install -r requirements.txt
```
## Methodology

The project follows the typical data science process:

1. **Data Preprocessing**: Import necessary libraries and read the data using Pandas.
2. **Exploratory Data Analysis**: Explore the data, including visualizations of the review scores and their distribution.
3. **Modeling**: Build machine learning models, including Logistic Regression, K-Nearest Neighbors, and Random Forest.
4. **Hyperparameter Tuning**: Optimize the models using RandomizedSearchCV and GridSearchCV.
5. **Model Evaluation**: Evaluate the models using metrics like accuracy, precision, recall, F1-score, ROC curve, and AUC score.
6. **Feature Importance**: Determine which features contributed most to the model's predictions.
7. **Cross-Validation**: Calculate classification metrics using n-fold cross-validation to ensure model robustness.
8. **Experimentation**: Experiment with new data samples to predict whether a patient has heart disease.

## Usage

You can run the project in a Jupyter Notebook environment. Follow the code in the notebook to understand the sentiment analysis process using different machine learning models.

### Feature Importance

The project analyzes feature importance for the Logistic Regression model. Features are weighted based on their contribution to the model's predictions.


