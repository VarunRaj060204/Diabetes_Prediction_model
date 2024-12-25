# Diabetes-Prediction
![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUvjW9ie9klAez9AppdpcQzmvJ8MPV9qBYGg&s "Diabetes Prediction")

## Table Of Contents
  - [Project Introduction](#project-introduction)
  - [Dataset Description](#dataset-description)
  - [EDA](#eda)
  - [Data Preprocessing](#data-preparation)
  - [Modeling Phase](#modeling-phase)
  - [Evaluation Metric](#evaluation-metric)
  - [Conclusion](#conclusion)

### Project Introduction
Diabetes is a chronic medical condition that affects the body's ability to process sugar in the blood. With the rising number of diabetic patients globally, it's essential to develop predictive models to detect diabetes early, enabling better treatment. Our project focuses on predicting whether a person has diabetes or not based on various health-related attributes. Using a dataset containing information like pregnancies, glucose levels, blood pressure, and BMI, we developed a machine learning model to predict the likelihood of diabetes in individuals. The goal is to help healthcare providers identify high-risk patients for early intervention.
### Dataset Description
The dataset used for this project is the PIMA Indian Diabetes Dataset, which contains 8 medical diagnostic features along with an outcome label (0 for non-diabetic and 1 for diabetic). The dataset has 768 records with the following features:
 - Pregnancies
 - Glucose
 - Blood Pressure
 - Skin Thickness
 - Insulin
 - BMI (Body Mass Index)
 - Diabetes Pedigree Function
 - Age
 - Outcome (Target Label)
The goal of the project is to train a machine learning model that predicts the Outcome based on the 8 input features.

###  EDA:
Below are the observations which we have made from the data visualization done as part of the Data Understanding process.
* The Pregnancies feature indicates that more pregnancies may correlate with a higher likelihood of diabetes, as women with a history of gestational diabetes may develop Type 2 diabetes later in life.
* Glucose levels are strongly associated with diabetes, as elevated glucose is a key indicator of the disease.
* BMI (Body Mass Index) is another significant predictor, with higher BMI values being correlated with higher diabetes prevalence.
* The Age feature shows that older individuals are at higher risk of diabetes, as the disease is more common in older populations.
* A histogram of the Pregnancies feature revealed that most individuals in the dataset had fewer than 5 pregnancies, with a small proportion having more than 5.

### Data Preparation
* Standardization: Used StandardScaler to standardize the features and ensure all input features are on the same scale for better model performance.
* Data Split: The dataset was split into training (80%) and testing (20%) sets using train_test_split to ensure a fair evaluation of model performance.
* Label Encoding: The target variable (Outcome) is already numeric (0 or 1), so no encoding was needed.

### Modeling Phase
- Implemented a Support Vector Machine (SVM) classifier with a linear kernel for the classification task.
- Training: The SVM classifier was trained on the standardized training data.
- Prediction: The model was evaluated using the accuracy score on both training and testing data.

### Evaluation Metric
The model's performance is evaluated using the accuracy score, which is the ratio of correctly predicted instances to the total instances. Higher accuracy indicates better model performance. Additionally, confusion matrix and classification report can be used to evaluate other metrics like precision, recall, and F1-score.
### Conclusion
The Support Vector Machine (SVM) classifier demonstrated reasonable performance in predicting diabetes based on the given medical features. The model achieved satisfactory accuracy scores on both the training and testing datasets, indicating it is capable of identifying high-risk individuals for diabetes. However, further improvements can be made by exploring hyperparameter tuning, using other models like Random Forest or XGBoost, and adding more features for better prediction accuracy.

