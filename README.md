# Heart-Disease-Detection & Classification 

This project involves analyzing a heart disease dataset and developing machine learning models to predict the presence of heart disease in patients. The project follows a structured workflow from data exploration and feature engineering to model implementation and evaluation. The goal is to identify significant predictors of heart disease and provide a reliable prediction model.

## Dataset Information

The dataset used in this project includes various features related to a patient's medical profile, such as:

Age
Sex
Chest Pain Type
Resting Blood Pressure
Cholesterol
Fasting Blood Sugar
Resting ECG results
Max Heart Rate
Exercise-induced Angina
ST Depression (Oldpeak)
ST Slope
The target variable is a binary indicator of heart disease presence (1 for disease, 0 for no disease).


## Exploratory Data Analysis (EDA)

During EDA, we explored various relationships between the features and target variable. Key findings include:

Chest Pain Type, Exercise-Induced Angina, ST Slope, and Max Heart Rate are significant predictors of heart disease.
Age Distribution: Most patients are between 40-60 years, indicating higher prevalence in middle-aged individuals.
Correlation Analysis: Positive correlations with heart disease were found for ST depression (oldpeak) and specific chest pain types.
The insights from EDA guided feature selection and model development.

# Machine Learning Models

Three machine learning models were implemented and evaluated:

### Logistic Regression:
Simple, interpretable model.
Accuracy: 0.80, ROC-AUC: 0.8564

## Support Vector Machine (SVM):
Best-performing model in terms of recall and F1-score.
Accuracy: 0.82, ROC-AUC: 0.8639

## Naive Bayes:
Quick baseline model with reasonable performance.
Accuracy: 0.78, ROC-AUC: 0.8444
Each model was evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

## Feature Engineering and Optimization

### Feature Engineering:
Categorical features were one-hot encoded.
Numerical features were standardized for models like SVM and Logistic Regression.
Important features identified from EDA were used in the models, including chest pain type, max heart rate, and ST slope.
Model Optimization:
Logistic Regression: Optimized with regularization (L2).
SVM: Optimized with hyperparameters C and gamma.
Naive Bayes: Used var_smoothing to handle low variance issues and improve stability.
Permutation Importance: Used to identify top features impacting each model's predictions.
Results and Comparison

The models were compared based on the following metrics:

Metric	Logistic Regression	SVM	Naive Bayes

Accuracy	0.80,	0.82,	0.78

Precision	0.80,	0.82,	0.78

Recall	0.79,	0.82,	0.78

F1-Score	0.80,	0.82,	0.78

ROC-AUC	0.8564,	0.8639,	0.8444

## Summary of Results
Best Model: The SVM model demonstrated the best performance in terms of accuracy, recall, and ROC-AUC.
Interpretable Model: Logistic Regression was highly interpretable and performed well, making it a suitable alternative.
Baseline Model: Naive Bayes was efficient and useful for initial analysis but underperformed compared to the other models.

