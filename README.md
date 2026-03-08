# bank customer-account-type-prediction
A machine learning project that predicts a bank customer's Account Type (Salary, Savings, or Current) based on demographic and financial attributes. Built as part of a data analytics internship, this project covers the full ML pipeline — from exploratory data analysis to multi-model evaluation.
# Problem Statement:
Banks serve customers across different account categories. Correctly identifying the type of account a customer is likely to hold helps banks personalize offerings, improve customer segmentation, and drive targeted marketing. This project uses supervised machine learning to predict account type from customer profile data.
 
 Dataset:


CreditScore   -     Customer's credit score (350–850),
Geography    -      Country of residence (encoded: France, Germany, Spain),
Gender       -      Male / Female (encoded),
Age        -        Customer's age,
Tenure         -    Number of years as a bank customer (0–10),
Balance      -      Account balance ,
EstimatedSalary  -  Estimated annual salary,
AccountType      -  Target variable — Salary (0), Current (1), Savings (2)

Records: 10,000 customers
Missing values: None
Class distribution: Slight imbalance across the three account types


Exploratory Data Analysis
A thorough EDA was performed to understand distributions, feature relationships, and class patterns:

Distribution plots for all numerical features (CreditScore, Age, Tenure, Balance, EstimatedSalary)
Count plots for categorical features (Gender, Geography, AccountType)
Correlation heatmap — revealed weak inter-feature correlations, confirming feature independence
Pairplot of numerical features to detect multivariate patterns
Box plot: Age vs. AccountType — slight variation in age distribution across account types
Count plot: Geography vs. AccountType — geography emerged as a key differentiator, with notable variation in account type distribution by country


Key EDA Insights:

Balance shows a bimodal distribution with a large spike at zero
Geography is strongly associated with account type — a critical predictor
Most numerical features are weakly correlated with each other
Class imbalance in the target variable (AccountType) was identified and considered during modeling

Models Implemented
Four classification algorithms were trained, evaluated, and compared:
1. Decision Tree (CART)

max_depth=3 to prevent overfitting
Visualized using export_graphviz + pydotplus
Revealed Geography and Balance as top-level split features
Accuracy: ~33% | Strong bias towards predicting Savings accounts

2. K-Nearest Neighbors (KNN)

n_neighbors=5, metric='euclidean'
Non-parametric, instance-based learner
Accuracy: ~35% | More balanced predictions than Decision Tree

3. Gaussian Naive Bayes

Assumes Gaussian distribution for continuous features
Fast and effective baseline for probabilistic classification
Accuracy: ~33% | Trade-off between precision and recall across classes

4. Logistic Regression

Implemented via both scikit-learn (LogisticRegression) and statsmodels (MNLogit)
Multinomial approach for 3-class prediction
statsmodels summary provided statistical significance (p-values, confidence intervals) of each feature
Accuracy: ~31–33% | Most interpretable; feature coefficients available for analysis.

Evaluation Metrics
Each model was assessed using:

Accuracy Score
Classification Report (Precision, Recall, F1-Score per class)
Confusion Matrix (visualized as seaborn heatmaps)

ModelAccuracyK-Nearest Neighbors~35%Decision Tree (max_depth=3)~33%Naive Bayes~33%Logistic Regression (sklearn)~33%Logistic Regression (statsmodels)~31%

Note: The relatively low accuracy across all models suggests that the selected features do not have strong linear or simple non-linear separability for this three-class problem. Ensemble methods (e.g., Random Forest, Gradient Boosting) or feature engineering are identified as logical next steps.
