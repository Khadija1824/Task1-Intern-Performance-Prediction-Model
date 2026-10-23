Intern Performance Prediction Model

An end-to-end machine learning pipeline designed to predict intern performance scores and classify performance brackets (Excel, Moderate, Struggle) based on activity metrics, task completion rates, feedback ratings, and attendance records.

Project Overview

Managing and assessing intern performance objectively is crucial for talent acquisition and organizational growth. This project builds a regression and classification pipeline using Python and robust machine learning algorithms (Random Forest Regressor and Gradient Boosting / XGBoost Regressor) to evaluate intern performance data effectively.

Data Cleaning & PreprocessingThe pipeline applies rigorous cleaning steps to ensure data integrity before model ingestion:

Missing Value Imputation: Numerical features are checked and missing values are safely imputed using column medians (np.median).

Duplicate Removal: Identical duplicate entries are dropped to prevent bias.

Boundary Clipping: Ratios and rates (task_completion_rate, attendance_rate) are clipped to logical boundaries [0.0, 1.0], and ratings (avg_feedback_rating) are clipped to [1.0, 5.0].

Machine Learning PipelineThe dataset is partitioned into training and testing sets ($80\%$ train, $20\%$ test with a fixed random_state=42). Two ensemble regression models are trained and compared:

Random Forest Regressor (n_estimators=150, max_depth=10)Gradient Boosting Regressor (n_estimators=150, learning_rate=0.08, max_depth=5)Evaluation MetricsBoth models achieve high predictive accuracy on unseen test data ($R^2 > 0.91$):

================ EVALUATION METRICS ================

Random Forest     -> R2: 0.9196 | RMSE: 2.5716 | MAE: 1.7165

Gradient Boosting -> R2: 0.9199 | RMSE: 2.5663 | MAE: 1.7453

Visualizations1. Actual vs. Predicted Performance ScoresCompares the model's predictions against ground truth values alongside an ideal fit reference line ($y = x$).Strong alignment along the diagonal demonstrates low variance and high generalization capability.

2. Feature Importance AnalysisIdentifies which metrics drive intern performance scores the most based on Random Forest feature importances, offering actionable insights for internship coordinators.

3.Requirements & SetupTo run this notebook or pipeline locally, install the necessary dependencies:pip install pandas numpy matplotlib seaborn scikit-learn

Run the Jupyter notebook or execute the script using your Python environment with the dataset intern-data.csv placed in the root directory.

