📊 Telecom Churn Prediction: $38.9K Annual ROI Strategy
An end-to-end machine learning pipeline and interactive Streamlit application designed to predict customer churn and recommend targeted retention interventions. By utilizing XGBoost and SHAP explainability to target the top 20% highest-risk accounts, this project projects $38.9K in annual retention savings.

💼 Business Value
The Problem: False Negatives (missing a churner) cost the business their entire Customer Lifetime Value and incur steep replacement marketing costs.

The Solution: Rank customers by churn probability and intervene with tailored retention offers only for the highest-risk segment.

The ROI: Protects revenue while minimizing the cost of False Positives (promotional discounts wasted on loyal customers).

🛠️ Technical Pipeline
Robust Preprocessing: Engineered non-linear risk features (tenure_group) and built a ColumnTransformer to handle scaling and one-hot encoding seamlessly without data leakage.

Imbalance Handling: Handled the 26.5% minority class natively using XGBoost's scale_pos_weight (~2.76) to heavily penalize False Negatives, bypassing the computational overhead of SMOTE.

Modeling (XGBoost): Selected XGBoost for production due to its high ROC-AUC and gradient boosting architecture, which automatically captures complex, non-linear feature interactions (e.g., high charges combined with month-to-month contracts).

Explainable AI (SHAP): Cracked open the "black box" using TreeExplainer to provide real-time, mathematically precise justifications for every risk score.

💻 Streamlit Deployment
The winning model is serialized (churn_pipeline.joblib) and deployed as a live CRM-style dashboard for customer service agents.

Instant Prediction: Agents input customer profiles and immediately receive a probability score and risk flag.

Actionable Insights: Dynamically renders a personalized SHAP bar chart for each user, isolating exactly which factors are pushing them to leave (red) or stay (blue).

Targeted Retention: Empowers agents to offer data-backed, highly specific incentives (e.g., targeting a contract upgrade directly at a customer flagged for month-to-month risk).
