# keerthishap_fixed.py

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ------------------------------
# 1. Generate Synthetic Dataset
# ------------------------------
X, y = make_classification(
    n_samples=5000,
    n_features=15,
    n_informative=10,
    n_redundant=2,
    n_classes=2,
    weights=[0.7,0.3],
    class_sep=1.5,
    random_state=42
)

df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(15)])
df['Churn'] = y

print("Dataset Shape:", df.shape)
print(df.head())

print("\nChurn distribution:\n", df['Churn'].value_counts())

# ------------------------------
# 2. Train/Test Split
# ------------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 3. Handle Class Imbalance
# ------------------------------
print("\nBefore SMOTE:\n", y_train.value_counts())
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("\nAfter SMOTE:\n", pd.Series(y_train_res).value_counts())

# ------------------------------
# 4. Train XGBoost Model
# ------------------------------
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_res, y_train_res)

# ------------------------------
# 5. Evaluate Model
# ------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\n--- Model Performance ---")
print("AUC Score:", roc_auc_score(y_test, y_proba))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# 6. SHAP Analysis
# ------------------------------
explainer = shap.Explainer(model, X_train_res)
shap_values = explainer(X_test)

# ------------------------------
# 6a. Global Feature Importance
# ------------------------------
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary_plot.png")
plt.close()
print("\nSHAP summary plot saved as 'shap_summary_plot.png'.")

# ------------------------------
# 6b. Local explanation: High-Risk Churner
# ------------------------------
high_risk_idx = np.argmax(y_proba)
high_risk = X_test.iloc[high_risk_idx]

plt.figure()
shap.force_plot(
    explainer.expected_value,               # scalar for binary classification
    shap_values[high_risk_idx].values,
    high_risk,
    matplotlib=True,
    show=False
)
plt.savefig("shap_force_high_risk.png")
plt.close()
print("High-risk churner SHAP force plot saved as 'shap_force_high_risk.png'.")

# ------------------------------
# 6c. Local explanation: High-Value Customer (Stay)
# ------------------------------
high_value_idx = np.argmin(y_proba)
high_value_stay = X_test.iloc[high_value_idx]

plt.figure()
shap.force_plot(
    explainer.expected_value,               # scalar again
    shap_values[high_value_idx].values,
    high_value_stay,
    matplotlib=True,
    show=False
)
plt.savefig("shap_force_high_value.png")
plt.close()
print("High-value customer SHAP force plot saved as 'shap_force_high_value.png'.")

# ------------------------------
# 7. Strategic Insights
# ------------------------------
print("\n--- Strategic Insights ---")
print("1. Target retention campaigns for customers at high churn risk (high predicted probability).")
print("2. Focus on high-value customers with low churn risk to maintain loyalty and reduce churn impact.")
print("3. Monitor key features driving churn and adjust business strategy to reduce risk.")
