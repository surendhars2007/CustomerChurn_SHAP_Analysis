# CustomerChurn_SHAP_Analysis

## Overview
This project predicts customer churn using a Gradient Boosting model (XGBoost) and provides interpretability using SHAP (SHapley Additive exPlanations). The focus is not just on high predictive accuracy but also on understanding **why customers churn** by analyzing feature contributions both globally and locally.

The project includes:
- Data preprocessing and handling class imbalance using SMOTE
- Training and evaluating a Gradient Boosting model (XGBoost)
- Generating **SHAP summary plots** for global feature importance
- Generating **SHAP force plots** for individual high-risk and high-value customers
- Extracting actionable business insights from model explanations

---

## Dataset
A synthetic inbuilt dataset is used for demonstration with 5000 samples and 15 features + target (`Churn`).  
- `Churn = 1` indicates a customer churned.  
- `Churn = 0` indicates a customer stayed.

---

## Setup

1. Clone this repository:

```bash
git clone <your-repo-url>
cd CustomerChurn_SHAP_Analysis

Install required dependencies:
pip install -r requirements.txt
Run the main script:
python keerthishap.py

Outputs generated:

shap_summary_plot.png → Global feature importance

shap_force_high_risk.png → Force plot for high-risk churner

shap_force_high_value.png → Force plot for high-value customer predicted to stay

Console output → Model metrics and strategic insights
Model Performance

Sample performance metrics from execution:

AUC Score: 0.9899

F1 Score: 0.9635

Classification report:

Class	Precision	Recall	F1-score	Support
0	0.98	0.99	0.98	696
1	0.97	0.96	0.96	304
Accuracy	-	-	0.98	1000
Strategic Insights

Retention campaigns should target customers with high predicted churn risk.

High-value customers with low churn probability should be prioritized to maintain loyalty.

Monitor and manage key churn drivers identified by SHAP to proactively reduce churn risk.

Repository Structure

keerthishap.py → Main Python script

requirements.txt → Python dependencies

shap_summary_plot.png → Global feature importance visualization

shap_force_high_risk.png → SHAP force plot for high-risk churner

shap_force_high_value.png → SHAP force plot for high-value customer

Dependencies

Python 3.10+

numpy

pandas

scikit-learn

imbalanced-learn

xgboost

shap

matplotlib

Install dependencies using:

pip install -r requirements.txt

Author

[keerthika]
