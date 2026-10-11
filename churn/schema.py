"""Dataset schema: columns, allowed values, friendly labels and business driver groups."""
from __future__ import annotations

ID_COL = "customerID"
TARGET = "Churn"
TARGET_FLAG = "ChurnFlag"

NUMERIC_RAW = ["tenure", "MonthlyCharges", "TotalCharges"]
ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies"]
CATEGORICAL_RAW = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
                   "InternetService", *ADDON_SERVICES, "Contract", "PaperlessBilling", "PaymentMethod"]
RAW_COLUMNS = [ID_COL, "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
               "MultipleLines", "InternetService", *ADDON_SERVICES, "Contract", "PaperlessBilling",
               "PaymentMethod", "MonthlyCharges", "TotalCharges", TARGET]

_YN = ["No", "Yes"]
_YN_NOINT = ["No", "Yes", "No internet service"]
ALLOWED_VALUES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": _YN,  # raw file stores 0/1; mapped to No/Yes during cleaning
    "Partner": _YN,
    "Dependents": _YN,
    "PhoneService": _YN,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{c: _YN_NOINT for c in ADDON_SERVICES},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": _YN,
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"],
    TARGET: _YN,
}

# ---- engineered features used by the model --------------------------------------------------
CONTINUOUS_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "charge_delta"]
DISCRETE_FEATURES = ["total_services", "protection_count", "streaming_count", "billing_risk_score",
                     "contract_risk_score", "bill_increase_flag", "no_support_flag", "fiber_no_support",
                     "is_month_to_month", "electronic_check", "auto_pay"]
CATEGORICAL_FEATURES = CATEGORICAL_RAW + ["tenure_group", "customer_value_segment", "service_profile"]

FEATURE_LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Has partner",
    "Dependents": "Has dependents", "tenure": "Tenure (months)", "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines", "InternetService": "Internet service type",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies", "Contract": "Contract type",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
    "avg_monthly_spend": "Average monthly spend", "charge_delta": "Current bill vs. historical average",
    "total_services": "Number of services", "protection_count": "Protection add-ons (count)",
    "streaming_count": "Streaming add-ons (count)", "billing_risk_score": "Billing risk score",
    "contract_risk_score": "Contract risk score", "bill_increase_flag": "Bill above historical average",
    "no_support_flag": "Internet without security/tech support", "fiber_no_support": "Fiber without tech support",
    "is_month_to_month": "Month-to-month contract", "electronic_check": "Pays by electronic check",
    "auto_pay": "Automatic payment", "tenure_group": "Tenure band",
    "customer_value_segment": "Customer value segment", "service_profile": "Service profile",
}

# Business "driver groups": related raw/engineered columns that carry one business idea.
# Used for grouped permutation importance and local explanations so that redundant features
# (e.g. Contract + is_month_to_month) are measured together instead of splitting credit.
DRIVER_GROUPS = {
    "Contract type": ["Contract", "is_month_to_month", "contract_risk_score"],
    "Tenure / customer age": ["tenure", "tenure_group", "TotalCharges", "avg_monthly_spend"],
    "Monthly price level": ["MonthlyCharges", "customer_value_segment", "charge_delta", "bill_increase_flag"],
    "Payment & billing": ["PaymentMethod", "electronic_check", "auto_pay", "PaperlessBilling",
                          "billing_risk_score"],
    "Internet service type": ["InternetService", "fiber_no_support"],
    "Security & tech support": ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                                "protection_count", "no_support_flag"],
    "Streaming services": ["StreamingTV", "StreamingMovies", "streaming_count"],
    "Service mix": ["PhoneService", "MultipleLines", "total_services", "service_profile"],
    "Household / demographics": ["SeniorCitizen", "Partner", "Dependents", "gender"],
}


def column_label(col: str) -> str:
    return FEATURE_LABELS.get(col, col)
