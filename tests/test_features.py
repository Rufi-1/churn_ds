import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from churn.cleaning import clean_dataset, standardize_frame
from churn.config import load_config
from churn.features import (Winsorizer, build_preprocessor, engineer_features, input_columns, model_columns,
                            prepare_modeling_data, pretty_feature_name, source_column)
from churn.predict import make_customer_frame


def eng(**kw):
    return engineer_features(standardize_frame(make_customer_frame(**kw)), load_config()).iloc[0]


def test_engineered_values_for_known_customer():
    r = eng(tenure=3, MonthlyCharges=100.0, TotalCharges=300.0, Contract="Month-to-month", InternetService="Fiber optic",
            PaymentMethod="Electronic check", PaperlessBilling="Yes", TechSupport="No", OnlineSecurity="No",
            StreamingTV="Yes", StreamingMovies="Yes")
    assert r["tenure_group"] == "0-6 months"
    assert r["avg_monthly_spend"] == pytest.approx(100.0)
    assert r["is_month_to_month"] == 1 and r["contract_risk_score"] == 2
    assert r["electronic_check"] == 1 and r["auto_pay"] == 0
    assert r["streaming_count"] == 2 and r["protection_count"] == 0
    assert r["total_services"] == 1 + 1 + 2          # phone + internet + 2 streaming
    assert r["no_support_flag"] == 1 and r["fiber_no_support"] == 1
    assert r["billing_risk_score"] == 3              # m2m + e-check + paperless
    assert r["customer_value_segment"] == "Premium"
    assert r["service_profile"] == "Phone+Internet+Streaming"


def test_zero_tenure_uses_monthly_charge_not_division_by_zero():
    r = eng(tenure=0, MonthlyCharges=55.0, TotalCharges=0.0)
    assert r["avg_monthly_spend"] == 55.0 and np.isfinite(r["charge_delta"])


def test_two_year_autopay_customer_is_low_risk_encoded():
    r = eng(Contract="Two year", PaymentMethod="Bank transfer (automatic)", PaperlessBilling="No", tenure=60,
            MonthlyCharges=50.0, TotalCharges=3000.0)
    assert r["contract_risk_score"] == 0 and r["auto_pay"] == 1 and r["billing_risk_score"] == 0
    assert r["tenure_group"] == "49+ months"


def test_modeling_data_excludes_gender_and_target_leakage(mini_raw):
    cfg = load_config()
    clean, _ = clean_dataset(mini_raw, cfg)
    m = prepare_modeling_data(clean, cfg, save=False)
    assert "gender" not in m.columns and "Churn" not in m.columns
    assert "ChurnFlag" in m.columns and set(input_columns(cfg)) <= set(m.columns)
    assert "ChurnFlag" not in input_columns(cfg)


def test_winsorizer_uses_training_quantiles_only():
    train = np.arange(100, dtype=float).reshape(-1, 1)
    w = Winsorizer(0.05, 0.95).fit(train)
    out = w.transform(np.array([[-1000.0], [50.0], [1000.0]]))
    assert out[0, 0] == pytest.approx(np.quantile(train, 0.05)) and out[2, 0] == pytest.approx(np.quantile(train, 0.95))
    assert out[1, 0] == 50.0


def test_preprocessor_fit_on_train_only_no_leakage(mini_raw):
    cfg = load_config()
    clean, _ = clean_dataset(mini_raw, cfg)
    m = prepare_modeling_data(clean, cfg, save=False)
    X = m[input_columns(cfg)]
    train, test = X.iloc[:350], X.iloc[350:]
    pre = build_preprocessor(cfg).fit(train)
    scaler = pre.named_transformers_["cont"].named_steps["scale"]
    before = scaler.mean_.copy()
    pre.transform(test)
    assert np.allclose(before, scaler.mean_)                       # transform never refits
    full = build_preprocessor(cfg).fit(X).named_transformers_["cont"].named_steps["scale"]
    assert not np.allclose(before, full.mean_)                     # train-only statistics differ from all-data stats


def test_preprocessor_handles_missing_and_unknown_categories(mini_raw):
    cfg = load_config()
    clean, _ = clean_dataset(mini_raw, cfg)
    X = prepare_modeling_data(clean, cfg, save=False)[input_columns(cfg)]
    pre = build_preprocessor(cfg).fit(X)
    new = X.head(3).copy()
    new.loc[new.index[0], "Contract"] = "Never seen"
    new.loc[new.index[1], "MonthlyCharges"] = np.nan
    out = pre.transform(new)
    assert np.isfinite(out).all() and out.shape[1] == len(pre.get_feature_names_out())


def test_feature_name_helpers():
    assert pretty_feature_name("cat__Contract_Month-to-month") == "Contract type: Month-to-month"
    assert pretty_feature_name("cont__tenure") == "Tenure (months)"
    assert source_column("cat__PaymentMethod_Electronic check", ["PaymentMethod", "tenure"]) == "PaymentMethod"


def test_model_columns_partition():
    cont, disc, cat = model_columns(load_config())
    assert not set(cont) & set(disc) and not set(disc) & set(cat)
