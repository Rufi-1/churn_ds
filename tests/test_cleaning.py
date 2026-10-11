import numpy as np
import pandas as pd
import pytest

from churn.cleaning import (clean_dataset, detect_outliers, standardize_frame, strip_whitespace, CleaningLog)
from churn.config import load_config


@pytest.fixture()
def dirty(mini_raw):
    d = mini_raw.head(30).copy()
    d["TotalCharges"] = d["TotalCharges"].astype(object)
    d.loc[0, "gender"] = "  Female "
    d.loc[1, "Contract"] = "month to month"
    d.loc[2, "Partner"] = "yes"
    d.loc[3, "InternetService"] = "Satellite"            # invalid category
    d.loc[4, ["tenure", "TotalCharges"]] = [0, " "]      # not billed yet
    d.loc[5, "MonthlyCharges"] = -10                     # impossible value
    d.loc[6, "Churn"] = None                             # unusable label
    d.loc[7, "InternetService"] = "No"                   # add-ons must become 'No internet service'
    d.loc[7, "OnlineSecurity"] = "Yes"
    d = pd.concat([d, d.iloc[[8]]], ignore_index=True)   # exact duplicate
    return d


def test_clean_returns_log_and_flag(dirty):
    clean, log = clean_dataset(dirty, load_config())
    assert {"step", "column", "issue", "rows_affected", "action"} <= set(log.columns)
    assert "ChurnFlag" in clean and set(clean["ChurnFlag"].unique()) <= {0, 1}
    assert (clean["ChurnFlag"] == (clean["Churn"] == "Yes")).all()


def test_whitespace_and_case_are_standardised(dirty):
    clean, _ = clean_dataset(dirty, load_config())
    assert clean.loc[0, "gender"] == "Female"
    assert clean.loc[1, "Contract"] == "Month-to-month"
    assert clean.loc[2, "Partner"] == "Yes"


def test_invalid_category_becomes_nan_and_logged(dirty):
    clean, log = clean_dataset(dirty, load_config())
    assert pd.isna(clean.loc[3, "InternetService"])
    assert ((log["column"] == "InternetService") & (log["issue"] == "invalid category")).any()


def test_blank_total_charges_for_new_customer_is_zero(dirty):
    clean, _ = clean_dataset(dirty, load_config())
    assert clean.loc[4, "TotalCharges"] == 0.0
    assert clean["TotalCharges"].dtype == float


def test_negative_charge_nulled_not_imputed_here(dirty):
    clean, _ = clean_dataset(dirty, load_config())
    assert pd.isna(clean.loc[5, "MonthlyCharges"])  # imputation is deferred to the training pipeline


def test_missing_target_rows_dropped_and_duplicates_removed(dirty):
    clean, log = clean_dataset(dirty, load_config())
    assert len(clean) == len(dirty) - 2  # one missing label + one duplicate
    assert clean["customerID"].is_unique
    assert (log["step"] == "duplicates").any() and (log["step"] == "target").any()


def test_internet_addon_consistency(dirty):
    clean, _ = clean_dataset(dirty, load_config())
    cid = dirty.loc[7, "customerID"]  # row 6 is dropped earlier, so look up by id, not position
    assert clean.set_index("customerID").loc[cid, "OnlineSecurity"] == "No internet service"


def test_seniorcitizen_mapped_to_yes_no(mini_raw):
    clean, _ = clean_dataset(mini_raw, load_config())
    assert set(clean["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_missing_required_column_raises(mini_raw):
    with pytest.raises(ValueError, match="missing required columns"):
        clean_dataset(mini_raw.drop(columns=["Contract"]), load_config())


def test_cleaning_is_idempotent_and_reproducible(mini_raw):
    cfg = load_config()
    a, _ = clean_dataset(mini_raw, cfg)
    b, _ = clean_dataset(mini_raw, cfg)
    pd.testing.assert_frame_equal(a, b)
    c, _ = clean_dataset(a.drop(columns=["ChurnFlag"]), cfg)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), c.reset_index(drop=True))


def test_detect_outliers_flags_extreme_value():
    df = pd.DataFrame({"tenure": [10] * 50 + [1000]})
    out = detect_outliers(df, ["tenure"])
    assert out.loc[0, "n_outliers"] == 1


def test_strip_whitespace_collapses_spaces():
    df = pd.DataFrame({"a": ["  x   y ", "   ", None]})
    out = strip_whitespace(df, CleaningLog())
    assert out["a"].iloc[0] == "x y" and pd.isna(out["a"].iloc[1]) and pd.isna(out["a"].iloc[2])


def test_standardize_frame_keeps_all_rows(mini_raw):
    d = mini_raw.head(10).drop(columns=["Churn"])
    out = standardize_frame(pd.concat([d, d.iloc[[0]]]))
    assert len(out) == 11
