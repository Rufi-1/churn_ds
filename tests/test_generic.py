import numpy as np
import pandas as pd
import pytest

from churn.cleaning import CleaningLog
from churn.generic import (MAX_GENERIC_TRAIN, Y, clean_generic, explain_one, guess_positive_label, statistics_generic,
                           suggest_settings, validate_generic)
from churn.predict import predict_proba


def test_suggest_settings_finds_target_id_and_value(other_churn_df):
    s = suggest_settings(other_churn_df)
    assert s["target"] == "Churn" and s["id_col"] == "CustomerID" and s["value_col"] == "Total Spend" and s["positive"] == 1


def test_guess_positive_label_variants():
    assert guess_positive_label(pd.Series(["Yes", "No", "No"])) == "Yes"
    assert guess_positive_label(pd.Series(["stay", "stay", "leave"])) == "leave"       # minority class
    assert guess_positive_label(pd.Series([0, 1, 0, 0])) == 1


def test_validation_messages(other_churn_df):
    s = suggest_settings(other_churn_df)
    assert validate_generic(other_churn_df, s)[0] == []
    bad = {**s, "target": "Age"}
    assert any("does not look like a yes/no" in e for e in validate_generic(other_churn_df, bad)[0])
    tiny = other_churn_df.head(100)
    assert any("At least" in e for e in validate_generic(tiny, s)[0])
    rare = other_churn_df.copy(); rare["Churn"] = 0; rare.loc[:5, "Churn"] = 1
    assert any("Both outcomes" in e for e in validate_generic(rare, s)[0])
    assert any("numeric" in e for e in validate_generic(other_churn_df, {**s, "value_col": "Gender"})[0])


def test_cleaning_types_and_logging(other_churn_df):
    s = suggest_settings(other_churn_df)
    out = clean_generic(other_churn_df, s, log := CleaningLog())
    c = out["clean"]
    assert c.loc[3, "Gender"] == "Male" and Y in c and set(c[Y].unique()) <= {0, 1}
    assert "CustomerID" not in out["numeric"] + out["categorical"]
    assert "Support Calls" in out["numeric"] and "Gender" in out["categorical"]
    assert (log.to_frame()["column"] == "Age").any()          # missing value noted, left for the pipeline imputer


def test_text_numbers_and_idlike_columns_are_handled():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({"id": [f"u{i}" for i in range(n)], "spend": rng.uniform(1, 9, n).round(1).astype(str),
                       "token": [f"t{i}" for i in range(n)], "flag": rng.choice([True, False], n), "const": "x",
                       "y": rng.choice(["Yes", "No"], n)})
    out = clean_generic(df, {"target": "y", "positive": "Yes", "id_col": "id", "value_col": None, "exclude": []}, CleaningLog())
    assert "spend" in out["numeric"] and "token" not in out["categorical"] and "const" not in out["categorical"]
    assert "flag" in out["categorical"]


def test_statistics_generic(other_churn_df):
    s = suggest_settings(other_churn_df)
    out = clean_generic(other_churn_df, s, CleaningLog())
    r = statistics_generic(out["clean"], out["numeric"], out["categorical"], "Churn = 1")
    assert r["chi_square"].set_index("variable").loc["Contract Length", "significant"]
    n = r["numeric"].set_index("variable")
    assert n.loc["Support Calls", "median_positive"] > n.loc["Support Calls", "median_negative"] and n.loc["Support Calls", "significant"]
    assert r["numeric"]["p_adjusted"].between(0, 1).all()


def test_full_generic_run_outputs(generic_run, other_churn_df):
    r = generic_run
    assert r["model_name"] in {"Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"}
    assert r["metrics"]["roc_auc"] > 0.7 and not r["leakage_warning"]
    assert r["n_train"] + r["n_test"] == len(r["eda_df"])
    assert r["importance"].iloc[0]["feature"] in {"Support Calls", "Contract Length", "Tenure", "Payment Delay"}
    sc = r["scores"]
    assert len(sc) == len(other_churn_df) - 0 and sc["churn_probability"].between(0, 1).all() and sc["priority_rank"].is_unique
    assert set(sc["score_source"]) <= {"training (out-of-fold score)", "hold-out test", "not used in training (unseen)"}
    assert (sc["flagged"] == (sc["churn_probability"] >= r["threshold"])).all()
    assert sc.head(5)["top_risk_drivers"].str.len().gt(0).all()
    assert r["kpis"]["value_at_risk"] == pytest.approx(sc.loc[sc["flagged"], "Total Spend"].sum())
    assert "Analysis report" in r["report_md"] and "association" in r["report_md"].lower()
    assert set(r["figures"]) >= {"ROC curve", "Confusion matrix"} and all(b[:4] == b"\x89PNG" for b in r["figures"].values())


def test_test_rows_never_in_training(generic_run):
    sc = generic_run["scores"]
    assert (sc["score_source"] == "hold-out test").sum() == generic_run["n_test"]


def test_single_customer_prediction_and_explanation(generic_run):
    r = generic_run
    row = r["eda_df"][r["features"]].head(1)
    p = predict_proba(r["bundle"], row)[0]
    assert 0 <= p <= 1
    ex = explain_one(r, row, top_n=4)
    assert len(ex) == 4 and all("percentage points" in t for t in ex["explanation"])
    hi = row.copy(); hi["Support Calls"] = 9; hi["Contract Length"] = "Monthly"; hi["Tenure"] = 1
    lo = row.copy(); lo["Support Calls"] = 0; lo["Contract Length"] = "Annual"; lo["Tenure"] = 55
    assert predict_proba(r["bundle"], hi)[0] > predict_proba(r["bundle"], lo)[0] + 0.3


def test_string_labels_and_large_file_sampling(monkeypatch, other_churn_df):
    import churn.generic as g
    df = other_churn_df.copy()
    df["Churn"] = np.where(df["Churn"] == 1, "Yes", "No")
    s = g.suggest_settings(df)
    assert s["positive"] == "Yes"
    monkeypatch.setattr(g, "MAX_GENERIC_TRAIN", 600)
    monkeypatch.setattr(g, "MAX_GENERIC_SCORE", 900)
    from churn.upload import TRAINING_PROFILES
    r = g.run_generic_analysis(df, s, profile=list(TRAINING_PROFILES)[0])
    assert len(r["eda_df"]) == 600 and len(r["scores"]) == 900
    assert (r["scores"]["score_source"] == "not used in training (unseen)").any()
