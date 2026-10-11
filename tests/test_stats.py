import numpy as np
import pandas as pd
import pytest

from churn.cleaning import clean_dataset
from churn.config import load_config
from churn.features import engineer_features
from churn.stats_tests import (CAUSATION_NOTE, benjamini_hochberg, chi_square_tests, cohens_d, correlation_analysis,
                               cramers_v, key_group_comparisons, numeric_tests, rank_biserial, run_statistical_tests,
                               statistics_markdown, two_proportion_test)


@pytest.fixture(scope="module")
def eng_df(mini_raw):
    cfg = load_config()
    clean, _ = clean_dataset(mini_raw, cfg)
    return engineer_features(clean, cfg)


def test_cramers_v_extremes():
    assert cramers_v(np.array([[50, 0], [0, 50]])) == pytest.approx(1.0)
    assert cramers_v(np.array([[25, 25], [25, 25]])) == pytest.approx(0.0)


def test_benjamini_hochberg_properties():
    p = np.array([0.001, 0.01, 0.03, 0.2, 0.5])
    adj = benjamini_hochberg(p)
    assert (adj >= p).all() and (adj <= 1).all()
    assert np.all(np.diff(adj[np.argsort(p)]) >= 0)


def test_effect_sizes():
    a, b = np.array([5., 6, 7, 8]), np.array([1., 2, 3, 4])
    assert cohens_d(a, b) > 2
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(a, b)[0]
    assert rank_biserial(u, 4, 4) == pytest.approx(-1.0) or rank_biserial(u, 4, 4) == pytest.approx(1.0)


def test_chi_square_table(eng_df):
    t = chi_square_tests(eng_df, ["Contract", "gender", "InternetService"])
    assert {"chi2", "p_value", "p_adjusted", "cramers_v", "interpretation", "significant"} <= set(t.columns)
    assert t["p_value"].between(0, 1).all()
    assert t.iloc[0]["variable"] == "Contract" or t.iloc[0]["cramers_v"] >= t.iloc[-1]["cramers_v"]
    assert t.set_index("variable").loc["Contract", "significant"]  # contract is strongly associated with churn


def test_numeric_tests_pick_valid_test(eng_df):
    t = numeric_tests(eng_df, ["tenure", "MonthlyCharges"])
    assert set(t["test"]) <= {"Welch t-test", "Mann-Whitney U"}
    assert t.set_index("variable").loc["tenure", "median_churned"] < t.set_index("variable").loc["tenure", "median_retained"]


def test_two_proportion_test():
    a, b = pd.Series([1] * 40 + [0] * 60), pd.Series([1] * 10 + [0] * 90)
    r = two_proportion_test(None, a, b)
    assert r["difference"] == pytest.approx(0.3) and r["p_value"] < 0.001 and r["ci_low"] < 0.3 < r["ci_high"]


def test_group_comparisons_and_correlation(eng_df):
    g = key_group_comparisons(eng_df)
    assert len(g) >= 5 and g["p_adjusted"].between(0, 1).all()
    c = correlation_analysis(eng_df, ["tenure", "MonthlyCharges", "TotalCharges"])
    assert c["pearson"].shape == (4, 4) and c["pearson"].loc["tenure", "tenure"] == pytest.approx(1.0)


def test_report_states_association_not_causation(eng_df):
    res = {"chi_square": chi_square_tests(eng_df, ["Contract"]), "numeric": numeric_tests(eng_df, ["tenure"]),
           "groups": key_group_comparisons(eng_df), "correlation": correlation_analysis(eng_df, ["tenure", "MonthlyCharges"])}
    md = statistics_markdown(res, 0.05)
    assert "causation" in md.lower() and "Benjamini-Hochberg" in md and CAUSATION_NOTE in md
