"""Hypothesis tests: chi-square (categorical), t-test / Mann-Whitney U (numeric), correlations,
two-proportion comparisons. Reports statistics, p-values (raw + Benjamini-Hochberg adjusted),
effect sizes and business-friendly wording. Association is never presented as causation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config, resolve_path
from .features import engineer_features
from .schema import CATEGORICAL_RAW, TARGET, TARGET_FLAG, column_label
from .utils import df_to_markdown, get_logger, now_iso, save_json

log = get_logger()
CAUSATION_NOTE = ("These tests show statistical **association**, not causation. Customers who differ on one "
                  "attribute usually differ on many others (e.g. month-to-month customers are also newer), "
                  "so a significant result is a lead to investigate, not proof that changing the attribute "
                  "would change churn. Only a controlled retention experiment (A/B test) can show cause.")
NUMERIC_TEST_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "total_services"]


def cramers_v(table: pd.DataFrame | np.ndarray) -> float:
    t = np.asarray(table, dtype=float)
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    n = t.sum()
    k = min(t.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else 0.0


def cramers_v_label(v: float) -> str:
    return "negligible" if v < 0.1 else "small" if v < 0.3 else "moderate" if v < 0.5 else "large"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def cohens_d_label(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


def rank_biserial(u: float, n1: int, n2: int) -> float:
    return float(1 - 2 * u / (n1 * n2))


def rank_biserial_label(r: float) -> str:
    r = abs(r)
    return "negligible" if r < 0.1 else "small" if r < 0.3 else "medium" if r < 0.5 else "large"


def benjamini_hochberg(p_values) -> np.ndarray:
    """FDR-adjusted p-values (controls false discoveries across many tests)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def chi_square_tests(df: pd.DataFrame, cols: list[str], alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for col in cols:
        d = df[[col, TARGET]].dropna()
        table = pd.crosstab(d[col], d[TARGET])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p, dof, expected = stats.chi2_contingency(table)
        v = cramers_v(table)
        rates = (table["Yes"] / table.sum(axis=1)).sort_values()
        hi, lo = rates.index[-1], rates.index[0]
        rows.append({"variable": col, "label": column_label(col), "chi2": chi2, "dof": dof, "p_value": p,
                     "cramers_v": v, "effect_size": cramers_v_label(v),
                     "min_expected_count": float(expected.min()),
                     "highest_churn_group": hi, "highest_churn_rate": rates.iloc[-1],
                     "lowest_churn_group": lo, "lowest_churn_rate": rates.iloc[0]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_adjusted"] = benjamini_hochberg(out["p_value"])
    out["significant"] = out["p_adjusted"] < alpha
    out["interpretation"] = [_interp_cat(r, alpha) for r in out.itertuples()]
    return out.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def _interp_cat(r, alpha: float) -> str:
    if r.p_adjusted < alpha:
        return (f"Churn differs significantly across {r.label.lower()} groups ({r.effect_size} association, "
                f"Cramer's V={r.cramers_v:.2f}): '{r.highest_churn_group}' churns at "
                f"{100 * r.highest_churn_rate:.1f}% vs {100 * r.lowest_churn_rate:.1f}% for "
                f"'{r.lowest_churn_group}'.")
    return (f"No statistically significant difference in churn across {r.label.lower()} groups "
            f"(adjusted p={fmt_p(r.p_adjusted)}).")


def numeric_tests(df: pd.DataFrame, cols: list[str], alpha: float = 0.05, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for col in cols:
        d = df[[col, TARGET]].dropna()
        a = d.loc[d[TARGET] == "Yes", col].to_numpy(float)
        b = d.loc[d[TARGET] == "No", col].to_numpy(float)
        if len(a) < 3 or len(b) < 3:
            continue
        sh = [stats.shapiro(rng.choice(x, size=min(500, len(x)), replace=False))[1] if np.ptp(x) > 0 else 0.0
              for x in (a, b)]
        normal = all(p > 0.05 for p in sh)
        if normal:
            stat, p = stats.ttest_ind(a, b, equal_var=False)
            test = "Welch t-test"
        else:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            stat, test = u, "Mann-Whitney U"
        u_stat = stats.mannwhitneyu(a, b, alternative="two-sided")[0]
        d_val, rb = cohens_d(a, b), rank_biserial(u_stat, len(a), len(b))
        eff_name, eff, eff_lab = (("Cohen's d", d_val, cohens_d_label(d_val)) if normal
                                  else ("rank-biserial r", rb, rank_biserial_label(rb)))
        rows.append({"variable": col, "label": column_label(col), "test": test, "statistic": float(stat),
                     "p_value": float(p), "effect_size_name": eff_name, "effect_size": eff,
                     "effect_magnitude": eff_lab, "cohens_d": d_val, "rank_biserial": rb,
                     "normal_both_groups": normal, "n_churned": len(a), "n_retained": len(b),
                     "mean_churned": a.mean(), "mean_retained": b.mean(),
                     "median_churned": float(np.median(a)), "median_retained": float(np.median(b))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_adjusted"] = benjamini_hochberg(out["p_value"])
    out["significant"] = out["p_adjusted"] < alpha
    out["interpretation"] = [
        (f"Churned customers have {'higher' if r.median_churned > r.median_retained else 'lower'} "
         f"{r.label.lower()} (median {r.median_churned:,.1f} vs {r.median_retained:,.1f}); "
         f"{r.test} p={fmt_p(r.p_adjusted)}, {r.effect_magnitude} effect ({r.effect_size_name}={r.effect_size:.2f}).")
        if r.p_adjusted < alpha else
        f"No significant difference in {r.label.lower()} between churned and retained customers "
        f"(adjusted p={fmt_p(r.p_adjusted)})." for r in out.itertuples()]
    return out


def two_proportion_test(flag_a: pd.Series, y_a: pd.Series, y_b: pd.Series) -> dict:
    na, nb = len(y_a), len(y_b)
    pa, pb = y_a.mean(), y_b.mean()
    pooled = (y_a.sum() + y_b.sum()) / (na + nb)
    se0 = np.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
    z = (pa - pb) / se0 if se0 > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se1 = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    return {"n_a": na, "n_b": nb, "rate_a": pa, "rate_b": pb, "difference": pa - pb,
            "ci_low": pa - pb - 1.96 * se1, "ci_high": pa - pb + 1.96 * se1,
            "risk_ratio": pa / pb if pb > 0 else np.nan, "z": z, "p_value": p}


def key_group_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET_FLAG]
    inet = df["InternetService"].isin(["DSL", "Fiber optic"])
    specs = [
        ("Month-to-month vs Two-year contract", df["Contract"] == "Month-to-month", df["Contract"] == "Two year"),
        ("Fiber optic vs DSL internet", df["InternetService"] == "Fiber optic", df["InternetService"] == "DSL"),
        ("Electronic check vs other payment methods", df["PaymentMethod"] == "Electronic check",
         df["PaymentMethod"] != "Electronic check"),
        ("Senior citizens vs non-seniors", df["SeniorCitizen"] == "Yes", df["SeniorCitizen"] == "No"),
        ("No tech support vs tech support (internet customers)", inet & (df["TechSupport"] == "No"),
         inet & (df["TechSupport"] == "Yes")),
        ("First-year (<=12 months) vs longer-tenure customers", df["tenure"] <= 12, df["tenure"] > 12),
        ("Paperless vs paper billing", df["PaperlessBilling"] == "Yes", df["PaperlessBilling"] == "No"),
        ("Male vs female customers", df["gender"] == "Male", df["gender"] == "Female"),
    ]
    rows = []
    for name, ma, mb in specs:
        if ma.sum() < 5 or mb.sum() < 5:
            continue
        r = two_proportion_test(ma, y[ma], y[mb])
        rows.append({"comparison": name, **r})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adjusted"] = benjamini_hochberg(out["p_value"])
        out["interpretation"] = [
            f"{r.comparison}: churn {100 * r.rate_a:.1f}% vs {100 * r.rate_b:.1f}% "
            f"(difference {100 * r.difference:+.1f} pts, 95% CI {100 * r.ci_low:+.1f} to {100 * r.ci_high:+.1f}; "
            f"risk ratio {r.risk_ratio:.2f}; adjusted p={fmt_p(r.p_adjusted)})." for r in out.itertuples()]
    return out


def correlation_analysis(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.DataFrame]:
    num = df[cols + [TARGET_FLAG]].dropna()
    pear = num.corr(method="pearson")
    spear = num.corr(method="spearman")
    rows = []
    for c in cols:
        r, p = stats.pointbiserialr(num[TARGET_FLAG], num[c])
        rho, p2 = stats.spearmanr(num[c], num[TARGET_FLAG])
        rows.append({"variable": c, "label": column_label(c), "pearson_r": r, "pearson_p": p,
                     "spearman_rho": rho, "spearman_p": p2})
    return {"pearson": pear, "spearman": spear,
            "with_churn": pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)}


def run_statistical_tests(clean: pd.DataFrame, cfg: dict | None = None, save: bool = True) -> dict:
    cfg = cfg or load_config()
    alpha = cfg["stats"]["alpha"]
    df = engineer_features(clean, cfg)
    cat_cols = [c for c in CATEGORICAL_RAW if c in df.columns]
    res = {
        "chi_square": chi_square_tests(df, cat_cols, alpha),
        "numeric": numeric_tests(df, NUMERIC_TEST_COLS, alpha, cfg["random_seed"]),
        "groups": key_group_comparisons(df),
        "correlation": correlation_analysis(df, NUMERIC_TEST_COLS),
    }
    if save:
        tdir = resolve_path(cfg, "tables_dir", create=True)
        rdir = resolve_path(cfg, "reports_dir", create=True)
        res["chi_square"].to_csv(tdir / "stats_chi_square.csv", index=False)
        res["numeric"].to_csv(tdir / "stats_numeric_tests.csv", index=False)
        res["groups"].to_csv(tdir / "stats_group_comparisons.csv", index=False)
        res["correlation"]["with_churn"].to_csv(tdir / "stats_correlation_with_churn.csv", index=False)
        res["correlation"]["pearson"].to_csv(tdir / "stats_correlation_pearson.csv")
        res["correlation"]["spearman"].to_csv(tdir / "stats_correlation_spearman.csv")
        (rdir / "statistical_report.md").write_text(statistics_markdown(res, alpha), encoding="utf-8")
        save_json({"generated_at": now_iso(), "alpha": alpha}, tdir / "stats_meta.json")
        log.info("Statistical report saved to %s", rdir / "statistical_report.md")
    return res


def statistics_markdown(res: dict, alpha: float) -> str:
    chi, num, grp, cor = res["chi_square"], res["numeric"], res["groups"], res["correlation"]["with_churn"]
    show_chi = chi[["label", "chi2", "dof", "p_value", "p_adjusted", "cramers_v", "effect_size"]].copy()
    show_num = num[["label", "test", "statistic", "p_adjusted", "effect_size_name", "effect_size",
                    "effect_magnitude", "median_churned", "median_retained"]].copy()
    show_grp = grp[["comparison", "rate_a", "rate_b", "difference", "risk_ratio", "p_adjusted"]].copy()
    lines = [
        "# Statistical Insights", "",
        f"Significance level alpha = {alpha}. p-values are adjusted across tests with the "
        "Benjamini-Hochberg procedure to limit false discoveries.", "",
        f"> {CAUSATION_NOTE}", "",
        "## 1. Chi-square tests: churn vs categorical variables (sorted by effect size)",
        df_to_markdown(show_chi, 4), "", *[f"- {t}" for t in chi["interpretation"]], "",
        "## 2. Numeric variables: churned vs retained customers",
        "Mann-Whitney U is used when either group is non-normal (Shapiro-Wilk, p<0.05); otherwise Welch's t-test.",
        "", df_to_markdown(show_num, 4), "", *[f"- {t}" for t in num["interpretation"]], "",
        "## 3. Business-relevant group comparisons (two-proportion z-tests)",
        df_to_markdown(show_grp, 4), "", *[f"- {t}" for t in grp["interpretation"]], "",
        "## 4. Correlation with churn (numeric variables)", df_to_markdown(cor, 4), "",
        "Large |r| between predictors (e.g. tenure and total charges) signals multicollinearity: "
        "see the EDA report for VIF values.", "",
    ]
    return "\n".join(lines)
