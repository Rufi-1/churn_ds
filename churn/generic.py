"""Any-dataset mode: analyse and model ANY tabular file that has a binary outcome column.

The user chooses the target column (and which value is the "positive" outcome), optionally an ID column and a
value column (e.g. spend). Everything else is inferred: column types, cleaning, quality profile, EDA, statistical tests,
model tuning/selection (Logistic Regression, Random Forest, Gradient Boosting, XGBoost), evaluation on a hold-out set,
explainability, scoring of every customer, and a downloadable report.

Large files: modeling/EDA use a stratified sample of at most MAX_GENERIC_TRAIN rows; up to MAX_GENERIC_SCORE rows are scored.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cleaning import CleaningLog, _norm_key
from .config import load_config
from .evaluation import (compute_metrics, gains_table, plot_calibration, plot_confusion, plot_gains,
                         plot_probability_distribution, plot_pr, plot_roc)
from .explain import marginal_effects, permutation_importance_table
from .features import Winsorizer
from .modeling import DISPLAY, _search, get_candidates, native_importance, select_best, tune_threshold
from .predict import assign_risk_tier, predict_proba
from .stats_tests import (benjamini_hochberg, cohens_d, cohens_d_label, cramers_v, cramers_v_label, fmt_p, rank_biserial,
                          rank_biserial_label)
from .upload import PIPELINE_LOCK, TRAINING_PROFILES, _merge_over
from .utils import df_to_markdown, get_logger, now_iso, set_plot_style

log = get_logger()
MAX_GENERIC_TRAIN = 20_000
MAX_GENERIC_SCORE = 100_000
MIN_ROWS = 300
MIN_CLASS = 30
Y = "__target__"
POS_HINTS = ["yes", "1", "true", "churn", "churned", "left", "exited", "y", "cancelled", "canceled"]
TARGET_HINTS = ["churn", "exited", "attrition", "left", "cancel", "target", "label", "outcome", "default"]
VALUE_HINTS = ["spend", "charge", "revenue", "amount", "value", "price", "bill", "sales", "income", "fee"]
STEP_LABELS = {"prepare": "Cleaning data and profiling columns", "stats": "Exploratory statistics and hypothesis tests",
               "train": "Training, tuning and comparing models", "evaluate": "Evaluating on the hold-out test set",
               "explain": "Explaining the model", "score": "Scoring customers and building the report"}


def _norm_label(v) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v)).lower()
    return str(v).strip().lower()


# --------------------------------------------------------------------------------------- suggestions & validation
def guess_positive_label(series: pd.Series):
    vals = series.dropna().unique().tolist()
    for v in vals:
        if _norm_label(v) in POS_HINTS:
            return v
    counts = series.value_counts()
    return counts.index[-1] if len(counts) else None


def suggest_settings(df: pd.DataFrame) -> dict:
    """Best-guess target, positive label, ID and value columns."""
    n = len(df)

    def tscore(c):
        k, nun = _norm_key(c), df[c].nunique(dropna=True)
        s = 10 if any(h in k for h in TARGET_HINTS) else 0
        return s + (5 if nun == 2 else 1 if 2 < nun <= 5 else -100)

    cands = sorted(df.columns, key=tscore, reverse=True)
    target = cands[0] if cands and tscore(cands[0]) > 0 else None
    id_col = next((c for c in df.columns if (_norm_key(c).endswith("id") or _norm_key(c) in ("customer", "id"))
                   and df[c].nunique() > 0.9 * n and c != target), None)
    value = next((c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in (target, id_col)
                  and any(h in _norm_key(c) for h in VALUE_HINTS)), None)
    return {"target": target, "positive": guess_positive_label(df[target]) if target else None,
            "id_col": id_col, "value_col": value}


def validate_generic(df: pd.DataFrame, s: dict) -> tuple[list[str], list[str], dict]:
    errors, warnings, info = [], [], {}
    t = s.get("target")
    if not t or t not in df.columns:
        return ["Choose the target (outcome) column."], warnings, info
    if s.get("positive") is None:
        return ["Choose which value of the target means 'churned / positive'."], warnings, info
    lab = df[t].dropna().map(_norm_label)
    pos = _norm_label(s["positive"])
    n_pos, n_neg = int((lab == pos).sum()), int((lab != pos).sum())
    info = {"n_rows": len(df), "n_labeled": len(lab), "n_pos": n_pos, "n_neg": n_neg,
            "pos_rate": n_pos / max(len(lab), 1), "n_unique_target": int(lab.nunique())}
    if len(df) - len(lab):
        warnings.append(f"{len(df) - len(lab):,} rows have no target value and will be dropped.")
    if lab.nunique() > 10:
        errors.append(f"'{t}' has {lab.nunique()} distinct values - it does not look like a yes/no outcome. Pick another target column.")
    if len(lab) < MIN_ROWS:
        errors.append(f"At least {MIN_ROWS} labelled rows are needed (found {len(lab)}).")
    if min(n_pos, n_neg) < MIN_CLASS:
        errors.append(f"Both outcomes need at least {MIN_CLASS} rows (positive={n_pos}, other={n_neg}). Check the positive label.")
    if lab.nunique() > 2 and not errors:
        warnings.append(f"The target has {lab.nunique()} values: '{s['positive']}' is treated as positive, all others as negative.")
    excl = set(s.get("exclude", [])) | {t, s.get("id_col")}
    feats = [c for c in df.columns if c not in excl]
    if not feats:
        errors.append("No feature columns left to learn from.")
    if s.get("value_col") and not pd.api.types.is_numeric_dtype(df[s["value_col"]]):
        errors.append(f"The value column '{s['value_col']}' must be numeric.")
    if len(df) > MAX_GENERIC_TRAIN:
        warnings.append(f"{len(df):,} rows: the model is trained on a stratified sample of {MAX_GENERIC_TRAIN:,} rows to stay fast; "
                        f"up to {MAX_GENERIC_SCORE:,} rows are scored.")
    info["n_features"] = len(feats)
    return errors, warnings, info


# --------------------------------------------------------------------------------------- cleaning
def clean_generic(df: pd.DataFrame, s: dict, clog: CleaningLog) -> dict:
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    t, id_col, val = s["target"], s.get("id_col"), s.get("value_col")
    for c in d.columns:
        if pd.api.types.is_bool_dtype(d[c]):
            d[c] = d[c].astype(str)
        elif pd.api.types.is_object_dtype(d[c]) or pd.api.types.is_string_dtype(d[c]):
            o = d[c].astype(object)
            m = o.notna()
            stripped = o[m].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            changed = int((stripped != o[m].astype(str)).sum())
            out = pd.Series(np.nan, index=d.index, dtype=object)
            out[m] = stripped
            blank = out == ""
            out[blank] = np.nan
            d[c] = out
            if changed:
                clog.add("whitespace", c, "leading/trailing/duplicate whitespace", changed, "trimmed")
            if blank.sum():
                clog.add("missing", c, "blank string", int(blank.sum()), "converted to NaN")
            non_null = d[c].notna().sum()
            if c not in (t, id_col) and non_null:
                num = pd.to_numeric(d[c], errors="coerce")
                if num.notna().sum() >= 0.9 * non_null:
                    d[c] = num
                    bad = int(non_null - num.notna().sum())
                    clog.add("dtype", c, "numbers stored as text", int(non_null), "converted to numeric" +
                             (f"; {bad} unparsable values set to NaN" if bad else ""))
    miss_t = d[t].isna()
    if miss_t.sum():
        clog.add("target", t, "missing target value", int(miss_t.sum()), "rows dropped")
        d = d[~miss_t]
    before = len(d)
    d = d.drop_duplicates()
    if before - len(d):
        clog.add("duplicates", "all columns", "exact duplicate rows", before - len(d), "dropped (kept first)")
    d = d.reset_index(drop=True)
    d[Y] = (d[t].map(_norm_label) == _norm_label(s["positive"])).astype(int)
    excl = {t, id_col, Y} | set(s.get("exclude", []))
    feats = [c for c in d.columns if c not in excl]
    numeric, categorical = [], []
    for c in feats:
        nun = d[c].nunique(dropna=True)
        if nun <= 1:
            clog.add("features", c, "constant column", len(d), "excluded from analysis")
        elif pd.api.types.is_numeric_dtype(d[c]):
            numeric.append(c)
        elif nun > 200 or (nun > 50 and nun > 0.5 * len(d)):
            clog.add("features", c, f"{nun:,} distinct text values (ID-like / free text)", len(d), "excluded from analysis")
        else:
            categorical.append(c)
    for c in numeric:
        inf = int(np.isinf(d[c]).sum())
        if inf:
            d[c] = d[c].replace([np.inf, -np.inf], np.nan)
            clog.add("validity", c, "infinite values", inf, "set to NaN")
    remaining = {c: int(d[c].isna().sum()) for c in numeric + categorical if d[c].isna().any()}
    for c, n in remaining.items():
        clog.add("missing", c, "remaining missing values", n, "kept; imputed inside the training pipeline (no leakage)")
    return {"clean": d, "numeric": numeric, "categorical": categorical}


def quality_tables(raw: pd.DataFrame, clean: pd.DataFrame, numeric, categorical) -> dict:
    prof = pd.DataFrame({"column": raw.columns, "raw_dtype": [str(raw[c].dtype) for c in raw.columns],
                         "missing": [int(raw[c].isna().sum()) for c in raw.columns],
                         "missing_pct": [100 * raw[c].isna().mean() for c in raw.columns],
                         "unique_values": [int(raw[c].nunique(dropna=True)) for c in raw.columns]})
    prof["role"] = ["numeric feature" if c in numeric else "categorical feature" if c in categorical else "not used"
                    for c in raw.columns]
    num = clean[numeric].describe().T.reset_index().rename(columns={"index": "column"}) if numeric else pd.DataFrame()
    if len(num):
        num["skew"] = [clean[c].skew() for c in num["column"]]
        q1, q3 = clean[numeric].quantile(.25), clean[numeric].quantile(.75)
        iqr = q3 - q1
        num["iqr_outliers"] = [int(((clean[c] < q1[c] - 1.5 * iqr[c]) | (clean[c] > q3[c] + 1.5 * iqr[c])).sum()) for c in num["column"]]
    cat = pd.concat([clean[c].value_counts(dropna=False).head(15).rename_axis("value").reset_index(name="count").assign(
        column=c, pct=lambda x: 100 * x["count"] / len(clean)) for c in categorical], ignore_index=True)[
        ["column", "value", "count", "pct"]] if categorical else pd.DataFrame()
    return {"profile": prof, "numeric_summary": num, "category_distribution": cat,
            "overview": {"raw_rows": len(raw), "raw_columns": raw.shape[1], "clean_rows": len(clean),
                         "duplicate_rows_raw": int(raw.duplicated().sum()), "missing_cells_raw": int(raw.isna().sum().sum()),
                         "missing_cells_clean": int(clean[numeric + categorical].isna().sum().sum()),
                         "n_numeric": len(numeric), "n_categorical": len(categorical)}}


# --------------------------------------------------------------------------------------- statistics
def statistics_generic(df: pd.DataFrame, numeric, categorical, outcome: str, alpha: float = 0.05, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for c in categorical:
        d = df[[c, Y]].dropna()
        tab = pd.crosstab(d[c], d[Y])
        if tab.shape[0] < 2 or tab.shape[1] < 2 or tab.shape[0] > 100:
            continue
        chi2, p, dof, exp = stats.chi2_contingency(tab)
        rates = (tab[1] / tab.sum(axis=1)).sort_values()
        v = cramers_v(tab)
        rows.append({"variable": c, "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": v, "effect_size": cramers_v_label(v),
                     "highest_group": rates.index[-1], "highest_rate": rates.iloc[-1], "lowest_group": rates.index[0],
                     "lowest_rate": rates.iloc[0]})
    chi = pd.DataFrame(rows)
    if len(chi):
        chi["p_adjusted"] = benjamini_hochberg(chi["p_value"])
        chi["significant"] = chi["p_adjusted"] < alpha
        chi["interpretation"] = [
            (f"{r.variable}: {outcome} rate differs across groups ({r.effect_size} association, V={r.cramers_v:.2f}); "
             f"'{r.highest_group}' {100 * r.highest_rate:.1f}% vs '{r.lowest_group}' {100 * r.lowest_rate:.1f}%.")
            if r.significant else f"{r.variable}: no significant difference (adjusted p={fmt_p(r.p_adjusted)})."
            for r in chi.itertuples()]
        chi = chi.sort_values("cramers_v", ascending=False).reset_index(drop=True)
    rows = []
    for c in numeric:
        d = df[[c, Y]].dropna()
        a, b = d.loc[d[Y] == 1, c].to_numpy(float), d.loc[d[Y] == 0, c].to_numpy(float)
        if len(a) < 5 or len(b) < 5 or np.ptp(d[c]) == 0:
            continue
        sh = [stats.shapiro(rng.choice(x, size=min(500, len(x)), replace=False))[1] if np.ptp(x) > 0 else 0.0 for x in (a, b)]
        normal = all(x > 0.05 for x in sh)
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        if normal:
            stat, p = stats.ttest_ind(a, b, equal_var=False)
            test, en, ev, el = "Welch t-test", "Cohen's d", cohens_d(a, b), cohens_d_label(cohens_d(a, b))
        else:
            stat, p = u[0], u[1]
            rb = rank_biserial(u[0], len(a), len(b))
            test, en, ev, el = "Mann-Whitney U", "rank-biserial r", rb, rank_biserial_label(rb)
        rows.append({"variable": c, "test": test, "statistic": float(stat), "p_value": float(p), "effect_size_name": en,
                     "effect_size": ev, "effect_magnitude": el, "median_positive": float(np.median(a)),
                     "median_negative": float(np.median(b)), "mean_positive": a.mean(), "mean_negative": b.mean()})
    num = pd.DataFrame(rows)
    if len(num):
        num["p_adjusted"] = benjamini_hochberg(num["p_value"])
        num["significant"] = num["p_adjusted"] < alpha
        num["interpretation"] = [
            (f"{r.variable}: higher for {outcome}" if r.median_positive > r.median_negative else f"{r.variable}: lower for {outcome}")
            + f" (median {r.median_positive:,.2f} vs {r.median_negative:,.2f}; {r.test}, {r.effect_magnitude} effect, p={fmt_p(r.p_adjusted)})."
            if r.significant else f"{r.variable}: no significant difference (adjusted p={fmt_p(r.p_adjusted)})." for r in num.itertuples()]
        num = num.reindex(num["effect_size"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    corr = df[numeric + [Y]].corr() if numeric else pd.DataFrame()
    return {"chi_square": chi, "numeric": num, "correlation": corr}


# --------------------------------------------------------------------------------------- modeling
def generic_preprocessor(numeric, categorical, cfg) -> ColumnTransformer:
    lo, hi = cfg["features"]["winsor_quantiles"]
    parts = []
    if numeric:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("winsor", Winsorizer(lo, hi)),
                                       ("scale", StandardScaler())]), numeric))
    if categorical:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                                                min_frequency=0.01))]), categorical))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=True)


def _png(path: Path) -> bytes:
    return Path(path).read_bytes()


def _plot_tradeoff(y, p, thr, path):
    import matplotlib.pyplot as plt
    from .utils import ACCENT, BLUE, save_figure
    ths = np.round(np.arange(0.05, 0.951, 0.01), 2)
    pr, rc, f1 = [], [], []
    for t in ths:
        m = compute_metrics(y, p, t)
        pr.append(m["precision"]); rc.append(m["recall"]); f1.append(m["f1"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ths, pr, color=BLUE, label="Precision"); ax.plot(ths, rc, color=ACCENT, label="Recall")
    ax.plot(ths, f1, color="#59A14F", label="F1"); ax.axvline(thr, color="black", ls="--", label=f"Chosen {thr:.2f}")
    ax.set(xlabel="Probability threshold", ylabel="Score", title="Precision / recall trade-off (test set)"); ax.legend()
    return save_figure(fig, path)


def run_generic_analysis(df_raw: pd.DataFrame, s: dict, base_cfg: dict | None = None, profile: str | None = None,
                         progress=None, source_name: str = "uploaded file") -> dict:
    """Full pipeline on any dataset. Returns a results dict (DataFrames, PNG bytes, a model bundle, a markdown report)."""
    base_cfg = base_cfg or load_config()
    cfg = _merge_over(base_cfg, TRAINING_PROFILES.get(profile or "", {}))
    seed = cfg["random_seed"]
    set_plot_style()
    outcome = f"{s['target']} = {s['positive']}"
    tmp = Path(tempfile.mkdtemp(prefix="generic_"))
    res: dict = {"settings": s, "source": source_name, "outcome": outcome, "created": now_iso(), "profile": profile}
    step = lambda name, i: progress(name, i, 6) if progress else None  # noqa: E731
    with PIPELINE_LOCK:
        # 1 ---- prepare
        step("prepare", 0)
        clog = CleaningLog()
        prep = clean_generic(df_raw, s, clog)
        clean, numeric, categorical = prep["clean"], prep["numeric"], prep["categorical"]
        feats = numeric + categorical
        if not feats:
            raise ValueError("No usable feature columns remain after cleaning.")
        res["quality"] = quality_tables(df_raw, clean, numeric, categorical)
        res["cleaning_log"] = clog.to_frame()
        res.update(numeric=numeric, categorical=categorical, features=feats)
        # sample for modeling / EDA
        if len(clean) > MAX_GENERIC_TRAIN:
            _, samp = train_test_split(clean, test_size=MAX_GENERIC_TRAIN, stratify=clean[Y], random_state=seed)
        else:
            samp = clean
        samp = samp.sort_index()
        res["eda_df"] = samp[feats + [Y]].reset_index(drop=True)
        res["base_rate"] = float(clean[Y].mean())
        # 2 ---- statistics
        step("stats", 1)
        stat_df = clean.sample(min(len(clean), 100_000), random_state=seed)
        res["stats"] = statistics_generic(stat_df, numeric, categorical, outcome, cfg["stats"]["alpha"], seed)
        # 3 ---- train
        step("train", 2)
        X, y = samp[feats], samp[Y]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg["split"]["test_size"], stratify=y, random_state=seed)
        pre = generic_preprocessor(numeric, categorical, cfg)
        cv = StratifiedKFold(n_splits=cfg["modeling"]["cv_folds"], shuffle=True, random_state=seed)
        searches, rows, imps = {}, [], {}
        for name, (est, grid, kind) in get_candidates(cfg, y_tr).items():
            search, row = _search(name, est, grid, kind, X_tr, y_tr, cv, cfg, preprocessor=clone(pre))
            searches[name], imps[name] = search, native_importance(search.best_estimator_)
            rows.append(row)
        comp = pd.DataFrame(rows)
        best_name, rationale = select_best(comp, cfg)
        best = searches[best_name].best_estimator_
        oof = cross_val_predict(clone(best), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=cfg["modeling"]["n_jobs"])[:, 1]
        # When the outcome is common (>=35%) a recall-heavy F-beta would flag almost everyone, so use balanced F1 there.
        beta = 1.0 if res["base_rate"] >= 0.35 else float(cfg["threshold"]["beta"])
        thr, thr_tbl = tune_threshold(y_tr, oof, beta)
        res["threshold_beta"] = beta
        if cfg["threshold"]["mode"] == "fixed":
            thr = float(cfg["threshold"]["value"])
        bundle = {"pipeline": best, "input_columns": feats, "threshold": thr, "model_name": DISPLAY.get(best_name, best_name),
                  "background": X_tr.sample(min(int(cfg["explain"]["background_size"]), len(X_tr)), random_state=seed).reset_index(drop=True)}
        comp_out = comp.copy()
        comp_out["selected"] = comp_out["model"] == best_name
        comp_out["best_params"] = comp_out["best_params"].astype(str)
        res.update(comparison=comp_out, model_name=bundle["model_name"], selection_rationale=rationale, threshold=thr,
                   bundle=bundle, native_importance=imps[best_name], n_train=len(X_tr), n_test=len(X_te), cfg=cfg)
        # 4 ---- evaluate
        step("evaluate", 3)
        p_te = predict_proba(bundle, X_te)
        m = compute_metrics(y_te, p_te, thr)
        res["metrics"], res["oof_metrics"] = m, compute_metrics(y_tr, oof, thr)
        figs = {"ROC curve": plot_roc(y_te, p_te, tmp / "roc.png"), "Precision-recall curve": plot_pr(y_te, p_te, thr, tmp / "pr.png"),
                "Confusion matrix": plot_confusion(m, tmp / "cm.png"),
                "Precision / recall trade-off": _plot_tradeoff(y_te, p_te, thr, tmp / "tradeoff.png"),
                "Score distribution": plot_probability_distribution(y_te, p_te, thr, tmp / "dist.png"),
                "Calibration": plot_calibration(y_te, p_te, tmp / "cal.png"),
                "Cumulative gains": plot_gains(gains_table(y_te, p_te), tmp / "gains.png")}
        res["figures"] = {k: _png(v) for k, v in figs.items()}
        res["test_predictions"] = pd.DataFrame({"actual": y_te.to_numpy(), "churn_probability": p_te})
        res["threshold_table"] = pd.DataFrame([compute_metrics(y_te, p_te, t) for t in np.round(np.arange(0.05, 0.951, 0.05), 2)]
                                              ).drop(columns=["classification_report", "confusion"])
        # 5 ---- explain
        step("explain", 4)
        groups = {c: [c] for c in feats}
        imp = permutation_importance_table(bundle, X_te, y_te, groups, repeats=cfg["explain"]["permutation_repeats"], seed=seed)
        res["importance"] = imp.rename(columns={"driver": "feature"})
        sample = X_te.sample(min(250, len(X_te)), random_state=seed).reset_index(drop=True)
        eff = marginal_effects(bundle, sample, groups)
        rows = []
        for c in feats:
            if c in categorical:
                key = sample[c].astype(str)
            else:
                try:
                    key = pd.qcut(sample[c], 4, duplicates="drop").astype(str)
                except ValueError:
                    key = sample[c].astype(str)
            t_ = pd.DataFrame({"value": key.to_numpy(), "effect_pp": 100 * eff[c].to_numpy()})
            for v, g in t_.groupby("value"):
                rows.append({"feature": c, "value": v, "customers": len(g), "mean_effect_pp": float(g["effect_pp"].mean())})
        res["value_effects"] = pd.DataFrame(rows)
        res["leakage_warning"] = bool(m["roc_auc"] > 0.98)
        # 6 ---- score everyone + report
        step("score", 5)
        score_df = clean if len(clean) <= MAX_GENERIC_SCORE else clean.sample(MAX_GENERIC_SCORE, random_state=seed)
        proba = np.concatenate([predict_proba(bundle, score_df[feats].iloc[i:i + 20_000])
                                for i in range(0, len(score_df), 20_000)])
        used = pd.Series("not used in training (unseen)", index=score_df.index)
        used.loc[X_tr.index.intersection(score_df.index)] = "training (out-of-fold score)"
        used.loc[X_te.index.intersection(score_df.index)] = "hold-out test"
        # training rows get out-of-fold probabilities instead of in-sample ones
        oof_s = pd.Series(oof, index=X_tr.index)
        common = oof_s.index.intersection(score_df.index)
        proba_s = pd.Series(proba, index=score_df.index)
        proba_s.loc[common] = oof_s.loc[common]
        tbl = score_df.copy()
        tbl["churn_probability"] = proba_s
        tbl["risk_tier"] = assign_risk_tier(tbl["churn_probability"].to_numpy(), thr, cfg["threshold"]["critical_probability"])
        tbl["flagged"] = tbl["churn_probability"] >= thr
        tbl["score_source"] = used
        vc = s.get("value_col")
        pr = tbl["churn_probability"]
        tbl["priority_score"] = (100 * (0.6 * pr + 0.4 * tbl[vc].rank(pct=True)) if vc else 100 * pr).round(2)
        tbl["priority_rank"] = tbl["priority_score"].rank(ascending=False, method="first").astype(int)
        tbl = tbl.sort_values("priority_rank")
        top = tbl.head(200)
        te = marginal_effects(bundle, top[feats].reset_index(drop=True), groups)
        drivers = []
        for i in range(len(top)):
            e = te.iloc[i].sort_values(ascending=False).head(3)
            drivers.append("; ".join(f"{c} = {top.iloc[i][c]:,.4g} ({100 * v:+.1f} pp)" if c in numeric and pd.notna(top.iloc[i][c])
                                     else f"{c} = {top.iloc[i][c]} ({100 * v:+.1f} pp)" for c, v in e.items() if v > 0))
        tbl["top_risk_drivers"] = ""
        tbl.loc[top.index, "top_risk_drivers"] = drivers
        keep = ([s["id_col"]] if s.get("id_col") else []) + ["churn_probability", "risk_tier", "flagged", "priority_score",
                                                              "priority_rank", "top_risk_drivers", "score_source"] + feats
        keep = list(dict.fromkeys(keep))
        res["scores"] = tbl[keep].reset_index(drop=True)
        if vc and vc not in keep:
            res["scores"][vc] = tbl[vc].to_numpy()
        fl = tbl[tbl["flagged"]]
        res["kpis"] = {"rows_scored": len(tbl), "flagged": len(fl), "flagged_share": len(fl) / max(len(tbl), 1),
                       "expected_positives": float(tbl["churn_probability"].sum()),
                       "critical": int((tbl["risk_tier"] == "Critical").sum()),
                       "value_at_risk": float(fl[vc].sum()) if vc else None,
                       "value_total": float(tbl[vc].sum()) if vc else None,
                       "expected_value_loss": float((tbl["churn_probability"] * tbl[vc]).sum()) if vc else None}
        res["report_md"] = build_report(res)
    return res


def explain_one(res: dict, row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    b = res["bundle"]
    feats = res["features"]
    eff = marginal_effects(b, row[feats].reset_index(drop=True), {c: [c] for c in feats}).iloc[0]
    out = pd.DataFrame({"feature": eff.index, "effect_pp": 100 * eff.values,
                        "value": [row.iloc[0][c] for c in eff.index]})
    out["explanation"] = [f"{r.feature} = {r.value} {'raises' if r.effect_pp > 0 else 'lowers'} the predicted probability by "
                          f"about {abs(r.effect_pp):.1f} percentage points compared with a typical customer." for r in out.itertuples()]
    return out.reindex(out["effect_pp"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)


def build_report(res: dict) -> str:
    m, k, q = res["metrics"], res["kpis"], res["quality"]["overview"]
    chi, num, imp = res["stats"]["chi_square"], res["stats"]["numeric"], res["importance"]
    L = [f"# Analysis report: {res['source']}", f"_Generated {res['created']} | outcome analysed: **{res['outcome']}**_", "",
         "## 1. Data", f"- {q['raw_rows']:,} rows x {q['raw_columns']} columns read; {q['clean_rows']:,} rows after cleaning; "
         f"{q['n_numeric']} numeric and {q['n_categorical']} categorical features used.",
         f"- Overall rate of the outcome ({res['outcome']}): **{100 * res['base_rate']:.1f}%**.",
         f"- Duplicate rows (raw): {q['duplicate_rows_raw']:,}; missing cells raw/clean: {q['missing_cells_raw']:,} / {q['missing_cells_clean']:,}.", "",
         "### Cleaning log", df_to_markdown(res["cleaning_log"]), "",
         "## 2. Statistical findings (association, not causation)"]
    L += [f"- {t}" for t in chi["interpretation"].head(6)] if len(chi) else []
    L += [f"- {t}" for t in num["interpretation"].head(6)] if len(num) else []
    L += ["", "## 3. Model", f"- **Selected:** {res['model_name']}. {res['selection_rationale']}",
          f"- Trained on {res['n_train']:,} rows; tested on {res['n_test']:,} untouched rows; decision threshold {res['threshold']:.2f}.", "",
          df_to_markdown(res["comparison"][["display_name", "cv_roc_auc", "cv_pr_auc", "cv_recall", "cv_precision", "cv_f1", "selected"]], 4), "",
          df_to_markdown(pd.DataFrame([{"Metric": n, "Value": m[n]} for n in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]), 4), "",
          "## 4. What drives the outcome (permutation importance on the test set)",
          df_to_markdown(imp[["feature", "importance_mean", "importance_std"]].head(10), 4), "",
          "## 5. Customers", f"- {k['rows_scored']:,} rows scored; **{k['flagged']:,} flagged ({100 * k['flagged_share']:.1f}%)**; "
          f"{k['critical']:,} critical-risk."]
    if k["value_at_risk"] is not None:
        L.append(f"- Value at risk (sum of '{res['settings']['value_col']}' for flagged rows): {k['value_at_risk']:,.0f} of {k['value_total']:,.0f} total.")
    if res["leakage_warning"]:
        L.append("- **Warning:** test ROC-AUC above 0.98 is unusually high. Check that no feature is a copy of the outcome (data leakage).")
    L += ["", "## 6. Caveats", "- Associations and model drivers are not proven causes; confirm with experiments.",
          "- Scores for training rows are out-of-fold; rows never used for training are scored by the final model.", ""]
    return "\n".join(L)
