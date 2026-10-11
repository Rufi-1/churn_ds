"""Explainability: grouped/feature permutation importance, SHAP (optional), and a model-agnostic
local explanation (marginal effect vs the training population). Every statement is derived from
the fitted model's own predictions - nothing is hand-written about which drivers 'should' matter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

from .config import load_config, resolve_path
from .data_loader import load_modeling
from .features import pretty_feature_name, source_column
from .predict import load_bundle, predict_proba
from .schema import DRIVER_GROUPS, FEATURE_LABELS, ID_COL, TARGET_FLAG, column_label
from .utils import ACCENT, BLUE, df_to_markdown, get_logger, load_json, now_iso, save_figure, save_json, set_plot_style

log = get_logger()
RAW_DISPLAY = {"tenure", "MonthlyCharges", "TotalCharges", "gender", "SeniorCitizen", "Partner", "Dependents",
               "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
               "PaperlessBilling", "PaymentMethod"}
PRIMARY_COLUMN = {"Contract type": "Contract", "Tenure / customer age": "tenure_group",
                  "Monthly price level": "customer_value_segment", "Payment & billing": "PaymentMethod",
                  "Internet service type": "InternetService", "Security & tech support": "TechSupport",
                  "Streaming services": "StreamingTV", "Service mix": "service_profile",
                  "Household / demographics": "SeniorCitizen"}


def active_groups(bundle: dict) -> dict[str, list[str]]:
    cols = set(bundle["input_columns"])
    g = {k: [c for c in v if c in cols] for k, v in DRIVER_GROUPS.items()}
    return {k: v for k, v in g.items() if v}


def permutation_importance_table(bundle: dict, X: pd.DataFrame, y, groups: dict[str, list[str]],
                                 repeats: int = 10, seed: int = 42) -> pd.DataFrame:
    """Drop in ROC-AUC when a set of columns is shuffled jointly (held-out data)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base = roc_auc_score(y, predict_proba(bundle, X))
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            for c in cols:
                Xp[c] = X[c].to_numpy()[perm]
            drops.append(base - roc_auc_score(y, predict_proba(bundle, Xp)))
        rows.append({"driver": name, "columns": ", ".join(cols), "importance_mean": float(np.mean(drops)),
                     "importance_std": float(np.std(drops)), "baseline_auc": base})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def marginal_effects(bundle: dict, X: pd.DataFrame, groups: dict[str, list[str]] | None = None,
                     background: pd.DataFrame | None = None) -> pd.DataFrame:
    """effect[i, g] = P(churn | customer i) - mean P(churn | customer i with group g's columns replaced
    by those of random training customers). Positive = this customer's values push risk up vs a typical one."""
    groups = groups or active_groups(bundle)
    bg = (background if background is not None else bundle["background"]).reset_index(drop=True)
    n, B = len(X), len(bg)
    X = X[bundle["input_columns"]].reset_index(drop=True)
    p0 = predict_proba(bundle, X)
    out = pd.DataFrame(index=range(n))
    rep = X.loc[np.repeat(np.arange(n), B)].reset_index(drop=True)
    for name, cols in groups.items():
        Xr = rep.copy()
        for c in cols:
            Xr[c] = np.tile(bg[c].to_numpy(), n)
        pm = predict_proba(bundle, Xr).reshape(n, B).mean(axis=1)
        out[name] = p0 - pm
    out.index = X.index
    return out


def describe_group_value(row: pd.Series, cols: list[str]) -> str:
    parts = []
    for c in cols:
        if c in RAW_DISPLAY and c in row.index:
            v = row[c]
            v = f"{v:,.1f}" if isinstance(v, (float, np.floating)) else str(v)
            parts.append(f"{column_label(c)} = {v}")
    return "; ".join(parts[:3]) or ", ".join(cols[:2])


def explain_customer(bundle: dict, X_row: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Local, business-language explanation for ONE customer (model-agnostic marginal effects)."""
    groups = active_groups(bundle)
    eff = marginal_effects(bundle, X_row.iloc[[0]], groups).iloc[0]
    row = X_row.iloc[0]
    out = pd.DataFrame({"driver": eff.index, "effect_pp": 100 * eff.values,
                        "customer_values": [describe_group_value(row, groups[g]) for g in eff.index]})
    out["direction"] = np.where(out["effect_pp"] > 0, "raises risk", "lowers risk")
    out["explanation"] = [f"{r.driver} ({r.customer_values}) {r.direction.replace('risk', 'predicted churn risk')} "
                          f"by about {abs(r.effect_pp):.1f} percentage points compared with a typical customer."
                          for r in out.itertuples()]
    return out.reindex(out["effect_pp"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)


def shap_values_for(bundle: dict, X: pd.DataFrame, background: pd.DataFrame):
    """(shap_values[n, features], transformed X) or None if SHAP is unavailable/incompatible."""
    try:
        import shap
        pipe = bundle["pipeline"]
        pre, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
        Xt = pre.transform(X[bundle["input_columns"]])
        if hasattr(clf, "coef_"):
            sv = shap.LinearExplainer(clf, pre.transform(background[bundle["input_columns"]])).shap_values(Xt)
        else:
            sv = shap.TreeExplainer(clf).shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        return sv, Xt
    except Exception as exc:  # noqa: BLE001
        log.warning("SHAP unavailable (%s); using permutation importance and marginal effects only.", exc)
        return None


def shap_local(bundle: dict, X_row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame | None:
    res = shap_values_for(bundle, X_row.iloc[[0]], bundle["background"])
    if res is None:
        return None
    sv, _ = res
    df = pd.DataFrame({"feature": [pretty_feature_name(n) for n in bundle["feature_names"]], "shap_value": sv[0]})
    return df.reindex(df["shap_value"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)


def _plot_bar(df, x, y, title, path, xlabel, color=BLUE, top=15):
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 1.8))
    ax.barh(d[y], d[x], color=color)
    ax.set(title=title, xlabel=xlabel)
    return save_figure(fig, path)


def run_explainability(cfg: dict | None = None, save: bool = True) -> dict:
    cfg = cfg or load_config()
    set_plot_style()
    bundle = load_bundle(cfg)
    ecfg = cfg["explain"]
    split = load_json(resolve_path(cfg, "models_dir") / "split.json")
    data = load_modeling(cfg).set_index(ID_COL)
    test = data.loc[split["test_ids"]]
    X, y = test[bundle["input_columns"]], test[TARGET_FLAG].to_numpy()
    groups = active_groups(bundle)
    seed = cfg["random_seed"]

    grp = permutation_importance_table(bundle, X, y, groups, ecfg["permutation_repeats"], seed)
    feat = permutation_importance_table(bundle, X, y, {c: [c] for c in bundle["input_columns"]},
                                        max(3, ecfg["permutation_repeats"] // 2), seed)
    feat["label"] = [column_label(c) for c in feat["driver"]]
    sample = X.sample(min(250, len(X)), random_state=seed)
    eff = marginal_effects(bundle, sample, groups)
    rows = []
    for g, col in PRIMARY_COLUMN.items():
        if g in eff.columns and col in sample.columns:
            t = pd.DataFrame({"value": sample[col].astype(str).to_numpy(), "effect_pp": 100 * eff[g].to_numpy()})
            for v, d in t.groupby("value"):
                rows.append({"driver": g, "column": col, "value": v, "customers": len(d),
                             "mean_effect_pp": float(d["effect_pp"].mean())})
    value_effects = pd.DataFrame(rows).sort_values("mean_effect_pp", ascending=False)
    summary = pd.DataFrame({"driver": eff.columns, "mean_abs_effect_pp": 100 * eff.abs().mean().to_numpy()})
    grp = grp.merge(summary, on="driver", how="left")
    native = pd.read_csv(resolve_path(cfg, "tables_dir") / "feature_importance_native.csv") \
        if (resolve_path(cfg, "tables_dir") / "feature_importance_native.csv").exists() else pd.DataFrame()

    shap_df, shap_ok = None, False
    if ecfg.get("use_shap", True):
        res = shap_values_for(bundle, X.sample(min(int(ecfg["shap_sample_size"]), len(X)), random_state=seed),
                              bundle["background"])
        if res is not None:
            sv, Xt = res
            names = [pretty_feature_name(n) for n in bundle["feature_names"]]
            shap_df = pd.DataFrame({"feature": names, "source_column": [source_column(n, bundle["input_columns"])
                                                                       for n in bundle["feature_names"]],
                                    "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
            shap_ok = True
            if save:
                fdir = resolve_path(cfg, "figures_dir", create=True)
                try:
                    import shap
                    plt.figure(figsize=(9, 7))
                    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=15)
                    plt.title("SHAP summary (model-output impact; red = high feature value)")
                    save_figure(plt.gcf(), fdir / "explain_shap_summary.png")
                except Exception as exc:  # noqa: BLE001
                    plt.close("all")
                    log.warning("SHAP beeswarm skipped (%s)", exc)

    result = {"group_importance": grp, "feature_importance": feat, "value_effects": value_effects,
              "native": native, "shap": shap_df, "shap_available": shap_ok}
    if save:
        fdir = resolve_path(cfg, "figures_dir", create=True)
        tdir = resolve_path(cfg, "tables_dir", create=True)
        rdir = resolve_path(cfg, "reports_dir", create=True)
        grp.to_csv(tdir / "explain_group_importance.csv", index=False)
        feat.to_csv(tdir / "explain_feature_importance.csv", index=False)
        value_effects.to_csv(tdir / "explain_value_effects.csv", index=False)
        if shap_df is not None:
            shap_df.to_csv(tdir / "explain_shap_global.csv", index=False)
        _plot_bar(grp, "importance_mean", "driver", "Business drivers: permutation importance (drop in test ROC-AUC)",
                  fdir / "explain_group_importance.png", "Drop in ROC-AUC when shuffled", ACCENT)
        _plot_bar(feat, "importance_mean", "label", "Feature-level permutation importance (test set)",
                  fdir / "explain_feature_importance.png", "Drop in ROC-AUC when shuffled", BLUE, 20)
        if len(native):
            _plot_bar(native, "importance", "label", f"Model-native importance ({bundle['model_name']})",
                      fdir / "explain_native_importance.png",
                      "|coefficient| (standardised)" if "coefficient" in native else "Importance", BLUE, 20)
        if shap_df is not None:
            _plot_bar(shap_df, "mean_abs_shap", "feature", "Mean |SHAP value| (global importance)",
                      fdir / "explain_shap_bar.png", "mean |SHAP|", "#59A14F", 20)
        (rdir / "explainability_report.md").write_text(explain_markdown(bundle, result), encoding="utf-8")
        save_json({"generated_at": now_iso(), "shap_available": shap_ok,
                   "top_drivers": grp["driver"].head(5).tolist()}, tdir / "explain_meta.json")
        log.info("Explainability: top driver = %s (SHAP %s)", grp["driver"].iloc[0], "on" if shap_ok else "off")
    return result


def explain_markdown(bundle: dict, r: dict) -> str:
    g, v = r["group_importance"], r["value_effects"]
    lines = ["# Model Explainability", "",
             f"Model: **{bundle['model_name']}**. All numbers below are computed from this model on the held-out "
             "test set; they describe what the *model* relies on, which reflects patterns in the data (association) "
             "and not proven causes.", "",
             "## Global drivers (grouped permutation importance)",
             "Importance = how much test ROC-AUC drops when a driver's columns are shuffled. Related columns "
             "(e.g. contract type + month-to-month flag) are shuffled together so credit is not split.", "",
             df_to_markdown(g[["driver", "importance_mean", "importance_std", "mean_abs_effect_pp"]], 4), "",
             "## What pushes risk up or down (average model effect, percentage points vs a typical customer)", ""]
    for drv in g["driver"].head(5):
        sub = v[v["driver"] == drv].sort_values("mean_effect_pp", ascending=False)
        if len(sub):
            lines.append(f"- **{drv}**: " + "; ".join(f"{r.value} ({r.mean_effect_pp:+.1f} pp, n={r.customers})"
                                                       for r in sub.itertuples()))
    lines += ["", "## How to read a single customer's explanation",
              "Each driver's effect is the customer's predicted churn probability minus the average probability "
              "obtained when that driver's values are replaced by those of random training customers. Positive = "
              "pushes the customer above a typical customer's risk. Effects do not add up exactly (interactions).", ""]
    if r["shap"] is not None:
        lines += ["## SHAP global importance (top 15)", df_to_markdown(r["shap"].head(15), 4), ""]
    else:
        lines += ["_SHAP is not installed/compatible here; permutation importance is used instead._", ""]
    return "\n".join(lines)
