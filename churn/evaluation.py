"""Model evaluation on the untouched test set: metrics, curves, thresholds, business trade-offs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from .config import load_config, resolve_path
from .data_loader import load_modeling
from .predict import load_bundle, predict_proba, resolve_threshold
from .schema import ID_COL, TARGET_FLAG
from .utils import ACCENT, BLUE, df_to_markdown, get_logger, load_json, now_iso, save_figure, save_json, set_plot_style

log = get_logger()


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "n": int(len(y)), "positives": int(y.sum()),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "brier_score": float(brier_score_loss(y, p)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "classification_report": classification_report(y, pred, target_names=["Retained", "Churned"],
                                                           output_dict=True, zero_division=0)}


def threshold_table(y_true, proba, monthly_charges, cfg: dict, grid=None) -> pd.DataFrame:
    """Precision / recall / business net benefit at many thresholds."""
    b = cfg["business"]
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, float)
    m = np.asarray(monthly_charges, float)
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2) if grid is None else grid
    rows = []
    for t in grid:
        flag = p >= t
        tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum()); fn = int((~flag & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        protected = float(m[flag & (y == 1)].sum() * 12 * b["save_rate"])
        cost = int(flag.sum()) * b["retention_offer_cost"]
        rows.append({"threshold": float(t), "customers_flagged": int(flag.sum()), "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1,
                     "annual_revenue_protected": protected, "campaign_cost": cost,
                     "net_benefit": protected - cost})
    return pd.DataFrame(rows)


def gains_table(y_true, proba, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(proba, float)}).sort_values("p", ascending=False)
    d["decile"] = pd.qcut(np.arange(len(d)), bins, labels=range(1, bins + 1))
    g = d.groupby("decile", observed=True).agg(customers=("y", "size"), churners=("y", "sum"),
                                              avg_probability=("p", "mean")).reset_index()
    g["cum_capture_rate"] = g["churners"].cumsum() / max(d["y"].sum(), 1)
    g["lift"] = (g["churners"] / g["customers"]) / max(d["y"].mean(), 1e-9)
    return g


def plot_roc(y, p, path):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"Model (AUC = {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve (test set)")
    ax.legend(loc="lower right")
    return save_figure(fig, path)


def plot_pr(y, p, thr, path):
    prec, rec, ths = precision_recall_curve(y, p)
    base = float(np.mean(y))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(rec, prec, color=BLUE, lw=2, label=f"Model (AP = {average_precision_score(y, p):.3f})")
    ax.axhline(base, ls="--", color="grey", label=f"Baseline (churn rate {100 * base:.1f}%)")
    pred = (np.asarray(p) >= thr)
    ax.scatter([recall_score(y, pred)], [precision_score(y, pred, zero_division=0)], color=ACCENT, s=80, zorder=5,
               label=f"Chosen threshold {thr:.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test set)")
    ax.legend(loc="upper right")
    return save_figure(fig, path)


def plot_confusion(m: dict, path):
    c = m["confusion"]
    arr = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(arr, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{arr[i, j]:,}\n({100 * arr[i, j] / arr.sum():.1f}%)", ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Predicted retained", "Predicted churn"])
    ax.set_yticks([0, 1], ["Actually retained", "Actually churned"])
    ax.set_title(f"Confusion matrix @ threshold {m['threshold']:.2f}")
    ax.grid(False)
    return save_figure(fig, path)


def plot_threshold_tradeoff(tt: pd.DataFrame, thr, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for c, col in [("precision", BLUE), ("recall", ACCENT), ("f1", "#59A14F")]:
        axes[0].plot(tt["threshold"], tt[c], label=c.capitalize(), color=col, lw=2)
    axes[0].axvline(thr, color="black", ls="--", label=f"Chosen {thr:.2f}")
    axes[0].set(xlabel="Probability threshold", ylabel="Score", title="Precision / recall trade-off")
    axes[0].legend()
    axes[1].plot(tt["threshold"], tt["net_benefit"], color="#59A14F", lw=2)
    axes[1].axvline(thr, color="black", ls="--")
    axes[1].set(xlabel="Probability threshold", ylabel="Net benefit (annual, currency)",
                title="Illustrative campaign net benefit")
    return save_figure(fig, path)


def plot_probability_distribution(y, p, thr, path):
    d = pd.DataFrame({"p": p, "Outcome": np.where(np.asarray(y) == 1, "Churned", "Retained")})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, col in [("Retained", BLUE), ("Churned", ACCENT)]:
        ax.hist(d.loc[d.Outcome == k, "p"], bins=30, alpha=.6, color=col, label=k)
    ax.axvline(thr, color="black", ls="--", label=f"Threshold {thr:.2f}")
    ax.set(xlabel="Predicted churn probability", ylabel="Customers", title="Score distribution by actual outcome")
    ax.legend()
    return save_figure(fig, path)


def plot_calibration(y, p, path):
    frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(mean_p, frac, "o-", color=ACCENT, label="Model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate", title="Calibration (test set)")
    ax.legend()
    return save_figure(fig, path)


def plot_gains(g: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    x = np.r_[0, g["decile"].astype(int) / g["decile"].astype(int).max()]
    ax.plot(x, np.r_[0, g["cum_capture_rate"]], "o-", color=ACCENT, label="Model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random targeting")
    ax.set(xlabel="Share of customers contacted (highest risk first)", ylabel="Share of churners captured",
           title="Cumulative gains")
    ax.legend()
    return save_figure(fig, path)


def _test_frame(cfg: dict, bundle: dict):
    split = load_json(resolve_path(cfg, "models_dir") / "split.json")
    data = load_modeling(cfg).set_index(ID_COL)
    test = data.loc[split["test_ids"]]
    return test[bundle["input_columns"]], test[TARGET_FLAG].to_numpy(), test["MonthlyCharges"].to_numpy(), test.index


TRADEOFF_TEXT = (
    "**Precision vs recall for churn.** Recall is the share of real churners we catch; precision is the share of "
    "flagged customers who really would have left. Lowering the probability threshold flags more customers: recall "
    "rises (fewer missed churners, i.e. less lost revenue) but precision falls (more retention offers wasted on "
    "customers who would have stayed). Raising it does the opposite. Because a lost customer typically costs far "
    "more than one retention offer, the default threshold is tuned with an F-beta score that weights recall more than "
    "precision. The 'net benefit' column/curve uses the offer cost and save-rate assumptions in `config.yaml`; "
    "change `threshold.value`/`threshold.mode` there (or use the slider in the dashboard) to see the effect.")


def evaluate_model(cfg: dict | None = None, threshold: float | None = None, save: bool = True) -> dict:
    cfg = cfg or load_config()
    set_plot_style()
    bundle = load_bundle(cfg)
    thr = float(threshold) if threshold is not None else resolve_threshold(cfg, bundle)
    X, y, monthly, ids = _test_frame(cfg, bundle)
    p = predict_proba(bundle, X)
    metrics = compute_metrics(y, p, thr)
    tt = threshold_table(y, p, monthly, cfg)
    gains = gains_table(y, p)
    result = {"metrics": metrics, "threshold_table": tt, "gains": gains, "proba": p, "y": y, "ids": list(ids)}
    if not save:
        return result
    fdir = resolve_path(cfg, "figures_dir", create=True)
    tdir = resolve_path(cfg, "tables_dir", create=True)
    rdir = resolve_path(cfg, "reports_dir", create=True)
    mdir = resolve_path(cfg, "models_dir", create=True)
    figs = [plot_roc(y, p, fdir / "eval_roc_curve.png"), plot_pr(y, p, thr, fdir / "eval_precision_recall_curve.png"),
            plot_confusion(metrics, fdir / "eval_confusion_matrix.png"),
            plot_threshold_tradeoff(tt, thr, fdir / "eval_threshold_tradeoff.png"),
            plot_probability_distribution(y, p, thr, fdir / "eval_score_distribution.png"),
            plot_calibration(y, p, fdir / "eval_calibration.png"), plot_gains(gains, fdir / "eval_cumulative_gains.png")]
    tt.to_csv(tdir / "eval_threshold_table.csv", index=False)
    gains.to_csv(tdir / "eval_gains_table.csv", index=False)
    pd.DataFrame({ID_COL: ids, "actual_churn": y, "churn_probability": p,
                  "predicted_churn": (p >= thr).astype(int)}).to_csv(tdir / "test_predictions.csv", index=False)
    cr = pd.DataFrame(metrics["classification_report"]).T.reset_index().rename(columns={"index": "class"})
    cr.to_csv(tdir / "eval_classification_report.csv", index=False)
    save_json({**{k: v for k, v in metrics.items()}, "evaluated_at": now_iso(), "model": bundle["model_name"]},
              mdir / "metrics_test.json")
    comp_path = tdir / "model_comparison.csv"
    comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    (rdir / "model_evaluation_report.md").write_text(_eval_markdown(bundle, metrics, tt, cr, comp), encoding="utf-8")
    log.info("Evaluation: recall=%.3f precision=%.3f f1=%.3f roc_auc=%.3f @thr %.2f", metrics["recall"],
             metrics["precision"], metrics["f1"], metrics["roc_auc"], thr)
    result["figures"] = [str(f) for f in figs]
    return result


def _eval_markdown(bundle, m, tt, cr, comp) -> str:
    c = m["confusion"]
    summary = pd.DataFrame([{"Metric": k, "Value": m[k]} for k in
                            ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy",
                             "specificity", "brier_score"]])
    lines = [
        "# Model Evaluation Report", "",
        f"- **Final model:** {bundle['model_name']}  |  **Trained:** {bundle['trained_at']}  |  "
        f"**Dataset version:** `{bundle['dataset_version']}`",
        f"- **Evaluated on the untouched test set:** {m['n']:,} customers ({m['positives']:,} churners)",
        f"- **Decision threshold:** {m['threshold']:.2f}", "",
        "## Why this model was selected", bundle.get("selection_rationale", ""), "",
        "## Cross-validated model comparison (training set only)", df_to_markdown(comp, 4), "",
        "## Test-set metrics", df_to_markdown(summary, 4), "",
        "## Confusion matrix",
        df_to_markdown(pd.DataFrame({"": ["Actually retained", "Actually churned"],
                                     "Predicted retained": [c["tn"], c["fn"]],
                                     "Predicted churn": [c["fp"], c["tp"]]})), "",
        f"Of {m['positives']} real churners the model catches {c['tp']} (recall {100 * m['recall']:.1f}%); "
        f"{c['fn']} are missed. Of {c['tp'] + c['fp']} customers flagged, {c['tp']} truly churn "
        f"(precision {100 * m['precision']:.1f}%).", "",
        "## Classification report", df_to_markdown(cr, 3), "",
        "## Precision-recall trade-off and threshold choice", TRADEOFF_TEXT, "",
        df_to_markdown(tt[["threshold", "customers_flagged", "precision", "recall", "f1", "annual_revenue_protected",
                           "campaign_cost", "net_benefit"]], 3), "",
        "## Charts", "- `outputs/figures/eval_roc_curve.png`", "- `outputs/figures/eval_precision_recall_curve.png`",
        "- `outputs/figures/eval_confusion_matrix.png`", "- `outputs/figures/eval_threshold_tradeoff.png`",
        "- `outputs/figures/eval_score_distribution.png`", "- `outputs/figures/eval_calibration.png`",
        "- `outputs/figures/eval_cumulative_gains.png`", ""]
    return "\n".join(lines)
