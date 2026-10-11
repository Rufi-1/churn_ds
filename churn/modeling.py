"""Model development: stratified split, CV + hyperparameter search for four model families,
class-imbalance handling, threshold tuning on out-of-fold predictions, run registry and artifacts."""
from __future__ import annotations

import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from .config import load_config, resolve_path
from .data_loader import dataset_version, load_modeling
from .evaluation import compute_metrics
from .features import build_preprocessor, input_columns, pretty_feature_name
from .schema import ID_COL, TARGET_FLAG
from .utils import get_logger, now_iso, save_json

log = get_logger()
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1", "recall": "recall",
           "precision": "precision"}
DISPLAY = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
           "gradient_boosting": "Gradient Boosting", "xgboost": "XGBoost", "lightgbm": "LightGBM"}


def split_data(model_df: pd.DataFrame, cfg: dict):
    """Stratified train/test split. The test set is never used for fitting, tuning or selection."""
    cols = input_columns(cfg)
    X, y, ids = model_df[cols], model_df[TARGET_FLAG].astype(int), model_df[ID_COL]
    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, ids, test_size=cfg["split"]["test_size"], stratify=y, random_state=cfg["random_seed"])
    return X_tr, X_te, y_tr, y_te, id_tr, id_te


def get_candidates(cfg: dict, y_train: pd.Series) -> dict:
    """name -> (estimator, param_grid, search_type). Imbalance: class weights / scale_pos_weight;
    Gradient Boosting has no class weights, so it relies on threshold tuning."""
    seed, fast = cfg["random_seed"], cfg["modeling"]["fast"]
    pos_w = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    cands = {}
    wanted = cfg["modeling"]["models"]
    if "logistic_regression" in wanted:
        cands["logistic_regression"] = (
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed),
            {"clf__C": [0.1, 1.0] if fast else [0.01, 0.03, 0.1, 0.3, 1, 3]}, "grid")
    if "random_forest" in wanted:
        cands["random_forest"] = (
            RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=1),
            {"clf__n_estimators": [40] if fast else [200, 300], "clf__max_depth": [6, 10] if fast else [6, 8, 12],
             "clf__min_samples_leaf": [5, 10] if fast else [3, 5, 10],
             "clf__max_features": ["sqrt"] if fast else ["sqrt", 0.3]}, "random")
    if "gradient_boosting" in wanted:
        cands["gradient_boosting"] = (
            GradientBoostingClassifier(random_state=seed),
            {"clf__n_estimators": [40, 80] if fast else [100, 200, 300],
             "clf__learning_rate": [0.05, 0.1] if fast else [0.03, 0.05, 0.1],
             "clf__max_depth": [2, 3], "clf__subsample": [0.8, 1.0],
             "clf__min_samples_leaf": [10, 20]}, "random")
    if "xgboost" in wanted:
        try:
            from xgboost import XGBClassifier
            cands["xgboost"] = (
                XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                              scale_pos_weight=pos_w, random_state=seed, n_jobs=1, verbosity=0),
                {"clf__n_estimators": [60, 100] if fast else [200, 300, 400],
                 "clf__learning_rate": [0.05, 0.1] if fast else [0.02, 0.05, 0.1],
                 "clf__max_depth": [2, 3] if fast else [3, 4, 5], "clf__subsample": [0.8, 1.0],
                 "clf__colsample_bytree": [0.7, 1.0], "clf__min_child_weight": [1, 5],
                 "clf__reg_lambda": [1.0, 5.0]}, "random")
        except Exception as exc:  # noqa: BLE001
            try:
                from lightgbm import LGBMClassifier
                cands["lightgbm"] = (
                    LGBMClassifier(class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1),
                    {"clf__n_estimators": [60, 100] if fast else [200, 300],
                     "clf__learning_rate": [0.05, 0.1], "clf__num_leaves": [8, 16]}, "random")
            except Exception:  # noqa: BLE001
                log.warning("XGBoost/LightGBM unavailable (%s) - skipping boosted-tree model.", exc)
    return cands


def tune_threshold(y_true, proba, beta: float = 2.0) -> tuple[float, pd.DataFrame]:
    """Pick the probability threshold maximising F-beta (beta>1 favours recall)."""
    y, p = np.asarray(y_true).astype(int), np.asarray(proba, float)
    rows = []
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = p >= t
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fb = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec) if (beta ** 2 * prec + rec) else 0.0
        rows.append({"threshold": float(t), "precision": prec, "recall": rec, f"f{beta:g}": fb})
    tbl = pd.DataFrame(rows)
    best = tbl.loc[tbl[f"f{beta:g}"].idxmax()]
    return float(best["threshold"]), tbl


def native_importance(pipe: Pipeline) -> pd.DataFrame:
    names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        vals = clf.coef_.ravel()
        out = pd.DataFrame({"feature": names, "importance": np.abs(vals), "coefficient": vals})
    elif hasattr(clf, "feature_importances_"):
        out = pd.DataFrame({"feature": names, "importance": clf.feature_importances_})
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    out["label"] = [pretty_feature_name(n) for n in out["feature"]]
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def _search(name, est, grid, kind, X, y, cv, cfg, preprocessor=None):
    pipe = Pipeline([("prep", preprocessor if preprocessor is not None else build_preprocessor(cfg)), ("clf", est)])
    mcfg = cfg["modeling"]
    kw = dict(scoring=SCORING, refit=mcfg["tuning_metric"], cv=cv, n_jobs=mcfg["n_jobs"], return_train_score=False)
    if kind == "grid":
        search = GridSearchCV(pipe, grid, **kw)
    else:
        n_iter = 3 if mcfg["fast"] else int(mcfg["n_iter"])
        search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, random_state=cfg["random_seed"], **kw)
    t0 = time.time()
    search.fit(X, y)
    secs = time.time() - t0
    i = search.best_index_
    cvr = search.cv_results_
    row = {"model": name, "display_name": DISPLAY.get(name, name),
           **{f"cv_{m}": float(cvr[f"mean_test_{m}"][i]) for m in SCORING},
           **{f"cv_{m}_std": float(cvr[f"std_test_{m}"][i]) for m in SCORING},
           "best_params": {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                           for k, v in search.best_params_.items()}, "fit_seconds": round(secs, 1)}
    return search, row


def select_best(comp: pd.DataFrame, cfg: dict) -> tuple[str, str]:
    metric = f"cv_{cfg['modeling']['tuning_metric']}"
    best = comp.sort_values(metric, ascending=False).iloc[0]
    tol = cfg["modeling"]["parsimony_tolerance"]
    lr = comp[comp["model"] == "logistic_regression"]
    chosen, why = best, (f"{best['display_name']} had the highest cross-validated {metric[3:].upper()} "
                         f"({best[metric]:.4f}). ROC-AUC is threshold-independent and robust to class imbalance; "
                         f"PR-AUC, recall, precision and F1 are reported alongside it.")
    if len(lr) and best["model"] != "logistic_regression" and lr.iloc[0][metric] >= best[metric] - tol:
        chosen = lr.iloc[0]
        why = (f"Logistic Regression ({lr.iloc[0][metric]:.4f}) was within {tol} of the best model "
               f"{best['display_name']} ({best[metric]:.4f}) on cross-validated {metric[3:].upper()}; the simpler, "
               f"more interpretable model was preferred (parsimony rule).")
    return str(chosen["model"]), why


def train_and_save(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    mdir = resolve_path(cfg, "models_dir", create=True)
    tdir = resolve_path(cfg, "tables_dir", create=True)
    model_df = load_modeling(cfg)
    X_tr, X_te, y_tr, y_te, id_tr, id_te = split_data(model_df, cfg)
    log.info("Split: train=%d test=%d | churn rate train %.3f test %.3f", len(X_tr), len(X_te), y_tr.mean(), y_te.mean())
    save_json({"train_ids": id_tr.tolist(), "test_ids": id_te.tolist(), "test_size": cfg["split"]["test_size"],
               "random_seed": cfg["random_seed"]}, mdir / "split.json")

    cv = StratifiedKFold(n_splits=cfg["modeling"]["cv_folds"], shuffle=True, random_state=cfg["random_seed"])
    searches, rows, importances = {}, [], {}
    for name, (est, grid, kind) in get_candidates(cfg, y_tr).items():
        log.info("Tuning %s (%s search, %d-fold CV)...", DISPLAY.get(name, name), kind, cfg["modeling"]["cv_folds"])
        search, row = _search(name, est, grid, kind, X_tr, y_tr, cv, cfg)
        searches[name] = search
        rows.append(row)
        importances[name] = native_importance(search.best_estimator_)
        log.info("  %s: CV ROC-AUC %.4f | PR-AUC %.4f | recall %.3f | F1 %.3f (%.0fs)", row["display_name"],
                 row["cv_roc_auc"], row["cv_pr_auc"], row["cv_recall"], row["cv_f1"], row["fit_seconds"])
    comp = pd.DataFrame(rows)
    best_name, rationale = select_best(comp, cfg)
    best_pipe = searches[best_name].best_estimator_
    log.info("Selected model: %s. %s", DISPLAY.get(best_name, best_name), rationale)

    # out-of-fold probabilities on TRAIN only -> threshold tuning + honest scores for train customers
    oof = cross_val_predict(clone(best_pipe), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=cfg["modeling"]["n_jobs"])[:, 1]
    beta = float(cfg["threshold"]["beta"])
    tuned_thr, thr_tbl = tune_threshold(y_tr, oof, beta)
    thr = float(cfg["threshold"]["value"]) if cfg["threshold"]["mode"] == "fixed" else tuned_thr
    log.info("Threshold: tuned=%.2f (F%g on out-of-fold train predictions); using %.2f", tuned_thr, beta, thr)

    p_te = best_pipe.predict_proba(X_te)[:, 1]
    test_metrics = compute_metrics(y_te, p_te, thr)
    oof_metrics = compute_metrics(y_tr, oof, thr)

    scores = pd.concat([
        pd.DataFrame({ID_COL: id_tr.values, "churn_probability": oof, "score_source": "out-of-fold (train)",
                      "actual_churn": y_tr.values}),
        pd.DataFrame({ID_COL: id_te.values, "churn_probability": p_te, "score_source": "hold-out (test)",
                      "actual_churn": y_te.values})], ignore_index=True)
    scores.to_csv(tdir / "customer_scores.csv", index=False)
    thr_tbl.to_csv(tdir / "threshold_tuning_oof.csv", index=False)

    version = dataset_version(cfg)
    trained_at = now_iso()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    pre = best_pipe.named_steps["prep"]
    feat_names = [str(n) for n in pre.get_feature_names_out()]
    bg = X_tr.sample(min(int(cfg["explain"]["background_size"]), len(X_tr)), random_state=cfg["random_seed"]).reset_index(drop=True)
    lib = {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
    try:
        import xgboost
        lib["xgboost"] = xgboost.__version__
    except Exception:  # noqa: BLE001
        pass
    bundle = {"model_name": DISPLAY.get(best_name, best_name), "model_key": best_name, "pipeline": best_pipe,
              "preprocessor": pre, "feature_names": feat_names, "input_columns": list(X_tr.columns),
              "threshold": thr, "tuned_threshold": tuned_thr, "threshold_beta": beta,
              "metrics": test_metrics, "oof_metrics": oof_metrics,
              "cv_summary": comp[comp["model"] == best_name].iloc[0].to_dict(),
              "params": comp[comp["model"] == best_name].iloc[0]["best_params"], "trained_at": trained_at,
              "dataset_version": version, "background": bg, "selection_rationale": rationale,
              "library_versions": lib, "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "run_id": run_id}
    joblib.dump(bundle, mdir / "churn_model.joblib", compress=3)
    joblib.dump(pre, mdir / "preprocessor.joblib", compress=3)
    save_json(feat_names, mdir / "feature_names.json")
    save_json({"model": bundle["model_name"], "threshold": thr, "test": test_metrics, "train_oof": oof_metrics,
               "dataset_version": version, "trained_at": trained_at}, mdir / "metrics.json")

    # ---- run registry: every model run with params, date, dataset version, metrics, importance ----
    reg_rows = []
    for r in rows:
        sel = r["model"] == best_name
        reg_rows.append({"run_id": run_id, "model": r["display_name"], "params": str(r["best_params"]),
                         "trained_at": trained_at, "dataset_version": version, "n_train": len(X_tr),
                         **{k: r[k] for k in r if k.startswith("cv_")}, "fit_seconds": r["fit_seconds"],
                         "selected": sel, "threshold": thr if sel else np.nan,
                         **({f"test_{k}": test_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
                            if sel else {})})
        imp = importances[r["model"]].head(25)
        save_json({"run_id": run_id, "model": r["display_name"], "params": r["best_params"],
                   "trained_at": trained_at, "dataset_version": version, "selected": sel,
                   "cv_metrics": {k: r[k] for k in r if k.startswith("cv_")},
                   "test_metrics": {k: test_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]} if sel else None,
                   "feature_importance": imp.to_dict("records")},
                  mdir / "runs" / f"{run_id}_{r['model']}.json")
    reg_path = mdir / "model_runs.csv"
    reg_new = pd.DataFrame(reg_rows)
    if reg_path.exists():
        try:
            reg_new = pd.concat([pd.read_csv(reg_path), reg_new], ignore_index=True)
        except Exception:  # noqa: BLE001
            pass
    reg_new.to_csv(reg_path, index=False)

    comp_out = comp.copy()
    comp_out["best_params"] = comp_out["best_params"].astype(str)
    comp_out["selected"] = comp_out["model"] == best_name
    comp_out.to_csv(tdir / "model_comparison.csv", index=False)
    importances[best_name].to_csv(tdir / "feature_importance_native.csv", index=False)
    log.info("TEST (untouched) @thr %.2f: recall %.3f precision %.3f f1 %.3f roc_auc %.3f", thr, test_metrics["recall"],
             test_metrics["precision"], test_metrics["f1"], test_metrics["roc_auc"])
    return {"bundle": bundle, "comparison": comp, "metrics": test_metrics, "best": best_name, "run_id": run_id}
