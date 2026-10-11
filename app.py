"""Streamlit dashboard for churn_ds.  Run:  streamlit run app.py"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import streamlit as st

from churn.cleaning import standardize_frame
from churn.config import load_config, resolve_path
from churn.dashboard_helpers import (PAGE_SIZES, RISK_COLORS, append_history, artifacts_ready, clear_history,
                                     filter_customers, format_currency, friendly_columns, kpi_summary, load_history,
                                     metrics_at_threshold, recommend_for_frame, risk_label, scenario_net_benefit,
                                     search_sort_paginate, to_csv_bytes)
from churn.eda import churn_rate_by
from churn.explain import explain_customer, shap_local
from churn.features import engineer_features
from churn.predict import (DEFAULT_CUSTOMER, load_bundle, make_customer_frame, prepare_features, predict_proba,
                           resolve_threshold)
from churn.schema import ALLOWED_VALUES, ID_COL, TARGET_FLAG, column_label
from churn import generic as gen
from churn.evaluation import compute_metrics
from churn.schema import RAW_COLUMNS
from churn.upload import (MAX_SCORE_ROWS, MAX_TRAIN_ROWS, TRAINING_PROFILES, auto_map_columns, create_workspace,
                          prepare_upload, run_workspace_pipeline, score_batch)
Y_COL = gen.Y
from churn.utils import ACCENT, BLUE, set_plot_style

st.set_page_config(page_title="Churn Intelligence Dashboard", page_icon="📉", layout="wide")
set_plot_style()
BASE_CFG = load_config()
CFG = st.session_state.get("workspace_cfg") or BASE_CFG   # per-session private workspace after an upload
CUR = CFG["business"]["currency"]
TABLES = resolve_path(CFG, "tables_dir")
FIGS = resolve_path(CFG, "figures_dir")
REPORTS = resolve_path(CFG, "reports_dir")
MODELS = resolve_path(CFG, "models_dir")
HISTORY = resolve_path(CFG, "history_file")

st.markdown("""
<style>
[data-testid="stMetric"]{background:rgba(120,140,170,.10);border:1px solid rgba(120,140,170,.25);
  padding:12px 16px;border-radius:12px}
[data-testid="stMetricValue"]{font-size:1.55rem}
.block-container{padding-top:1.6rem}
.risk-pill{display:inline-block;padding:4px 14px;border-radius:999px;color:white;font-weight:700}
@media (max-width:640px){[data-testid="stMetricValue"]{font-size:1.2rem}}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- cached loaders
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def table(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    return _csv(str(p), _mtime(p)) if p.exists() else None


def jfile(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_resource(show_spinner=False)
def _bundle(path: str, mtime: float):
    return joblib.load(path)


def bundle():
    p = MODELS / "churn_model.joblib"
    return _bundle(str(p), _mtime(p))


@st.cache_data(show_spinner=False)
def _engineered(path: str, mtime: float) -> pd.DataFrame:
    return engineer_features(pd.read_csv(path), CFG)


def engineered() -> pd.DataFrame:
    p = resolve_path(CFG, "clean_data")
    return _engineered(str(p), _mtime(p))


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def show_image(name: str, caption: str | None = None):
    p = FIGS / name
    if p.exists():
        st.image(str(p), caption=caption)
    else:
        st.info(f"Chart `{name}` not found - run the pipeline step that creates it.")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def df_view(df: pd.DataFrame):
    st.dataframe(friendly_columns(df), hide_index=True)


def paginated(df: pd.DataFrame, key: str, default_sort: str, search_cols=None, ascending=False):
    """Search + sort + pagination controls. Returns (page_df, filtered_all_rows_df)."""
    c1, c2, c3, c4 = st.columns([3, 2, 1.2, 1.2])
    q = c1.text_input("Search", key=f"{key}_q", placeholder="Search customer, action, contract ...")
    sortable = [c for c in df.columns if c != "all_actions"]
    idx = sortable.index(default_sort) if default_sort in sortable else 0
    sort_by = c2.selectbox("Sort by", sortable, index=idx, key=f"{key}_sort",
                           format_func=lambda c: friendly_columns(pd.DataFrame(columns=[c])).columns[0])
    asc = c3.selectbox("Order", ["Descending", "Ascending"], index=1 if ascending else 0, key=f"{key}_ord") == "Ascending"
    size = c4.selectbox("Rows", PAGE_SIZES, index=1, key=f"{key}_size")
    _, total, pages, _ = search_sort_paginate(df, q, sort_by, asc, 1, size, search_cols)
    page = st.number_input(f"Page (1-{pages})", 1, pages, 1, key=f"{key}_page") if pages > 1 else 1
    page_df, total, pages, page = search_sort_paginate(df, q, sort_by, asc, int(page), size, search_cols)
    all_rows = search_sort_paginate(df, q, sort_by, asc, 1, max(total, 1), search_cols)[0]
    st.caption(f"{total:,} rows | page {page} of {pages}")
    return page_df, all_rows


def kpi_row(items):
    cols = st.columns(len(items))
    for c, (label, value, *help_) in zip(cols, items):
        c.metric(label, value, help=help_[0] if help_ else None)


# ----------------------------------------------------------------------------- readiness gate
GEN = st.session_state.get("generic")                 # result of an "any dataset" analysis for THIS session
ready, missing = artifacts_ready(CFG)
st.sidebar.title("📉 Churn Intelligence")
PAGES = (["Data Upload", "Dataset Analysis"] if GEN else
         ["Data Upload", "Executive Summary", "Data Quality", "Exploratory Analysis", "Statistical Insights",
          "Customer Explorer", "Churn Prediction", "Prediction History", "Model Performance", "Feature Importance",
          "Retention Recommendations", "Revenue at Risk"])
if "_goto" in st.session_state:                      # set by the upload page after a successful analysis
    st.session_state["page"] = st.session_state.pop("_goto")
if st.session_state.get("page") not in PAGES:
    st.session_state["page"] = PAGES[1]
page = st.sidebar.radio("Navigate", PAGES, index=1, key="page")
meta = st.session_state.get("workspace_meta")
if GEN:
    st.sidebar.success(f"Showing YOUR dataset: {GEN['source']} ({GEN['quality']['overview']['clean_rows']:,} rows)")
    if st.sidebar.button("Switch back to demo data", key="to_demo_g"):
        st.session_state.pop("generic", None)
        st.rerun()
elif meta:
    st.sidebar.success(f"Showing YOUR data: {meta['name']} ({meta['rows']:,} rows)")
    if st.sidebar.button("Switch back to demo data", key="to_demo"):
        for k in ("workspace_cfg", "workspace_meta"):
            st.session_state.pop(k, None)
        st.rerun()
else:
    st.sidebar.info("Showing the built-in IBM Telco demo data. Use **Data Upload** to analyse your own file.")
st.sidebar.caption("churn_ds v1.2")


def rebuild_ui(reason: str):
    """Offer to (re)build every artifact from the bundled sample data (fast mode)."""
    st.title("Churn Intelligence Dashboard")
    st.warning(reason)
    st.code("python run_pipeline.py seed\npython run_pipeline.py all", language="bash")
    st.write("Or build everything now with the bundled sample dataset and fast settings (about a minute):")
    if st.button("Run pipeline now (sample data, fast mode)", key="run_now"):
        from churn.pipeline import run_all
        fast_cfg = load_config(overrides={"modeling": {"fast": True, "cv_folds": 3}})
        with st.spinner("Running the full data-science pipeline ..."):
            run_all(fast_cfg, sample=True, force=True)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.stop()


if GEN:                                                # any-dataset mode needs no telecom demo artifacts
    B, THR, customers, rev, met = None, 0.5, None, None, None
else:
    if not ready:
        rebuild_ui("The pipeline has not produced all artifacts yet. Missing: " + ", ".join(missing))
    try:
        B = bundle()
    except Exception as exc:  # e.g. model pickled with a different scikit-learn version
        rebuild_ui(f"The saved model could not be loaded ({type(exc).__name__}: {exc}). "
                   "Retrain it in this environment.")
    THR = resolve_threshold(CFG, B)
    customers = table("customer_recommendations.csv")
    rev = jfile(TABLES / "revenue_at_risk_summary.json")
    met = jfile(MODELS / "metrics_test.json") or (jfile(MODELS / "metrics.json") or {}).get("test", {})


# ============================================================================= pages
def page_exec():
    st.title("Executive Summary")
    st.caption("Who is likely to leave, why, and what it is worth - for the whole customer base.")
    kpi_row([("Customers", f"{rev['customers']:,}"),
             ("Historical churn rate", pct(float((customers['Churn'] == 'Yes').mean()))),
             ("Flagged high-risk", f"{rev['flagged_customers']:,}", f"Churn probability >= {rev['threshold']:.2f}"),
             ("Revenue at risk / month", format_currency(rev["monthly_revenue_at_risk"], CUR, True)),
             ("Revenue at risk / year", format_currency(rev["annual_revenue_at_risk"], CUR, True))])
    kpi_row([("Model", B["model_name"]), ("ROC-AUC (test)", f"{met['roc_auc']:.3f}"),
             ("Recall (test)", f"{met['recall']:.3f}", "Share of real churners the model catches"),
             ("Precision (test)", f"{met['precision']:.3f}", "Share of flagged customers who really churn"),
             ("Critical-risk customers", f"{rev['critical_customers']:,}")])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Customers by risk tier")
        order = ["Low", "Medium", "High", "Critical"]
        t = customers.groupby("risk_tier").agg(customers=("flagged", "size"), mrr=("MonthlyCharges", "sum")).reindex(order).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.bar(t.index, t["customers"], color=[RISK_COLORS[x] for x in t.index])
        for i, v in enumerate(t["customers"]):
            ax.text(i, v, f"{int(v):,}", ha="center", va="bottom")
        ax.set_ylabel("Customers")
        show_fig(fig)
    with c2:
        st.subheader("Annual revenue at risk by contract")
        bc = table("insights_by_contract.csv")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.bar(bc["Contract"], bc["annual_revenue_at_risk"] / 1e6, color=ACCENT)
        ax.set_ylabel(f"{CUR} millions per year")
        show_fig(fig)
    st.subheader("Key insights")
    eda = jfile(TABLES / "eda_insights.json") or {"insights": []}
    for i in eda["insights"][:7]:
        st.markdown(f"- {i}")
    st.subheader("Top retention priorities")
    recs = table("insights_recommendations.csv")
    if recs is not None:
        df_view(recs[["recommended_action", "risk_factor", "flagged_customers_affected", "monthly_revenue_covered"]].head(5))
    rp = REPORTS / "final_report.pdf"
    if rp.exists():
        st.download_button("Download final report (PDF)", rp.read_bytes(), "churn_final_report.pdf", key="dl_pdf")


def page_quality():
    st.title("Data Quality")
    ov = jfile(TABLES / "quality_overview.json")
    if not ov:
        st.info("Run `python run_pipeline.py clean` first.")
        return
    kpi_row([("Raw rows", f"{ov['raw_rows']:,}"), ("Clean rows", f"{ov['clean_rows']:,}"),
             ("Duplicate rows (raw)", ov["duplicate_rows_raw"]), ("Missing cells (raw)", ov["missing_cells_raw"]),
             ("Missing cells (clean)", ov["missing_cells_clean"])])
    tabs = st.tabs(["Missing values", "Data types", "Invalid categories", "Numeric summary", "Distributions",
                    "Outliers", "Cleaning log"])
    names = ["quality_missing_raw", "quality_dtypes", "quality_invalid_categories", "quality_numeric_summary",
             "quality_categorical_distribution", "quality_outliers"]
    for tab, n in zip(tabs[:6], names):
        with tab:
            d = table(f"{n}.csv")
            if d is None or d.empty:
                st.success("Nothing to report - no issues found." if "invalid" in n else "No data.")
                continue
            if n == "quality_categorical_distribution":
                col = st.selectbox("Column", d["column"].unique(), key="qd_col")
                d = d[d["column"] == col]
            st.dataframe(d, hide_index=True)
            st.download_button("Download CSV", to_csv_bytes(d), f"{n}.csv", key=f"dl_{n}")
    with tabs[6]:
        log = pd.read_csv(resolve_path(CFG, "logs_dir") / "cleaning_log.csv")
        st.dataframe(log, hide_index=True)
        st.download_button("Download cleaning log", to_csv_bytes(log), "cleaning_log.csv", key="dl_clog")
    with st.expander("Full data-quality report (markdown)"):
        p = REPORTS / "data_quality_report.md"
        st.markdown(p.read_text(encoding="utf-8") if p.exists() else "_not generated_")


def page_eda():
    st.title("Exploratory Analysis")
    df = engineered()
    overall = float(df[TARGET_FLAG].mean())
    st.caption(f"Overall churn rate: {pct(overall)}. Charts below are computed live from the cleaned data.")
    tab1, tab2, tab3, tab4 = st.tabs(["Churn by segment", "Numeric distributions", "Multivariate", "Saved charts"])
    with tab1:
        dims = ["Contract", "tenure_group", "InternetService", "PaymentMethod", "SeniorCitizen", "gender", "TechSupport",
                "OnlineSecurity", "Partner", "Dependents", "PaperlessBilling", "customer_value_segment", "service_profile"]
        dim = st.selectbox("Break churn down by", dims, format_func=column_label, key="eda_dim")
        g = churn_rate_by(df, dim)
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(data=g, x=dim, y="churn_rate", color=BLUE, ax=ax)
        ax.axhline(overall, color=ACCENT, ls="--", label="Overall")
        for i, r in enumerate(g.itertuples()):
            ax.text(i, r.churn_rate, f"{100 * r.churn_rate:.1f}%", ha="center", va="bottom")
        ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        ax.set_xlabel(column_label(dim)); ax.legend(); plt.xticks(rotation=15)
        show_fig(fig)
        st.dataframe(g.assign(churn_rate=g["churn_rate"].round(4)), hide_index=True)
    with tab2:
        num = st.selectbox("Numeric variable", ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
                                                 "total_services"], format_func=column_label, key="eda_num")
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x=num, hue="Churn", multiple="layer", bins=30, palette={"No": BLUE, "Yes": ACCENT}, ax=ax)
        with c1:
            show_fig(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Churn", y=num, hue="Churn", palette={"No": BLUE, "Yes": ACCENT}, legend=False, ax=ax)
        with c2:
            show_fig(fig)
        st.dataframe(df.groupby("Churn")[num].describe().round(2))
    with tab3:
        c1, c2 = st.columns(2)
        rows = c1.selectbox("Rows", ["Contract", "tenure_group", "PaymentMethod", "InternetService"], key="mv_r",
                            format_func=column_label)
        cols = c2.selectbox("Columns", ["InternetService", "Contract", "customer_value_segment", "SeniorCitizen"],
                            key="mv_c", format_func=column_label)
        if rows == cols:
            st.warning("Choose two different variables.")
        else:
            pv = df.pivot_table(index=rows, columns=cols, values=TARGET_FLAG, aggfunc="mean", observed=True)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.heatmap(pv, annot=True, fmt=".0%", cmap="OrRd", ax=ax)
            ax.set_title("Churn rate")
            show_fig(fig)
        corr = table("eda_correlation_matrix.csv")
        if corr is not None:
            st.subheader("Correlation matrix & multicollinearity")
            corr = corr.set_index(corr.columns[0])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            show_fig(fig)
            vif = table("eda_vif.csv")
            if vif is not None:
                st.dataframe(vif.round(2), hide_index=True)
    with tab4:
        imgs = sorted(FIGS.glob("eda_*.png"))
        for p in imgs:
            with st.expander(p.stem.replace("_", " ")):
                st.image(str(p))


def page_stats():
    st.title("Statistical Insights")
    st.info("These tests show **association, not causation**. Customers who differ on one attribute usually differ on "
            "many others, so use results to prioritise investigation and run controlled retention tests.")
    chi, num, grp = table("stats_chi_square.csv"), table("stats_numeric_tests.csv"), table("stats_group_comparisons.csv")
    if chi is None:
        st.info("Run `python run_pipeline.py stats` first.")
        return
    only_sig = st.checkbox("Show only significant results (adjusted p < 0.05)", value=False, key="only_sig")
    t1, t2, t3, t4 = st.tabs(["Chi-square (categorical)", "Numeric tests", "Group comparisons", "Correlations"])
    with t1:
        d = chi[chi["significant"]] if only_sig else chi
        st.dataframe(d[["label", "chi2", "dof", "p_value", "p_adjusted", "cramers_v", "effect_size",
                        "highest_churn_group", "highest_churn_rate", "lowest_churn_group", "lowest_churn_rate"]]
                     .round(4), hide_index=True)
        for t in d["interpretation"].head(8):
            st.markdown(f"- {t}")
    with t2:
        d = num[num["significant"]] if only_sig else num
        st.dataframe(d[["label", "test", "statistic", "p_adjusted", "effect_size_name", "effect_size", "effect_magnitude",
                        "median_churned", "median_retained"]].round(4), hide_index=True)
        for t in d["interpretation"]:
            st.markdown(f"- {t}")
    with t3:
        st.dataframe(grp[["comparison", "rate_a", "rate_b", "difference", "risk_ratio", "p_adjusted"]].round(4),
                     hide_index=True)
        for t in grp["interpretation"]:
            st.markdown(f"- {t}")
    with t4:
        c = table("stats_correlation_with_churn.csv")
        st.dataframe(c.round(4), hide_index=True)
        show_image("eda_15_correlation_heatmap.png")
    st.download_button("Download chi-square results", to_csv_bytes(chi), "chi_square_results.csv", key="dl_chi")


def page_explorer():
    st.title("Customer Explorer")
    df = customers
    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        contracts = c1.multiselect("Contract", sorted(df["Contract"].unique()), key="f_contract")
        internet = c1.multiselect("Internet service", sorted(df["InternetService"].unique()), key="f_net")
        tiers = c2.multiselect("Risk tier", ["Low", "Medium", "High", "Critical"], key="f_tier")
        payments = c2.multiselect("Payment method", sorted(df["PaymentMethod"].unique()), key="f_pay")
        tenure = c3.slider("Tenure (months)", int(df["tenure"].min()), int(df["tenure"].max()),
                           (int(df["tenure"].min()), int(df["tenure"].max())), key="f_tenure")
        charges = c3.slider("Monthly charges", float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()),
                            (float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())), key="f_charges")
        only_flagged = st.checkbox("Only flagged high-risk customers", key="f_flagged")
    f = filter_customers(df, contracts, internet, tiers, payments, tenure, charges, only_flagged)
    k = kpi_summary(f)
    kpi_row([("Customers", f"{k['customers']:,}"), ("Flagged", f"{k['flagged']:,}"),
             ("Avg churn probability", pct(k["avg_probability"])),
             ("Monthly revenue at risk", format_currency(k["monthly_at_risk"], CUR, True)),
             ("Critical", f"{k['critical']:,}")])
    show = f[["customerID", "churn_probability", "risk_tier", "priority_score", "priority_rank", "MonthlyCharges", "tenure",
              "Contract", "InternetService", "PaymentMethod", "primary_action"]]
    page_df, all_rows = paginated(show, "exp", "priority_score", ["customerID", "Contract", "PaymentMethod", "primary_action",
                                                                  "InternetService", "risk_tier"])
    st.dataframe(friendly_columns(page_df), hide_index=True)
    st.download_button("Download filtered customers (CSV)", to_csv_bytes(f), "filtered_customers.csv", key="dl_exp")
    if len(page_df):
        st.subheader("Customer detail")
        cid = st.selectbox("Customer", page_df["customerID"].tolist(), key="exp_detail")
        row = f[f["customerID"] == cid].iloc[0]
        c1, c2 = st.columns(2)
        c1.markdown(f"**Churn probability:** {pct(row['churn_probability'])}  \n**Risk tier:** {row['risk_tier']}  \n"
                    f"**Priority rank:** {int(row['priority_rank']):,} (score {row['priority_score']})  \n"
                    f"**Monthly charges:** {CUR}{row['MonthlyCharges']:,.2f}  \n**Tenure:** {int(row['tenure'])} months")
        c2.markdown("**Recommended actions**")
        for a in (row["all_actions"].split(" | ") if isinstance(row["all_actions"], str) and row["all_actions"] else [row["primary_action"]]):
            c2.markdown(f"- {a}")
        if isinstance(row["risk_factors"], str) and row["risk_factors"]:
            c2.caption("Risk factors: " + row["risk_factors"])


FORM_OPTS = {c: ALLOWED_VALUES[c] for c in ALLOWED_VALUES}


def page_predict():
    st.title("Churn Prediction")
    st.caption("Enter a customer's details to estimate churn probability, see why, and get retention advice.")
    thr = st.slider("Decision threshold (flag as churn-risk at or above)", 0.05, 0.95, float(round(THR, 2)), 0.01,
                    key="pred_thr", help="Lower = catch more churners but contact more customers who would have stayed.")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        cid = c1.text_input("Customer ID (optional)", "NEW-CUSTOMER", key="p_id")
        gender = c1.selectbox("Gender", FORM_OPTS["gender"], key="p_gender")
        senior = c1.selectbox("Senior citizen", FORM_OPTS["SeniorCitizen"], key="p_senior")
        partner = c1.selectbox("Has partner", FORM_OPTS["Partner"], key="p_partner")
        dep = c1.selectbox("Has dependents", FORM_OPTS["Dependents"], key="p_dep")
        tenure = c1.number_input("Tenure (months)", 0, 120, 12, key="p_tenure")
        contract = c2.selectbox("Contract", FORM_OPTS["Contract"], key="p_contract")
        pay = c2.selectbox("Payment method", FORM_OPTS["PaymentMethod"], key="p_pay")
        paperless = c2.selectbox("Paperless billing", FORM_OPTS["PaperlessBilling"], index=1, key="p_paper")
        monthly = c2.number_input("Monthly charges", 0.0, 300.0, 75.0, 0.5, key="p_monthly")
        total = c2.number_input("Total charges (0 = estimate from tenure)", 0.0, 20000.0, 0.0, 10.0, key="p_total")
        phone = c3.selectbox("Phone service", FORM_OPTS["PhoneService"], index=1, key="p_phone")
        lines = c3.selectbox("Multiple lines", FORM_OPTS["MultipleLines"], key="p_lines")
        net = c3.selectbox("Internet service", FORM_OPTS["InternetService"], index=1, key="p_net")
        addon = {}
        for col, lab in [("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
                         ("DeviceProtection", "Device protection"), ("TechSupport", "Tech support"),
                         ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming movies")]:
            addon[col] = (c2 if col in ("StreamingTV", "StreamingMovies") else c3).selectbox(
                lab, FORM_OPTS[col], key=f"p_{col}")
        st.caption("Add-on services are only valid with internet service; inconsistent entries are corrected automatically.")
        submitted = st.form_submit_button("Predict churn risk")
    if submitted:
        frame = make_customer_frame(customerID=cid or "NEW-CUSTOMER", gender=gender, SeniorCitizen=senior, Partner=partner,
                                    Dependents=dep, tenure=int(tenure), PhoneService=phone, MultipleLines=lines,
                                    InternetService=net, Contract=contract, PaperlessBilling=paperless, PaymentMethod=pay,
                                    MonthlyCharges=float(monthly), TotalCharges=float(total) if total > 0 else None, **addon)
        X = prepare_features(frame, B, CFG)
        p = float(predict_proba(B, X)[0])
        eng = engineer_features(standardize_frame(frame), CFG)
        st.session_state["pred"] = {"frame": frame, "X": X, "eng": eng, "p": p, "id": cid}
        rec0 = recommend_for_frame(eng, p, thr, CFG)
        append_history(HISTORY, {"customerID": cid, "churn_probability": round(p, 4), "threshold": thr,
                                 "predicted_churn": int(p >= thr), "risk_tier": rec0["risk_tier"],
                                 "primary_action": rec0["primary_action"], "inputs": frame.iloc[0].to_dict()})
    res = st.session_state.get("pred")
    if not res:
        st.info("Fill in the form and press **Predict churn risk**.")
        return
    p, tier = res["p"], risk_label(res["p"], thr, CFG["threshold"]["critical_probability"])
    rec = recommend_for_frame(res["eng"], p, thr, CFG)
    st.divider()
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Churn probability", pct(p))
    c2.markdown(f"**Risk tier**<br><span class='risk-pill' style='background:{RISK_COLORS[tier]}'>{tier}</span>",
                unsafe_allow_html=True)
    c3.markdown(f"**Decision at threshold {thr:.2f}:** " + ("🔴 flag for retention action" if p >= thr else "🟢 no immediate action"))
    st.progress(min(max(p, 0.0), 1.0))
    mrr = float(res["frame"]["MonthlyCharges"].iloc[0])
    if p >= thr:
        st.write(f"Revenue at risk if this customer leaves: **{CUR}{mrr:,.0f}/month ({CUR}{12 * mrr:,.0f}/year)**.")
    left, right = st.columns(2)
    with left:
        st.subheader("Why this prediction")
        ex = explain_customer(B, res["X"], top_n=8)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        d = ex.iloc[::-1]
        ax.barh(d["driver"], d["effect_pp"], color=[ACCENT if v > 0 else BLUE for v in d["effect_pp"]])
        ax.axvline(0, color="black", lw=.8)
        ax.set_xlabel("Effect on churn probability (percentage points vs a typical customer)")
        show_fig(fig)
        for t in ex["explanation"].head(4):
            st.markdown(f"- {t}")
        st.caption("Effects are computed from this model by replacing each driver's values with those of typical "
                   "training customers. They show what the model uses, not proven causes.")
        if st.checkbox("Also show SHAP values (log-odds contribution of encoded features)", key="pred_shap"):
            sh = shap_local(B, res["X"])
            if sh is None:
                st.info("SHAP is not available in this environment.")
            else:
                st.dataframe(sh.round(3), hide_index=True)
    with right:
        st.subheader("Recommended retention actions")
        st.markdown(f"**Primary:** {rec['primary_action']}")
        for a in rec["all_actions"]:
            st.markdown(f"- {a}")
        if rec["risk_factors"]:
            st.caption("Triggered risk factors: " + ", ".join(rec["risk_factors"]))
        st.caption("Prediction saved to Prediction History.")


def page_history():
    st.title("Prediction History")
    h = load_history(HISTORY)
    if h.empty:
        st.info("No predictions yet. Make one on the **Churn Prediction** page.")
        return
    kpi_row([("Predictions", f"{len(h):,}"), ("Flagged as churn-risk", f"{int(h['predicted_churn'].sum()):,}"),
             ("Average probability", pct(float(h["churn_probability"].mean())))])
    view = h.drop(columns=["inputs"])
    page_df, _ = paginated(view, "hist", "timestamp", ["customerID", "risk_tier", "primary_action"])
    st.dataframe(friendly_columns(page_df), hide_index=True)
    st.download_button("Download history (CSV)", to_csv_bytes(h), "prediction_history.csv", key="dl_hist")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(len(h)), h["churn_probability"].iloc[::-1].to_numpy(), "o-", color=ACCENT)
    ax.set_xlabel("Prediction #"); ax.set_ylabel("Churn probability")
    show_fig(fig)
    if st.button("Clear history", key="clear_hist"):
        clear_history(HISTORY)
        st.rerun()


def page_performance():
    st.title("Model Performance")
    st.caption(f"Final model: **{B['model_name']}** | trained {B['trained_at']} | dataset version `{B['dataset_version']}` "
               f"| evaluated on {B['n_test']:,} untouched test customers")
    tp = table("test_predictions.csv")
    comp = table("model_comparison.csv")
    kpi_row([("ROC-AUC", f"{met['roc_auc']:.3f}"), ("PR-AUC", f"{met['pr_auc']:.3f}"), ("Accuracy", f"{met['accuracy']:.3f}"),
             ("Precision", f"{met['precision']:.3f}"), ("Recall", f"{met['recall']:.3f}"), ("F1", f"{met['f1']:.3f}")])
    st.markdown("**Why these metrics?** Churners are a minority (~27%), so accuracy hides missed churners. **Recall** = "
                "share of churners caught, **precision** = share of flagged customers who really leave, **F1** balances "
                "them, **ROC-AUC** measures how well the model ranks churners above stayers regardless of threshold.")
    st.info(B.get("selection_rationale", ""))
    if tp is not None:
        st.subheader("Try a different threshold")
        t = st.slider("Probability threshold", 0.05, 0.95, float(round(THR, 2)), 0.01, key="perf_thr")
        m = metrics_at_threshold(tp["actual_churn"], tp["churn_probability"], t)
        kpi_row([("Precision", f"{m['precision']:.3f}"), ("Recall", f"{m['recall']:.3f}"), ("F1", f"{m['f1']:.3f}"),
                 ("Customers flagged", f"{m['flagged']:,}"), ("Churners missed", f"{m['fn']:,}")])
        st.caption("Lower threshold: more churners caught, more wasted offers. Higher threshold: fewer offers, more missed churners.")
        arr = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(arr, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred. retained", "Pred. churn"], yticklabels=["Actually retained", "Actually churned"])
        ax.set_title(f"Confusion matrix @ {t:.2f}")
        show_fig(fig)
    c1, c2 = st.columns(2)
    with c1:
        show_image("eval_roc_curve.png")
        show_image("eval_confusion_matrix.png")
    with c2:
        show_image("eval_precision_recall_curve.png")
        show_image("eval_calibration.png")
    show_image("eval_threshold_tradeoff.png")
    show_image("eval_cumulative_gains.png")
    if comp is not None:
        st.subheader("Model comparison (5-fold CV on training data)")
        st.dataframe(comp.drop(columns=[c for c in comp.columns if c.endswith("_std")]).round(4), hide_index=True)
    runs = (MODELS / "model_runs.csv")
    if runs.exists():
        with st.expander("Model run registry"):
            st.dataframe(pd.read_csv(runs), hide_index=True)
    cr = table("eval_classification_report.csv")
    if cr is not None:
        with st.expander("Classification report"):
            st.dataframe(cr.round(3), hide_index=True)
    thr_tbl = table("eval_threshold_table.csv")
    if thr_tbl is not None:
        with st.expander("Threshold table with campaign net benefit"):
            st.dataframe(thr_tbl.round(3), hide_index=True)
            st.download_button("Download threshold table", to_csv_bytes(thr_tbl), "threshold_table.csv", key="dl_thr")


def page_importance():
    st.title("Feature Importance")
    gi, fi, ve = table("explain_group_importance.csv"), table("explain_feature_importance.csv"), table("explain_value_effects.csv")
    if gi is None:
        st.info("Run `python run_pipeline.py explain` first.")
        return
    st.markdown("**Business drivers** group related features (e.g. contract type + month-to-month flag). Importance is the drop in test "
                "ROC-AUC when the driver's values are shuffled - bigger drop = the model relies on it more.")
    show_image("explain_group_importance.png")
    st.dataframe(gi[["driver", "importance_mean", "importance_std", "mean_abs_effect_pp", "columns"]].round(4), hide_index=True)
    st.subheader("How each driver moves risk")
    drv = st.selectbox("Driver", gi["driver"].tolist(), key="imp_drv")
    sub = ve[ve["driver"] == drv].sort_values("mean_effect_pp", ascending=False) if ve is not None else pd.DataFrame()
    if len(sub):
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(sub) + 1.5))
        ax.barh(sub["value"].iloc[::-1], sub["mean_effect_pp"].iloc[::-1],
                color=[ACCENT if v > 0 else BLUE for v in sub["mean_effect_pp"].iloc[::-1]])
        ax.axvline(0, color="black", lw=.8); ax.set_xlabel("Average effect on churn probability (percentage points)")
        show_fig(fig)
    t1, t2, t3 = st.tabs(["Feature-level (permutation)", "Model-native", "SHAP"])
    with t1:
        show_image("explain_feature_importance.png")
    with t2:
        show_image("explain_native_importance.png")
    with t3:
        if (FIGS / "explain_shap_summary.png").exists():
            show_image("explain_shap_summary.png")
            show_image("explain_shap_bar.png")
        else:
            st.info("SHAP outputs not available (shap not installed or incompatible). Permutation importance is used instead.")
    st.subheader("Explain a specific customer")
    cid = st.text_input("Customer ID", value=str(customers["customerID"].iloc[0]), key="imp_cid")
    data = pd.read_csv(resolve_path(CFG, "modeling_data")).set_index(ID_COL)
    if cid in data.index:
        X = data.loc[[cid], B["input_columns"]]
        p = float(predict_proba(B, X)[0])
        st.metric("Churn probability (model fitted on training split)", pct(p))
        for t in explain_customer(B, X)["explanation"]:
            st.markdown(f"- {t}")
    else:
        st.warning("Customer ID not found.")


def page_recs():
    st.title("Retention Recommendations")
    recs, seg = table("insights_recommendations.csv"), table("insights_segments.csv")
    st.markdown("Recommendations come from transparent **rules** over contract, tenure, price, payment method, services and "
                "predicted risk - no external AI. Each customer gets a primary action plus any secondary actions.")
    st.subheader("Action playbook")
    df_view(recs)
    st.subheader("Segment-level plan (contract x tenure band)")
    urg = st.multiselect("Urgency", ["Urgent", "High", "Monitor"], default=["Urgent", "High"], key="seg_urg")
    s = seg[seg["urgency"].isin(urg)] if urg else seg
    df_view(s.round(3))
    st.download_button("Download segment plan", to_csv_bytes(seg), "segment_plan.csv", key="dl_seg")
    st.subheader("Customer-level recommendations")
    c1, c2 = st.columns(2)
    tiers = c1.multiselect("Risk tier", ["Medium", "High", "Critical"], default=["High", "Critical"], key="rec_tier")
    acts = c2.multiselect("Primary action", sorted(customers["primary_action"].unique()), key="rec_act")
    f = customers[customers["risk_tier"].isin(tiers)] if tiers else customers
    if acts:
        f = f[f["primary_action"].isin(acts)]
    cols = ["customerID", "churn_probability", "risk_tier", "priority_score", "MonthlyCharges", "tenure", "Contract",
            "primary_action", "risk_factors"]
    page_df, _ = paginated(f[cols], "rec", "priority_score", ["customerID", "primary_action", "Contract", "risk_factors"])
    st.dataframe(friendly_columns(page_df), hide_index=True)
    st.download_button("Download customer recommendations (CSV)", to_csv_bytes(f), "customer_recommendations.csv", key="dl_rec")


def page_revenue():
    st.title("Revenue at Risk")
    st.caption("Revenue = monthly charges of customers the model flags as likely to churn.")
    kpi_row([("Monthly revenue at risk", format_currency(rev["monthly_revenue_at_risk"], CUR)),
             ("Annual revenue at risk", format_currency(rev["annual_revenue_at_risk"], CUR)),
             ("Share of monthly revenue", pct(rev["pct_of_monthly_revenue_at_risk"])),
             ("Expected monthly loss (prob.-weighted)", format_currency(rev["expected_monthly_loss_all_customers"], CUR)),
             ("Critical-tier monthly revenue", format_currency(rev["critical_monthly_revenue"], CUR))])
    st.subheader("What-if campaign calculator")
    c1, c2, c3 = st.columns(3)
    t = c1.slider("Threshold", 0.05, 0.95, float(round(THR, 2)), 0.01, key="rev_thr")
    sr = c2.slider("Save rate (share of at-risk revenue retained)", 0.05, 0.80, float(CFG["business"]["save_rate"]), 0.05, key="rev_sr")
    oc = c3.number_input(f"Cost per retention offer ({CUR})", 0.0, 500.0, float(CFG["business"]["retention_offer_cost"]), 5.0, key="rev_oc")
    sc = scenario_net_benefit(customers, t, sr, oc)
    kpi_row([("Customers contacted", f"{sc['customers_contacted']:,}"),
             ("Annual revenue at risk", format_currency(sc["annual_revenue_at_risk"], CUR, True)),
             ("Expected revenue saved / yr", format_currency(sc["expected_annual_revenue_saved"], CUR, True)),
             ("Campaign cost", format_currency(sc["campaign_cost"], CUR, True)),
             ("Net benefit / yr", format_currency(sc["net_benefit"], CUR, True))])
    st.caption("Expected saved revenue = sum(churn probability x annual charges x save rate) over contacted customers. "
               "Illustrative: validate the save rate with an A/B test.")
    st.subheader("Where is the revenue at risk?")
    dims = {"Contract": "insights_by_contract.csv", "Tenure band": "insights_by_tenure.csv",
            "Internet service": "insights_by_internet.csv", "Value segment": "insights_by_value.csv",
            "Risk tier": "insights_by_tier.csv"}
    pick = st.selectbox("Break down by", list(dims), key="rev_dim")
    d = table(dims[pick])
    key = d.columns[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(d[key].astype(str), d["annual_revenue_at_risk"] / 1e3, color=ACCENT)
    ax.set_ylabel(f"Annual revenue at risk ({CUR}K)"); plt.xticks(rotation=15)
    show_fig(fig)
    st.dataframe(d.round(3), hide_index=True)
    st.subheader("Priority customers (highest first)")
    pr = customers[customers["flagged"].astype(bool)][["customerID", "churn_probability", "risk_tier", "priority_score",
                                                        "MonthlyCharges", "annual_revenue_at_risk", "primary_action"]]
    page_df, all_rows = paginated(pr, "rev", "priority_score", ["customerID", "primary_action"])
    st.dataframe(friendly_columns(page_df), hide_index=True)
    st.download_button("Download priority customers (CSV)", to_csv_bytes(pr.sort_values("priority_score", ascending=False)),
                       "priority_customers.csv", key="dl_prio")
    seg = table("insights_segments.csv")
    st.download_button("Download revenue-at-risk by segment (CSV)", to_csv_bytes(seg), "revenue_at_risk_segments.csv", key="dl_revseg")


STEP_LABELS = {"clean": "Cleaning data and running quality checks", "eda": "Exploratory analysis and charts",
               "stats": "Statistical hypothesis tests", "train": "Training, tuning and comparing models",
               "evaluate": "Evaluating on the hold-out test set", "explain": "Explainability (importance, SHAP)",
               "insights": "Revenue at risk and retention recommendations", "report": "Building the reports"}


def _read_upload(f) -> pd.DataFrame:
    name = f.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    try:
        return pd.read_csv(f)
    except UnicodeDecodeError:
        f.seek(0)
        return pd.read_csv(f, encoding="latin-1")


def _generic_flow(raw: pd.DataFrame, up, uid: str):
    st.subheader("Tell the app what to predict")
    st.caption("Works with any table that has a yes/no outcome (for example Churn = 0/1, or Yes/No). Everything else is inferred automatically.")
    s0 = gen.suggest_settings(raw)
    cols = list(raw.columns)
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Outcome column (what to predict)", cols, index=cols.index(s0["target"]) if s0["target"] in cols else 0,
                          key=f"g_t_{uid}")
    uniq = sorted(raw[target].dropna().unique().tolist(), key=str)[:50]
    guess = gen.guess_positive_label(raw[target])
    positive = c1.selectbox("Which value means 'churned / positive'?", uniq, index=uniq.index(guess) if guess in uniq else 0,
                            key=f"g_p_{uid}_{target}", format_func=str)
    id_opts = ["(none)"] + [c for c in cols if c != target]
    id_col = c2.selectbox("Customer ID column (optional)", id_opts, key=f"g_i_{uid}_{target}",
                          index=id_opts.index(s0["id_col"]) if s0["id_col"] in id_opts else 0)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(raw[c]) and c != target]
    v_opts = ["(none)"] + num_cols
    value_col = c3.selectbox("Value column for 'value at risk' (optional)", v_opts, key=f"g_v_{uid}_{target}",
                             index=v_opts.index(s0["value_col"]) if s0["value_col"] in v_opts else 0,
                             help="A numeric column such as spend or monthly charges; used to estimate the value held by at-risk customers.")
    ignorable = [c for c in cols if c not in (target, id_col)]
    exclude = st.multiselect("Columns to ignore (e.g. anything that reveals the outcome)", ignorable, key=f"g_x_{uid}_{target}")
    settings = {"target": target, "positive": positive, "id_col": None if id_col == "(none)" else id_col,
                "value_col": None if value_col == "(none)" else value_col, "exclude": exclude}
    errors, warnings, info = gen.validate_generic(raw, settings)
    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)
    if errors:
        return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{info['n_rows']:,}")
    k2.metric("Positive outcomes", f"{info['n_pos']:,}")
    k3.metric("Positive rate", f"{100 * info['pos_rate']:.1f}%")
    k4.metric("Feature columns", info["n_features"])
    profile = st.selectbox("Training depth", list(TRAINING_PROFILES), index=0, key=f"g_prof_{uid}",
                           help="Deeper search can improve the model slightly but takes much longer on a shared server.")
    st.caption("The app will clean and profile the data, run statistical tests, tune four models, evaluate on a hold-out set, explain "
               "the drivers, score every customer and build a report. Large files are sampled for training.")
    if st.button("Run analysis on my data", key=f"g_run_{uid}", type="primary"):
        bar = st.progress(0.0, text="Queued ...")

        def cb(step, i, n):
            bar.progress(i / n, text=f"Step {i + 1} of {n}: {gen.STEP_LABELS[step]}")
        try:
            res = gen.run_generic_analysis(raw, settings, BASE_CFG, profile, cb, up.name)
        except Exception as exc:  # noqa: BLE001
            st.error(f"The analysis failed: {type(exc).__name__}: {exc}")
            return
        bar.progress(1.0, text="Done")
        for k in ("workspace_cfg", "workspace_meta"):
            st.session_state.pop(k, None)
        st.session_state["generic"] = res
        st.session_state["_goto"] = "Dataset Analysis"
        st.rerun()


def page_generic():
    res = st.session_state["generic"]
    st.title("Dataset Analysis")
    st.caption(f"File: **{res['source']}** | outcome: **{res['outcome']}** | model: **{res['model_name']}** | created {res['created']}")
    num, cat, feats = res["numeric"], res["categorical"], res["features"]
    eda, q, st_ = res["eda_df"], res["quality"], res["stats"]
    m, k = res["metrics"], res["kpis"]
    tabs = st.tabs(["Overview", "Data Quality", "Exploration", "Statistics", "Model Performance", "Drivers", "Customers",
                    "Predict one customer", "Report & downloads"])
    with tabs[0]:
        kpi_row([("Rows analysed", f"{q['overview']['clean_rows']:,}"), ("Outcome rate", pct(res["base_rate"])),
                 ("Flagged high-risk", f"{k['flagged']:,}", f"Probability >= {res['threshold']:.2f}"),
                 ("ROC-AUC (test)", f"{m['roc_auc']:.3f}"), ("Recall (test)", f"{m['recall']:.3f}")])
        if k["value_at_risk"] is not None:
            kpi_row([("Value at risk", format_currency(k["value_at_risk"], "", True), f"Sum of {res['settings']['value_col']} for flagged rows"),
                     ("Share of total value", pct(k["value_at_risk"] / max(k["value_total"], 1e-9))),
                     ("Expected value loss", format_currency(k["expected_value_loss"], "", True)),
                     ("Critical-risk rows", f"{k['critical']:,}"), ("Rows scored", f"{k['rows_scored']:,}")])
        if k["flagged_share"] > 0.7:
            st.warning(f"{pct(k['flagged_share'])} of rows are flagged because the outcome is very common in this data. "
                       "Raise the threshold on the Model Performance tab to flag fewer, higher-risk customers.")
        if res["leakage_warning"]:
            st.error("Test ROC-AUC above 0.98 is unusually high. A column may be a copy of the outcome (data leakage) - "
                     "check the top drivers and ignore any such column.")
        st.info(res["selection_rationale"])
        st.subheader("Strongest findings")
        for t in list(st_["chi_square"]["interpretation"].head(4)) + list(st_["numeric"]["interpretation"].head(4)):
            st.markdown(f"- {t}")
        st.subheader("Top drivers (model)")
        imp = res["importance"].head(8)
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(imp) + 1.5))
        ax.barh(imp["feature"].iloc[::-1], imp["importance_mean"].iloc[::-1], color=ACCENT)
        ax.set_xlabel("Drop in test ROC-AUC when shuffled")
        show_fig(fig)
    with tabs[1]:
        ov = q["overview"]
        kpi_row([("Raw rows", f"{ov['raw_rows']:,}"), ("Clean rows", f"{ov['clean_rows']:,}"), ("Duplicates (raw)", f"{ov['duplicate_rows_raw']:,}"),
                 ("Missing cells (raw)", f"{ov['missing_cells_raw']:,}"), ("Numeric / categorical", f"{ov['n_numeric']} / {ov['n_categorical']}")])
        t1, t2, t3, t4 = st.tabs(["Column profile", "Numeric summary", "Categories", "Cleaning log"])
        with t1:
            st.dataframe(q["profile"].round(2), hide_index=True)
        with t2:
            st.dataframe(q["numeric_summary"].round(3), hide_index=True)
        with t3:
            st.dataframe(q["category_distribution"].round(2), hide_index=True)
        with t4:
            st.dataframe(res["cleaning_log"], hide_index=True)
    with tabs[2]:
        st.caption(f"Charts use a stratified sample of {len(eda):,} rows.")
        c1, c2 = st.columns(2)
        if cat:
            cc = c1.selectbox("Outcome rate by category", cat, key="gx_cat")
            g = eda.groupby(cc)[Y_COL].agg(["mean", "size"]).reset_index().sort_values("mean", ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(6.5, 4))
            sns.barplot(data=g, x=cc, y="mean", color=BLUE, ax=ax)
            ax.axhline(res["base_rate"], color=ACCENT, ls="--", label="Overall")
            ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
            ax.set_ylabel("Outcome rate"); ax.legend(); plt.xticks(rotation=20)
            with c1:
                show_fig(fig)
        if num:
            nn = c2.selectbox("Numeric column by outcome", num, key="gx_num")
            fig, ax = plt.subplots(figsize=(6.5, 4))
            sns.histplot(data=eda.assign(Outcome=np.where(eda[Y_COL] == 1, "Positive", "Other")), x=nn, hue="Outcome", bins=30,
                         palette={"Other": BLUE, "Positive": ACCENT}, ax=ax)
            with c2:
                show_fig(fig)
        if len(num) >= 2:
            st.subheader("Correlation matrix")
            fig, ax = plt.subplots(figsize=(7, 5.5))
            sns.heatmap(st_["correlation"].rename(index={Y_COL: "outcome"}, columns={Y_COL: "outcome"}), annot=True, fmt=".2f",
                        cmap="coolwarm", center=0, ax=ax)
            show_fig(fig)
        if len(cat) >= 2:
            st.subheader("Two-way outcome rate")
            a, b = st.columns(2)
            r_ = a.selectbox("Rows", cat, key="gx_r")
            c_ = b.selectbox("Columns", [c for c in cat if c != r_], key="gx_c")
            pv = eda.pivot_table(index=r_, columns=c_, values=Y_COL, aggfunc="mean")
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(pv, annot=True, fmt=".0%", cmap="OrRd", ax=ax)
            show_fig(fig)
    with tabs[3]:
        st.info("Association, not causation. p-values are Benjamini-Hochberg adjusted across tests.")
        only = st.checkbox("Only significant results", key="gs_sig")
        for title, d, cols_ in [("Categorical columns (chi-square)", st_["chi_square"], ["variable", "chi2", "p_adjusted", "cramers_v", "effect_size", "highest_group", "highest_rate", "lowest_group", "lowest_rate"]),
                                ("Numeric columns (t-test / Mann-Whitney U)", st_["numeric"], ["variable", "test", "p_adjusted", "effect_size_name", "effect_size", "effect_magnitude", "median_positive", "median_negative"])]:
            st.subheader(title)
            if len(d):
                dd = d[d["significant"]] if only else d
                st.dataframe(dd[cols_].round(4), hide_index=True)
                for t in dd["interpretation"]:
                    st.markdown(f"- {t}")
            else:
                st.write("No columns of this type.")
    with tabs[4]:
        kpi_row([("ROC-AUC", f"{m['roc_auc']:.3f}"), ("PR-AUC", f"{m['pr_auc']:.3f}"), ("Accuracy", f"{m['accuracy']:.3f}"),
                 ("Precision", f"{m['precision']:.3f}"), ("Recall", f"{m['recall']:.3f}"), ("F1", f"{m['f1']:.3f}")])
        st.caption(f"Trained on {res['n_train']:,} rows, tested on {res['n_test']:,} untouched rows, threshold {res['threshold']:.2f}.")
        tp = res["test_predictions"]
        t = st.slider("Try a different threshold", 0.05, 0.95, float(round(res["threshold"], 2)), 0.01, key="gm_thr")
        mm = metrics_at_threshold(tp["actual"], tp["churn_probability"], t)
        kpi_row([("Precision", f"{mm['precision']:.3f}"), ("Recall", f"{mm['recall']:.3f}"), ("F1", f"{mm['f1']:.3f}"),
                 ("Flagged", f"{mm['flagged']:,}"), ("Missed", f"{mm['fn']:,}")])
        names = list(res["figures"])
        cols_ = st.columns(2)
        for i, nme in enumerate(names):
            cols_[i % 2].image(res["figures"][nme], caption=nme)
        st.subheader("Model comparison (cross-validation on training data)")
        st.dataframe(res["comparison"].drop(columns=[c for c in res["comparison"].columns if c.endswith("_std")]).round(4), hide_index=True)
        with st.expander("Threshold table"):
            st.dataframe(res["threshold_table"].round(3), hide_index=True)
    with tabs[5]:
        st.markdown("Importance = how much the test ROC-AUC drops when a column is shuffled. The effect table shows how each value moves "
                    "the predicted probability versus a typical customer (percentage points). These are model associations, not causes.")
        st.dataframe(res["importance"][["feature", "importance_mean", "importance_std"]].round(4), hide_index=True)
        f_ = st.selectbox("Feature", res["importance"]["feature"].tolist(), key="gd_feat")
        ve = res["value_effects"]
        sub = ve[ve["feature"] == f_].sort_values("mean_effect_pp", ascending=False)
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(sub) + 1.5))
        ax.barh(sub["value"].iloc[::-1], sub["mean_effect_pp"].iloc[::-1],
                color=[ACCENT if v > 0 else BLUE for v in sub["mean_effect_pp"].iloc[::-1]])
        ax.axvline(0, color="black", lw=.8); ax.set_xlabel("Average effect on probability (percentage points)")
        show_fig(fig)
    with tabs[6]:
        sc = res["scores"]
        c1, c2, c3 = st.columns(3)
        tiers = c1.multiselect("Risk tier", ["Low", "Medium", "High", "Critical"], default=["High", "Critical"], key="gc_tier")
        flagged_only = c2.checkbox("Flagged only", key="gc_flag")
        f = sc[sc["risk_tier"].isin(tiers)] if tiers else sc
        if flagged_only:
            f = f[f["flagged"]]
        show = [c for c in sc.columns if c not in ("score_source",)][:12]
        page_df, _ = paginated(f[show], "gen", "priority_score", [c for c in show if c in cat or c == res["settings"].get("id_col")])
        st.dataframe(friendly_columns(page_df), hide_index=True)
        st.download_button("Download scored customers (CSV)", to_csv_bytes(f), "scored_customers.csv", key="dl_gen_scores")
    with tabs[7]:
        st.caption("Enter values for one customer to get a probability and an explanation from the trained model.")
        with st.form("gen_form"):
            cols_ = st.columns(3)
            vals = {}
            for i, c in enumerate(feats):
                if c in num:
                    integer = pd.api.types.is_integer_dtype(eda[c]) or (eda[c].dropna() % 1 == 0).all()
                    vals[c] = cols_[i % 3].number_input(c, value=float(eda[c].median()), step=1.0 if integer else 0.1, key=f"gp_{c}")
                else:
                    vals[c] = cols_[i % 3].selectbox(c, eda[c].value_counts().index.tolist()[:50], key=f"gp_{c}")
            go = st.form_submit_button("Predict")
        if go:
            row = pd.DataFrame([vals])
            p = float(gen.predict_proba(res["bundle"], row)[0])
            st.session_state["gen_pred"] = {"p": p, "row": row}
        gp = st.session_state.get("gen_pred")
        if gp:
            thr = res["threshold"]
            tier = risk_label(gp["p"], thr, 0.7)
            a, b = st.columns([1, 2])
            a.metric("Probability", pct(gp["p"]))
            b.markdown(f"<span class='risk-pill' style='background:{RISK_COLORS[tier]}'>{tier}</span> &nbsp; "
                       + ("flag for action" if gp["p"] >= thr else "no immediate action"), unsafe_allow_html=True)
            ex = gen.explain_one(res, gp["row"])
            fig, ax = plt.subplots(figsize=(7, 4))
            d = ex.iloc[::-1]
            ax.barh(d["feature"], d["effect_pp"], color=[ACCENT if v > 0 else BLUE for v in d["effect_pp"]])
            ax.axvline(0, color="black", lw=.8); ax.set_xlabel("Effect (percentage points vs a typical customer)")
            show_fig(fig)
            for t in ex["explanation"].head(4):
                st.markdown(f"- {t}")
    with tabs[8]:
        st.markdown(res["report_md"])
        st.download_button("Download report (Markdown)", res["report_md"].encode("utf-8"), "analysis_report.md", key="dl_gen_report")
        st.download_button("Download scored customers (CSV)", to_csv_bytes(res["scores"]), "scored_customers_all.csv", key="dl_gen_all")
        st.download_button("Download cleaning log (CSV)", to_csv_bytes(res["cleaning_log"]), "cleaning_log.csv", key="dl_gen_log")


def page_upload():
    st.title("Data Upload")
    st.markdown("Bring **your own customer file** and either (A) train a brand-new model and get the full dashboard for your data, "
                "or (B) score your customers with the model that is currently loaded. Each browser session gets a **private "
                "temporary workspace**: other visitors never see your file, nothing is stored permanently, and workspaces are "
                "deleted after a few hours.")
    st.warning("Do not upload real personal data (names, phone numbers, addresses). Only the telecom fields listed below are "
               "needed; use anonymised customer IDs.")
    with st.expander("Which columns are required?", expanded=False):
        st.markdown("Same layout as the IBM Telco dataset. Column names are matched automatically (case, spaces and underscores "
                    "are ignored, e.g. `monthly_charges` -> `MonthlyCharges`); anything else can be mapped by hand below.")
        st.dataframe(pd.DataFrame({"Column": RAW_COLUMNS,
                                   "Needed for": ["optional (row numbers used if absent)" if c == "customerID" else
                                                  "training only (the label)" if c == "Churn" else "training and scoring"
                                                  for c in RAW_COLUMNS],
                                   "Allowed values": [", ".join(map(str, ALLOWED_VALUES.get(c, []))) if c in ALLOWED_VALUES
                                                      else "number" if c in ("tenure", "MonthlyCharges", "TotalCharges") else "text"
                                                      for c in RAW_COLUMNS]}), hide_index=True)
        sample_path = resolve_path(BASE_CFG, "sample_data")
        if sample_path.exists():
            st.download_button("Download a template / sample file (CSV)", pd.read_csv(sample_path).head(200).to_csv(index=False).encode(),
                               "churn_template.csv", key="dl_template")
    up = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"], key="uploader")
    if up is None:
        st.info("No file yet. You can also keep exploring the demo data using the other pages.")
        _show_batch_results()
        return
    ck = f"{up.name}-{up.size}"
    cached = st.session_state.get("_raw")
    if cached and cached[0] == ck:                      # avoid re-parsing a large file on every widget click
        raw = cached[1]
    else:
        try:
            raw = _read_upload(up)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the file: {exc}")
            return
        st.session_state["_raw"] = (ck, raw)
    st.success(f"Read **{up.name}**: {len(raw):,} rows x {raw.shape[1]} columns")
    st.dataframe(raw.head(8), hide_index=True)

    auto = auto_map_columns(raw.columns)
    n_match = sum(v is not None for v in auto.values())
    uid = f"{up.name}-{len(raw)}-{raw.shape[1]}"
    st.caption(f"{n_match} of the 21 telecom columns were recognised in your file.")
    layout = st.radio("How should this file be analysed?",
                      ["Any dataset - I choose the outcome column (works for any churn / yes-no data)",
                       "Telecom layout - the 21 IBM Telco columns (full telecom dashboard, recommendations, revenue at risk)"],
                      index=1 if (n_match >= 18 and B is not None) else 0, key=f"layout_{uid}")
    if layout.startswith("Any"):
        _generic_flow(raw, up, uid)
        return
    if B is None:
        st.info("The telecom dashboard needs the built-in demo model. Press **Switch back to demo data** in the sidebar first.")
        return
    with st.expander("Column mapping (auto-detected - change only if needed)", expanded=any(v is None for k, v in auto.items() if k != ID_COL)):
        mapping, cols = {}, st.columns(3)
        options = ["(none)"] + list(raw.columns)
        for i, req in enumerate(RAW_COLUMNS):
            default = options.index(auto[req]) if auto[req] in options else 0
            choice = cols[i % 3].selectbox(req, options, index=default, key=f"map_{uid}_{req}")
            mapping[req] = None if choice == "(none)" else choice

    action = st.radio("What do you want to do?", ["A) Analyse my data and train a new model", "B) Score my customers with the loaded model"],
                      key="upload_action")
    mode = "train" if action.startswith("A") else "score"
    prepared, rep = prepare_upload(raw, mapping, mode)
    if len(rep["missing_columns"]) > 8:
        st.info("This file does not look like the telecom layout (most required columns are missing). "
                "Choose **Any dataset** above to analyse it with your own outcome column instead.")
    for e in rep["errors"]:
        st.error(e)
    for w in rep["warnings"]:
        st.warning(w)
    if prepared is None:
        return
    st.caption(f"Validation passed: {rep['n_rows_used']:,} rows will be used.")
    if mode == "train":
        profile = st.selectbox("Training depth", list(TRAINING_PROFILES), index=0, key="profile",
                               help="Deeper search can improve the model slightly but takes much longer on a shared server.")
        st.caption("The full pipeline runs: cleaning, quality report, EDA, statistical tests, feature engineering, tuning four models, "
                   "evaluation on a hold-out test set, explainability, recommendations and reports.")
        if st.button("Run full analysis on my data", key="run_train", type="primary"):
            cfg, _ws = create_workspace(profile, BASE_CFG)
            bar = st.progress(0.0, text="Queued ...")

            def cb(step, i, n):
                bar.progress(i / n, text=f"Step {i + 1} of {n}: {STEP_LABELS[step]}")
            try:
                timings = run_workspace_pipeline(cfg, prepared, cb)
            except Exception as exc:  # noqa: BLE001
                st.error(f"The analysis failed: {type(exc).__name__}: {exc}")
                return
            bar.progress(1.0, text="Done")
            st.session_state["workspace_cfg"] = cfg
            st.session_state["workspace_meta"] = {"name": up.name, "rows": rep["n_rows_used"], "profile": profile,
                                                   "seconds": round(sum(timings.values()))}
            st.session_state["_goto"] = "Executive Summary"
            st.cache_data.clear()
            st.rerun()
    else:
        thr = st.slider("Decision threshold", 0.05, 0.95, float(round(THR, 2)), 0.01, key="batch_thr",
                        help="Customers with churn probability at or above this are flagged.")
        if st.button("Score my customers", key="run_score", type="primary"):
            with st.spinner("Scoring ..."):
                table_, revenue_ = score_batch(prepared, B, CFG, thr)
            st.session_state["batch"] = {"table": table_, "revenue": revenue_, "name": up.name, "threshold": thr}
    _show_batch_results()


def _show_batch_results():
    res = st.session_state.get("batch")
    if not res:
        return
    t, r = res["table"], res["revenue"]
    st.divider()
    st.subheader(f"Scoring results for {res['name']}")
    kpi_row([("Customers scored", f"{len(t):,}"), ("Flagged high-risk", f"{r['flagged_customers']:,}"),
             ("Revenue at risk / month", format_currency(r["monthly_revenue_at_risk"], CUR, True)),
             ("Revenue at risk / year", format_currency(r["annual_revenue_at_risk"], CUR, True)),
             ("Critical-risk customers", f"{r['critical_customers']:,}")])
    known = t[t["Churn"].isin(["Yes", "No"])]
    if len(known) >= 30 and known["Churn"].nunique() == 2:
        m = compute_metrics((known["Churn"] == "Yes").astype(int), known["churn_probability"], res["threshold"])
        st.info(f"Your file included churn labels, so the model could be checked on it: ROC-AUC {m['roc_auc']:.3f}, recall "
                f"{m['recall']:.3f}, precision {m['precision']:.3f} at threshold {res['threshold']:.2f}.")
    c1, c2 = st.columns(2)
    with c1:
        order = ["Low", "Medium", "High", "Critical"]
        cnt = t["risk_tier"].value_counts().reindex(order).fillna(0)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.bar(cnt.index, cnt.values, color=[RISK_COLORS[x] for x in cnt.index])
        ax.set_title("Customers by risk tier")
        show_fig(fig)
    with c2:
        top = t[t["flagged"]]["primary_action"].value_counts().head(6)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(top.index[::-1], top.values[::-1], color=ACCENT)
        ax.set_title("Top recommended actions (flagged customers)")
        show_fig(fig)
    cols = ["customerID", "churn_probability", "risk_tier", "priority_score", "MonthlyCharges", "tenure", "Contract",
            "primary_action", "risk_factors"]
    page_df, _ = paginated(t[cols], "batch", "priority_score", ["customerID", "primary_action", "Contract"])
    st.dataframe(friendly_columns(page_df), hide_index=True)
    st.download_button("Download all scored customers (CSV)", to_csv_bytes(t[cols + ["all_actions", "annual_revenue_at_risk"]]),
                       "scored_customers.csv", key="dl_batch_all")
    st.download_button("Download flagged priority list (CSV)",
                       to_csv_bytes(t[t["flagged"]][cols].sort_values("priority_score", ascending=False)),
                       "priority_customers.csv", key="dl_batch_prio")


{"Data Upload": page_upload, "Dataset Analysis": (page_generic if GEN else page_upload), "Executive Summary": page_exec, "Data Quality": page_quality, "Exploratory Analysis": page_eda,
 "Statistical Insights": page_stats, "Customer Explorer": page_explorer, "Churn Prediction": page_predict,
 "Prediction History": page_history, "Model Performance": page_performance, "Feature Importance": page_importance,
 "Retention Recommendations": page_recs, "Revenue at Risk": page_revenue}[page]()
