"""Exploratory data analysis: univariate, bivariate and multivariate analysis with saved charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import load_config, resolve_path
from .features import engineer_features
from .schema import TARGET, TARGET_FLAG, column_label
from .stats_tests import cramers_v
from .utils import ACCENT, BLUE, PALETTE, df_to_markdown, get_logger, now_iso, save_figure, save_json, set_plot_style

log = get_logger()
KEY_BIVARIATE = ["Contract", "tenure_group", "InternetService", "PaymentMethod", "SeniorCitizen", "gender",
                 "TechSupport", "OnlineSecurity", "Partner", "Dependents", "PaperlessBilling",
                 "customer_value_segment"]
EDA_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "total_services"]


def churn_rate_by(df: pd.DataFrame, col: str, order: list | None = None) -> pd.DataFrame:
    g = df.groupby(col, observed=True)[TARGET_FLAG].agg(customers="size", churned="sum").reset_index()
    g["churn_rate"] = g["churned"] / g["customers"]
    if order:
        g[col] = pd.Categorical(g[col], categories=[o for o in order if o in set(g[col])], ordered=True)
        g = g.sort_values(col)
    else:
        g = g.sort_values("churn_rate", ascending=False)
    return g.reset_index(drop=True)


def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Variance inflation factor via OLS R^2 (numpy only)."""
    X = df[cols].dropna().to_numpy(float)
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    rows = []
    for j, c in enumerate(cols):
        y, others = X[:, j], np.delete(X, j, axis=1)
        A = np.c_[np.ones(len(y)), others]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss_res = ((y - A @ beta) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rows.append({"feature": c, "vif": float(1 / max(1 - r2, 1e-6))})
    out = pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)
    out["concern"] = np.where(out["vif"] > 10, "severe (>10)", np.where(out["vif"] > 5, "moderate (>5)", "ok"))
    return out


def _bar_churn(ax, g: pd.DataFrame, col: str, overall: float, rotate: int = 0):
    sns.barplot(data=g, x=col, y="churn_rate", color=BLUE, ax=ax)
    for i, r in enumerate(g.itertuples()):
        ax.text(i, r.churn_rate + 0.01, f"{100 * r.churn_rate:.1f}%\n(n={r.customers:,})", ha="center", va="bottom",
                fontsize=8)
    ax.axhline(overall, color=ACCENT, ls="--", lw=1.2, label=f"Overall {100 * overall:.1f}%")
    ax.set_ylim(0, min(1.0, g["churn_rate"].max() * 1.35 + 0.05))
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_xlabel(column_label(col))
    ax.set_ylabel("Churn rate")
    ax.tick_params(axis="x", rotation=rotate)
    ax.legend(loc="upper right", fontsize=8)


def run_eda(clean: pd.DataFrame, cfg: dict | None = None, save: bool = True) -> dict:
    cfg = cfg or load_config()
    set_plot_style()
    df = engineer_features(clean, cfg)
    fdir = resolve_path(cfg, "figures_dir", create=True)
    tdir = resolve_path(cfg, "tables_dir", create=True)
    rdir = resolve_path(cfg, "reports_dir", create=True)
    overall = float(df[TARGET_FLAG].mean())
    figures: list[str] = []
    tables: dict[str, pd.DataFrame] = {}

    def keep(fig, name):
        if save:
            figures.append(str(save_figure(fig, fdir / name)))
        else:
            plt.close(fig)

    # ---- univariate -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vc = df[TARGET].value_counts().reindex(["No", "Yes"])
    sns.barplot(x=vc.index, y=vc.values, hue=vc.index, palette=PALETTE, legend=False, ax=axes[0])
    for i, v in enumerate(vc.values):
        axes[0].text(i, v, f"{v:,}\n({100 * v / len(df):.1f}%)", ha="center", va="bottom")
    axes[0].set_title("Customers by churn status"); axes[0].set_xlabel("Churned"); axes[0].set_ylabel("Customers")
    axes[1].pie(vc.values, labels=["Retained", "Churned"], colors=[BLUE, ACCENT], autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Class balance")
    keep(fig, "eda_01_target_distribution.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), EDA_NUMERIC + ["charge_delta"]):
        sns.histplot(df[c], bins=30, kde=True, color=BLUE, ax=ax)
        ax.set_title(f"Distribution: {column_label(c)}")
        ax.set_xlabel(column_label(c))
    fig.suptitle("Univariate analysis: numeric variables", fontweight="bold")
    keep(fig, "eda_02_numeric_distributions.png")

    cat_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "InternetService", "Contract",
                "PaymentMethod", "PaperlessBilling"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, c in zip(axes.ravel(), cat_cols):
        order = df[c].value_counts().index
        sns.countplot(data=df, x=c, order=order, color=BLUE, ax=ax)
        ax.set_title(column_label(c)); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Univariate analysis: customer mix", fontweight="bold")
    keep(fig, "eda_03_categorical_counts.png")

    # ---- bivariate: churn by key categoricals ---------------------------------------------------
    orders = {"tenure_group": cfg["features"]["tenure_labels"],
              "customer_value_segment": cfg["features"]["value_labels"],
              "Contract": ["Month-to-month", "One year", "Two year"]}
    for i, c in enumerate(["Contract", "tenure_group", "InternetService", "PaymentMethod", "SeniorCitizen",
                           "gender", "TechSupport", "OnlineSecurity"], start=4):
        g = churn_rate_by(df, c, orders.get(c))
        tables[f"churn_by_{c}"] = g
        fig, ax = plt.subplots(figsize=(8 if c != "PaymentMethod" else 10, 4.5))
        _bar_churn(ax, g, c, overall, rotate=15 if c == "PaymentMethod" else 0)
        ax.set_title(f"Churn rate by {column_label(c).lower()}")
        keep(fig, f"eda_{i:02d}_churn_by_{c}.png")
    for c in KEY_BIVARIATE:
        tables.setdefault(f"churn_by_{c}", churn_rate_by(df, c, orders.get(c)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, c in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.boxplot(data=df, x=TARGET, y=c, hue=TARGET, palette=PALETTE, order=["No", "Yes"], legend=False, ax=ax)
        ax.set_title(f"{column_label(c)} by churn")
    keep(fig, "eda_12_numeric_by_churn_boxplots.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, c in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.kdeplot(data=df, x=c, hue=TARGET, palette=PALETTE, common_norm=False, fill=True, alpha=.35, ax=ax)
        ax.set_title(f"{column_label(c)}: churned vs retained")
    keep(fig, "eda_13_numeric_by_churn_density.png")

    # ---- multivariate ---------------------------------------------------------------------------
    sample = df.sample(min(len(df), 2500), random_state=cfg["random_seed"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=sample, x="tenure", y="MonthlyCharges", hue=TARGET, palette=PALETTE, alpha=.55, s=22, ax=ax)
    ax.set_title("Tenure vs monthly charges (colour = churn)")
    keep(fig, "eda_14_tenure_vs_monthly_scatter.png")

    num_cols = EDA_NUMERIC + ["charge_delta", "protection_count", "streaming_count", TARGET_FLAG]
    corr = df[num_cols].corr()
    tables["correlation_matrix"] = corr
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title("Correlation matrix (numeric variables and churn)")
    keep(fig, "eda_15_correlation_heatmap.png")

    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(num_cols[:-1]) for b in num_cols[i + 1:-1]
             if abs(corr.loc[a, b]) >= 0.8]
    # charge_delta is an exact linear combination (MonthlyCharges - avg_monthly_spend), so it is excluded
    vif = compute_vif(df, EDA_NUMERIC + ["protection_count", "streaming_count"])
    tables["vif"] = vif
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=vif, y="feature", x="vif", color=BLUE, ax=ax)
    ax.axvline(5, color="orange", ls="--", label="VIF=5"); ax.axvline(10, color=ACCENT, ls="--", label="VIF=10")
    ax.set_title("Multicollinearity check (variance inflation factor)"); ax.legend()
    keep(fig, "eda_16_vif.png")

    assoc = []
    for c in ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
              "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
              "tenure_group", "customer_value_segment", "service_profile"]:
        t = pd.crosstab(df[c], df[TARGET])
        assoc.append({"variable": c, "label": column_label(c), "cramers_v": cramers_v(t)})
    assoc = pd.DataFrame(assoc).sort_values("cramers_v", ascending=False)
    tables["association_with_churn"] = assoc
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.barplot(data=assoc, y="label", x="cramers_v", color=BLUE, ax=ax)
    ax.set_title("Strength of association with churn (Cramer's V)"); ax.set_xlabel("Cramer's V"); ax.set_ylabel("")
    keep(fig, "eda_17_cramers_v.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (r, c) in zip(axes, [("Contract", "InternetService"), ("tenure_group", "Contract"),
                                 ("PaymentMethod", "Contract")]):
        pv = df.pivot_table(index=r, columns=c, values=TARGET_FLAG, aggfunc="mean", observed=True)
        if r == "tenure_group":
            pv = pv.reindex(cfg["features"]["tenure_labels"])
        sns.heatmap(pv, annot=True, fmt=".0%", cmap="OrRd", ax=ax, vmin=0, vmax=max(.6, float(np.nanmax(pv.values))))
        ax.set_title(f"Churn rate: {column_label(r)} x {column_label(c)}")
    keep(fig, "eda_18_multivariate_heatmaps.png")

    sv = df.groupby("total_services")[TARGET_FLAG].agg(["mean", "size"]).reset_index()
    tables["churn_by_total_services"] = sv.rename(columns={"mean": "churn_rate", "size": "customers"})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=sv, x="total_services", y="mean", marker="o", color=ACCENT, ax=ax)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_title("Churn rate by number of services"); ax.set_ylabel("Churn rate"); ax.set_xlabel("Services used")
    keep(fig, "eda_19_churn_by_services.png")

    ten = df.groupby("tenure")[TARGET_FLAG].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(data=ten, x="tenure", y=TARGET_FLAG, color=ACCENT, ax=ax)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_title("Churn rate by tenure month"); ax.set_ylabel("Churn rate"); ax.set_xlabel("Tenure (months)")
    keep(fig, "eda_20_churn_by_tenure_month.png")

    insights = build_insights(df, tables, pairs, vif, overall)
    result = {"overall_churn_rate": overall, "insights": insights, "figures": figures,
              "high_correlation_pairs": [(a, b, float(r)) for a, b, r in pairs]}
    if save:
        for name, t in tables.items():
            t.to_csv(tdir / f"eda_{name}.csv", index=name == "correlation_matrix")
        save_json({"generated_at": now_iso(), **{k: v for k, v in result.items() if k != "figures"}},
                  tdir / "eda_insights.json")
        (rdir / "eda_summary.md").write_text(eda_markdown(result, tables), encoding="utf-8")
        log.info("EDA: %d figures saved to %s", len(figures), fdir)
    result["tables"] = tables
    return result


def build_insights(df, tables, pairs, vif, overall) -> list[str]:
    out = [f"Overall churn rate is {100 * overall:.1f}% ({int(df[TARGET_FLAG].sum()):,} of {len(df):,} customers); "
           f"the classes are imbalanced, so accuracy alone is misleading."]

    def hi_lo(col, label):
        g = tables[f"churn_by_{col}"]
        g = g[(g["customers"] >= 30) & (g[col].astype(str) != "No internet service")]
        g = g.sort_values("churn_rate", ascending=False)
        if len(g) < 2:
            return
        h, l = g.iloc[0], g.iloc[-1]
        out.append(f"{label}: '{h[col]}' customers churn at {100 * h.churn_rate:.1f}% versus "
                   f"{100 * l.churn_rate:.1f}% for '{l[col]}'.")

    for col, label in [("Contract", "Contract type"), ("InternetService", "Internet service"),
                       ("PaymentMethod", "Payment method"), ("tenure_group", "Tenure"),
                       ("TechSupport", "Tech support"), ("OnlineSecurity", "Online security"),
                       ("SeniorCitizen", "Senior citizen status"), ("PaperlessBilling", "Paperless billing"),
                       ("customer_value_segment", "Customer value segment")]:
        hi_lo(col, label)
    g = tables["churn_by_gender"]
    out.append(f"Gender shows little difference in churn ({', '.join(f'{r.gender} {100 * r.churn_rate:.1f}%' for r in g.itertuples())}).")
    a = df.groupby(TARGET)[["tenure", "MonthlyCharges"]].median()
    if {"Yes", "No"} <= set(a.index):
        out.append(f"Churned customers have a median tenure of {a.loc['Yes', 'tenure']:.0f} months vs "
                   f"{a.loc['No', 'tenure']:.0f} for retained, and median monthly charges of "
                   f"${a.loc['Yes', 'MonthlyCharges']:.2f} vs ${a.loc['No', 'MonthlyCharges']:.2f}.")
    if pairs:
        out.append("High correlation (|r| >= 0.8) between predictors: " +
                   "; ".join(f"{column_label(a)} & {column_label(b)} (r={r:.2f})" for a, b, r in pairs) +
                   ". These are redundant for linear models; regularisation / tree models are used.")
    sev = vif[vif["vif"] > 5]
    if len(sev):
        out.append("Multicollinearity (VIF > 5): " + ", ".join(f"{column_label(r.feature)} ({r.vif:.1f})"
                                                                for r in sev.itertuples()) + ".")
    return out


def eda_markdown(result: dict, tables: dict) -> str:
    lines = ["# Exploratory Data Analysis Summary", "", "## Key business insights", "",
             *[f"- {i}" for i in result["insights"]], "",
             "## Strength of association with churn (Cramer's V)",
             df_to_markdown(tables["association_with_churn"][["label", "cramers_v"]], 3), "",
             "## Multicollinearity (VIF)", df_to_markdown(tables["vif"], 2), "",
             "## Charts", *[f"- `{p.split('outputs/')[-1]}`" for p in result["figures"]], ""]
    return "\n".join(lines)
