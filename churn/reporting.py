"""Report generation: final business-insights report as Markdown, print-friendly HTML and PDF."""
from __future__ import annotations

import base64
import html
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .config import load_config, resolve_path
from .utils import df_to_markdown, get_logger, load_json, now_iso

log = get_logger()


def _money(x, cur="$") -> str:
    return f"{cur}{x:,.0f}"


def _read(path: Path, default=None):
    return pd.read_csv(path) if path.exists() else default


def collect_report_data(cfg: dict) -> dict:
    t, m = resolve_path(cfg, "tables_dir"), resolve_path(cfg, "models_dir")
    d = {"revenue": load_json(t / "revenue_at_risk_summary.json") if (t / "revenue_at_risk_summary.json").exists() else None,
         "segments": _read(t / "insights_segments.csv"), "recs": _read(t / "insights_recommendations.csv"),
         "priority": _read(t / "priority_customers_top500.csv"), "by_contract": _read(t / "insights_by_contract.csv"),
         "group_imp": _read(t / "explain_group_importance.csv"), "value_effects": _read(t / "explain_value_effects.csv"),
         "comparison": _read(t / "model_comparison.csv"), "chi": _read(t / "stats_chi_square.csv"),
         "groups": _read(t / "stats_group_comparisons.csv"),
         "eda": load_json(t / "eda_insights.json") if (t / "eda_insights.json").exists() else {"insights": []},
         "metrics": load_json(m / "metrics_test.json") if (m / "metrics_test.json").exists() else
         (load_json(m / "metrics.json")["test"] if (m / "metrics.json").exists() else None),
         "model_meta": load_json(m / "metrics.json") if (m / "metrics.json").exists() else {}}
    return d


def business_insights_markdown(cfg: dict, d: dict) -> str:
    cur = cfg["business"]["currency"]
    rev, met = d["revenue"], d["metrics"]
    if rev is None or met is None:
        raise FileNotFoundError("Run training and insights before generating reports.")
    meta = d["model_meta"]
    L = ["# Customer Churn - Business Insights Report", f"_Generated {now_iso()} | model: **{meta.get('model', '?')}** | "
         f"dataset version `{meta.get('dataset_version', '?')}`_", "",
         "## 1. Executive summary",
         f"- The model flags **{rev['flagged_customers']:,} of {rev['customers']:,} customers "
         f"({100 * rev['flagged_share']:.1f}%)** as likely to churn at a probability threshold of {rev['threshold']:.2f}.",
         f"- **Revenue at risk:** {_money(rev['monthly_revenue_at_risk'], cur)} per month, "
         f"**{_money(rev['annual_revenue_at_risk'], cur)} per year** "
         f"({100 * rev['pct_of_monthly_revenue_at_risk']:.1f}% of monthly revenue is held by flagged customers).",
         f"- Probability-weighted expected monthly loss across all customers: {_money(rev['expected_monthly_loss_all_customers'], cur)}.",
         f"- On the untouched test set the model reaches ROC-AUC {met['roc_auc']:.3f}, recall {met['recall']:.3f}, "
         f"precision {met['precision']:.3f}, F1 {met['f1']:.3f}.",
         f"- Illustrative campaign economics (assumes {100 * rev['assumed_save_rate']:.0f}% of true churners are "
         f"saved and {cur}{rev['assumed_offer_cost']:.0f} per offer): cost {_money(rev['campaign_cost'], cur)}, "
         f"retained revenue about {_money(rev['estimated_annual_revenue_retained'], cur)}/year, "
         f"net benefit about {_money(rev['estimated_net_benefit'], cur)}.", "",
         "## 2. The five business questions", ""]
    top = d["priority"].head(5) if d["priority"] is not None else pd.DataFrame()
    L += ["### Which customers are most likely to churn?",
          "The priority list (ranked by a score combining churn probability, revenue, early tenure and number of "
          "risk factors) is in `outputs/tables/priority_customers_top500.csv`. Top 5:", "",
          df_to_markdown(top[["customerID", "churn_probability", "risk_tier", "MonthlyCharges", "tenure", "priority_score",
                              "primary_action"]]) if len(top) else "_n/a_", ""]
    L += ["### Which factors drive churn?"]
    if d["group_imp"] is not None:
        gi = d["group_imp"].head(5)
        L += ["Business drivers ranked by permutation importance on the test set:", "",
              df_to_markdown(gi[["driver", "importance_mean", "mean_abs_effect_pp"]], 4), ""]
        ve = d["value_effects"]
        if ve is not None:
            for drv in gi["driver"].head(3):
                s = ve[ve["driver"] == drv].sort_values("mean_effect_pp", ascending=False)
                if len(s):
                    L.append(f"- **{drv}** - highest risk: {s.iloc[0]['value']} ({s.iloc[0]['mean_effect_pp']:+.1f} pp); "
                             f"lowest risk: {s.iloc[-1]['value']} ({s.iloc[-1]['mean_effect_pp']:+.1f} pp).")
        L.append("")
    L += [*[f"- {i}" for i in d["eda"]["insights"][:6]], "",
          "_Drivers describe associations learned by the model, not proven causes._", "",
          "### How much monthly and annual revenue is at risk?",
          f"{_money(rev['monthly_revenue_at_risk'], cur)} per month / {_money(rev['annual_revenue_at_risk'], cur)} per year "
          f"among flagged customers; {_money(rev['critical_monthly_revenue'], cur)} per month sits with the "
          f"{rev['critical_customers']:,} 'Critical' customers (probability >= {cfg['threshold']['critical_probability']:.0%}).", ""]
    if d["by_contract"] is not None:
        bc = d["by_contract"][["Contract", "customers", "flagged", "monthly_revenue_at_risk", "annual_revenue_at_risk"]]
        L += [df_to_markdown(bc, 0), ""]
    L += ["### Which customer segments need urgent retention action?"]
    if d["segments"] is not None:
        s = d["segments"].head(8)[["segment", "customers", "flagged_pct", "avg_churn_probability",
                                    "monthly_revenue_at_risk", "urgency", "top_recommended_action"]]
        L += [df_to_markdown(s, 2), ""]
    L += ["### What retention actions should be prioritised?"]
    if d["recs"] is not None:
        L += [df_to_markdown(d["recs"][["recommended_action", "risk_factor", "impact", "flagged_customers_affected",
                                         "monthly_revenue_covered"]], 0), "",
              "Actions are rule-based (contract, tenure, price, payment method, services, risk tier) and each "
              "customer gets one primary action plus any secondary ones; see `customer_recommendations.csv`.", ""]
    L += ["## 3. Statistical evidence (association, not causation)"]
    if d["groups"] is not None:
        L += [f"- {t}" for t in d["groups"]["interpretation"].head(6)]
    L += ["", "## 4. Assumptions and limitations",
          "- Scores for training customers are out-of-fold predictions and for test customers are hold-out predictions, "
          "so no customer is scored by a model that saw their own label.",
          "- Revenue figures use MonthlyCharges as revenue; margins, discounts and customer lifetime value are not modelled.",
          "- Save rate and offer cost are planning assumptions in `config.yaml`; validate them with an A/B retention test.",
          "- The IBM Telco data is a single snapshot of 7,043 customers; model drift should be monitored and the model "
          "retrained regularly. `gender` is excluded from the model as a fairness precaution.", ""]
    return "\n".join(L)


def _inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return re.sub(r"(?<![\w*])_(.+?)_(?![\w*])", r"<em>\1</em>", s)


def markdown_to_html(md: str, images: list[Path] | None = None, title: str = "Churn report") -> str:
    out, i, lines = [], 0, md.split("\n")
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("| ---"):
            hdr = [c.strip() for c in ln.strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline(h)}</th>" for h in hdr) + "</tr></thead><tbody>" +
                       "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows) +
                       "</tbody></table>")
            continue
        if ln.startswith("# "):
            out.append(f"<h1>{_inline(ln[2:])}</h1>")
        elif ln.startswith("## "):
            out.append(f"<h2>{_inline(ln[3:])}</h2>")
        elif ln.startswith("### "):
            out.append(f"<h3>{_inline(ln[4:])}</h3>")
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif ln.strip():
            out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    figs = ""
    for p in images or []:
        if Path(p).exists():
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            figs += f'<figure><img src="data:image/png;base64,{b64}" alt="{Path(p).stem}"><figcaption>{Path(p).stem.replace("_", " ")}</figcaption></figure>'
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.5}"
           "h1{border-bottom:3px solid #E4572E;padding-bottom:.3rem}h2{color:#1f3b57;margin-top:2rem}"
           "table{border-collapse:collapse;width:100%;font-size:.8rem;margin:.8rem 0}th,td{border:1px solid #ccc;padding:4px 6px;text-align:left}"
           "th{background:#f2f5f9}code{background:#f4f4f4;padding:1px 4px}figure{page-break-inside:avoid;margin:1rem 0}"
           "img{max-width:100%}figcaption{font-size:.8rem;color:#666}@media print{body{margin:0;max-width:none}}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head>"
            f"<body>{''.join(out)}<h2>Charts</h2>{figs}</body></html>")


def build_pdf(cfg: dict, d: dict, path: Path, figure_paths: list[Path]) -> Path:
    cur = cfg["business"]["currency"]
    rev, met = d["revenue"], d["metrics"]
    A4 = (8.27, 11.69)
    with plt.rc_context({"text.parse_math": False}), PdfPages(path) as pdf:  # '$' must not trigger mathtext
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.94, "Customer Churn Prediction", fontsize=22, fontweight="bold", color="#1f3b57")
        fig.text(0.07, 0.915, f"Business insights report  |  {now_iso()[:10]}  |  model: {d['model_meta'].get('model', '?')}",
                 fontsize=9, color="grey")
        kpis = [("Customers", f"{rev['customers']:,}"), ("Flagged high-risk", f"{rev['flagged_customers']:,}"),
                ("Revenue at risk / month", _money(rev["monthly_revenue_at_risk"], cur)),
                ("Revenue at risk / year", _money(rev["annual_revenue_at_risk"], cur)),
                ("ROC-AUC (test)", f"{met['roc_auc']:.3f}"), ("Recall (test)", f"{met['recall']:.3f}")]
        for k, (lab, val) in enumerate(kpis):
            x, y = 0.07 + (k % 3) * 0.31, 0.82 - (k // 3) * 0.1
            fig.patches.append(plt.Rectangle((x, y), 0.28, 0.08, transform=fig.transFigure, fc="#f2f5f9", ec="#c9d3e0"))
            fig.text(x + 0.015, y + 0.045, val, fontsize=15, fontweight="bold", color="#E4572E")
            fig.text(x + 0.015, y + 0.012, lab, fontsize=8, color="#444")
        body = ["Key findings"] + [f"- {t}" for t in d["eda"]["insights"][:7]] + [
            "", f"- Precision {met['precision']:.3f}, F1 {met['f1']:.3f} at threshold {met['threshold']:.2f}.",
            f"- Estimated campaign net benefit: {_money(rev['estimated_net_benefit'], cur)} per year "
            f"(save rate {rev['assumed_save_rate']:.0%}, offer cost {cur}{rev['assumed_offer_cost']:.0f}).",
            "", "Statistical associations are not proof of causation."]
        y = 0.62
        for ln in body:
            for w in textwrap.wrap(ln, 95) or [""]:
                fig.text(0.07, y, w, fontsize=9.5, fontweight="bold" if ln == "Key findings" else "normal")
                y -= 0.022
        pdf.savefig(fig); plt.close(fig)

        def table_page(title, df, widths=None, fs=7):
            f2 = plt.figure(figsize=A4)
            f2.text(0.07, 0.95, title, fontsize=14, fontweight="bold", color="#1f3b57")
            ax = f2.add_axes([0.04, 0.45, 0.92, 0.45]); ax.axis("off")
            cell = df.copy()
            for c in cell.columns:
                cell[c] = cell[c].map(lambda v: f"{v:,.3f}" if isinstance(v, (float, np.floating)) else textwrap.shorten(str(v), 38))
            tb = ax.table(cellText=cell.values, colLabels=list(cell.columns), loc="upper center", cellLoc="left")
            tb.auto_set_font_size(False); tb.set_fontsize(fs); tb.scale(1, 1.5)
            for (r, _), c in tb.get_celld().items():
                if r == 0:
                    c.set_facecolor("#1f3b57"); c.set_text_props(color="white", fontweight="bold")
            return f2

        if d["comparison"] is not None:
            cmp = d["comparison"][["display_name", "cv_roc_auc", "cv_pr_auc", "cv_recall", "cv_precision", "cv_f1", "selected"]]
            f2 = table_page("Model comparison (cross-validation, training data)", cmp)
            f2.text(0.07, 0.40, "\n".join(textwrap.wrap(d["model_meta"].get("model", ""), 90)), fontsize=1, alpha=0)
            pdf.savefig(f2); plt.close(f2)
        if d["segments"] is not None:
            seg = d["segments"].head(10)[["segment", "customers", "flagged_pct", "monthly_revenue_at_risk", "urgency"]]
            f2 = table_page("Segments needing retention action", seg, fs=8)
            pdf.savefig(f2); plt.close(f2)
        if d["recs"] is not None:
            rc = d["recs"][["recommended_action", "impact", "flagged_customers_affected", "monthly_revenue_covered"]]
            f2 = table_page("Prioritised retention actions", rc, fs=7)
            pdf.savefig(f2); plt.close(f2)
        existing = [p for p in figure_paths if Path(p).exists()]
        for k in range(0, len(existing), 2):
            f2 = plt.figure(figsize=A4)
            for j, p in enumerate(existing[k:k + 2]):
                ax = f2.add_axes([0.05, 0.53 - j * 0.48, 0.9, 0.44]); ax.axis("off")
                ax.imshow(plt.imread(p))
            pdf.savefig(f2); plt.close(f2)
    return path


def generate_reports(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    rdir = resolve_path(cfg, "reports_dir", create=True)
    fdir = resolve_path(cfg, "figures_dir")
    d = collect_report_data(cfg)
    md = business_insights_markdown(cfg, d)
    (rdir / "business_insights_report.md").write_text(md, encoding="utf-8")
    parts = [md]
    for name in ["data_quality_report.md", "eda_summary.md", "statistical_report.md", "model_evaluation_report.md",
                 "explainability_report.md"]:
        p = rdir / name
        if p.exists():
            body = p.read_text(encoding="utf-8").replace("\n# ", "\n## ")
            parts.append("\n---\n\n" + re.sub(r"^# ", "## Appendix: ", body, count=1))
    final_md = "\n".join(parts)
    (rdir / "final_report.md").write_text(final_md, encoding="utf-8")
    fig_names = ["eda_04_churn_by_Contract.png", "eda_05_churn_by_tenure_group.png", "eda_15_correlation_heatmap.png",
                 "eval_roc_curve.png", "eval_precision_recall_curve.png", "eval_confusion_matrix.png",
                 "eval_threshold_tradeoff.png", "explain_group_importance.png", "explain_shap_summary.png"]
    figs = [fdir / n for n in fig_names]
    (rdir / "final_report.html").write_text(markdown_to_html(final_md, figs, "Customer Churn - Final Report"),
                                            encoding="utf-8")
    pdf = build_pdf(cfg, d, rdir / "final_report.pdf", figs)
    log.info("Reports written: %s", ", ".join(p.name for p in [rdir / "final_report.md", rdir / "final_report.html", pdf]))
    return {"markdown": rdir / "final_report.md", "business_markdown": rdir / "business_insights_report.md",
            "html": rdir / "final_report.html", "pdf": pdf}
