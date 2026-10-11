import pytest

from churn.config import resolve_path
from churn.reporting import business_insights_markdown, collect_report_data, generate_reports, markdown_to_html


def test_reports_exist(cfg):
    r = resolve_path(cfg, "reports_dir")
    for f in ["final_report.md", "final_report.html", "final_report.pdf", "business_insights_report.md",
              "eda_summary.md", "statistical_report.md", "model_evaluation_report.md", "explainability_report.md"]:
        assert (r / f).exists() and (r / f).stat().st_size > 200, f
    assert (r / "final_report.pdf").read_bytes()[:5] == b"%PDF-"


def test_business_report_answers_the_questions(cfg):
    md = business_insights_markdown(cfg, collect_report_data(cfg))
    for q in ["Which customers are most likely to churn?", "Which factors drive churn?",
              "How much monthly and annual revenue is at risk?", "Which customer segments need urgent retention action?",
              "What retention actions should be prioritised?"]:
        assert q in md
    assert "association" in md.lower() and "Revenue at risk" in md


def test_report_numbers_match_source_tables(cfg):
    d = collect_report_data(cfg)
    md = business_insights_markdown(cfg, d)
    assert f"{d['revenue']['annual_revenue_at_risk']:,.0f}" in md
    assert f"{d['revenue']['flagged_customers']:,}" in md


def test_markdown_to_html_renders_tables_and_escapes():
    html = markdown_to_html("# T\n\n| a | b |\n| --- | --- |\n| 1 | <x> |\n\n- **bold** item", [], "t")
    assert "<table>" in html and "&lt;x&gt;" in html and "<strong>bold</strong>" in html and "<h1>" in html


def test_missing_inputs_raise_clear_error(blank_cfg):
    with pytest.raises(FileNotFoundError):
        generate_reports(blank_cfg)
