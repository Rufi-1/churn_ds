"""Runs the real Streamlit script headlessly (streamlit.testing) against the pipeline output."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["Data Upload", "Executive Summary", "Data Quality", "Exploratory Analysis", "Statistical Insights", "Customer Explorer",
         "Churn Prediction", "Prediction History", "Model Performance", "Feature Importance",
         "Retention Recommendations", "Revenue at Risk"]


@pytest.fixture()
def app(pipeline_run, monkeypatch):
    monkeypatch.setenv("CHURN_CONFIG", str(pipeline_run["cfg_file"]))
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_every_page_renders_without_error(app):
    for p in PAGES:
        app.sidebar.radio[0].set_value(p).run()
        assert not app.exception, (p, [str(e.value)[:300] for e in app.exception])
        assert app.title and app.title[0].value == p


def test_prediction_form_end_to_end(app):
    app.sidebar.radio[0].set_value("Churn Prediction").run()
    app.selectbox(key="p_contract").set_value("Month-to-month")
    app.number_input(key="p_tenure").set_value(1)
    [b for b in app.button if "Predict" in b.label][0].click().run()
    assert not app.exception
    prob = [m for m in app.metric if m.label == "Churn probability"][0].value
    assert prob.endswith("%") and float(prob[:-1]) > 50
    app.sidebar.radio[0].set_value("Prediction History").run()
    assert not app.exception and any(m.label == "Predictions" and m.value == "1" for m in app.metric)


def test_explorer_filters_and_search(app):
    app.sidebar.radio[0].set_value("Customer Explorer").run()
    app.multiselect(key="f_contract").set_value(["Two year"]).run()
    assert not app.exception
    app.text_input(key="exp_q").set_value("no-such-customer").run()
    assert not app.exception


def test_missing_artifacts_show_friendly_message(blank_cfg, tmp_path, monkeypatch):
    import yaml
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(blank_cfg))
    monkeypatch.setenv("CHURN_CONFIG", str(f))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and any("pipeline has not produced" in w.value for w in at.warning)


def test_dashboard_runs_on_a_users_private_workspace(pipeline_run, mini_raw, tmp_path, monkeypatch):
    """Simulate a finished upload: the app must show the user's own results from their own workspace."""
    from churn.upload import auto_map_columns, create_workspace, prepare_upload, run_workspace_pipeline
    monkeypatch.setenv("CHURN_CONFIG", str(pipeline_run["cfg_file"]))
    prepared, _ = prepare_upload(mini_raw.head(400), auto_map_columns(mini_raw.columns), "train")
    ws_cfg, _ = create_workspace("Quick (about 1-2 minutes, tiny search)", pipeline_run["cfg"], root=tmp_path / "ws")
    run_workspace_pipeline(ws_cfg, prepared)
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["workspace_cfg"] = ws_cfg
    at.session_state["workspace_meta"] = {"name": "mine.csv", "rows": 400, "profile": "Quick", "seconds": 1}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("YOUR data" in s.value for s in at.sidebar.success)
    assert any(m.label == "Customers" and m.value == "400" for m in at.metric)
    for p in ["Data Quality", "Churn Prediction", "Revenue at Risk"]:
        at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, (p, [str(e.value)[:200] for e in at.exception])


def test_batch_scoring_results_render(app, pipeline_run, mini_raw):
    from churn.predict import load_bundle
    from churn.upload import auto_map_columns, prepare_upload, score_batch
    x = mini_raw.drop(columns=["Churn"])
    prepared, _ = prepare_upload(x, auto_map_columns(x.columns), "score")
    table, rev = score_batch(prepared, load_bundle(pipeline_run["cfg"]), pipeline_run["cfg"])
    app.session_state["batch"] = {"table": table, "revenue": rev, "name": "x.csv", "threshold": 0.45}
    app.sidebar.radio[0].set_value("Data Upload").run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    assert any(m.label == "Customers scored" and m.value == "500" for m in app.metric)


def test_any_dataset_mode_renders(app, generic_run):
    app.session_state["generic"] = generic_run
    app.run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    assert [r.options for r in app.sidebar.radio][0] == ["Data Upload", "Dataset Analysis"]
    assert any("YOUR dataset" in s.value for s in app.sidebar.success)
    assert app.title[0].value == "Dataset Analysis"
    [b for b in app.button if b.label == "Predict"][0].click().run()
    assert not app.exception, [str(e.value)[:300] for e in app.exception]
    assert any(m.label == "Probability" for m in app.metric)
    app.sidebar.radio[0].set_value("Data Upload").run()
    assert not app.exception
