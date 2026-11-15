import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest
from bike.app import EXAMPLE, validate, create_app
from bike.data import ORIGINAL_COLUMNS, load_data, validate_and_rename
from bike.model import fit_model, predict

@pytest.fixture
def scenario():
    return dict(date="2018-10-15", hour="18", temperature="18", humidity="55", wind_speed="1.5", visibility="1800", dew_point="9", solar="0.2", rainfall="0", snowfall="0", operating="Yes", holiday="No Holiday")

@pytest.mark.parametrize("field,value", [("hour", "24"), ("hour", "1.5"), ("humidity", "101"), ("temperature", "nan"), ("rainfall", "-1"), ("operating", "maybe"), ("date", "bad"), ("dew_point", "30")])
def test_bad_inputs(scenario, field, value):
    scenario[field] = value
    with pytest.raises(ValueError):
        validate(scenario)

def test_calendar(scenario):
    frame = validate(scenario)
    assert frame.weekday.iloc[0] == 0
    assert frame.season.iloc[0] == "Autumn"

def test_baseline_and_closure():
    frame = pd.DataFrame(dict(hour=[0, 0, 0], rentals=[10, 30, 0], operating=["Yes", "Yes", "No"]))
    model = fit_model("Hourly mean", None, frame)
    assert np.array_equal(predict(model, frame), [20, 20, 0])

def test_saved_pipeline_and_interface(scenario):
    import joblib
    from bike.data import ROOT
    model = joblib.load(ROOT / "artifacts/model.joblib")
    expected = round(float(predict(model, validate(scenario))[0]))
    client = create_app().test_client()
    assert client.get("/").status_code == 200
    response = client.post("/", data=scenario)
    assert f'class="number">{expected}<' in response.text
    scenario["operating"] = "No"
    assert 'class="number">0<' in client.post("/", data=scenario).text
    scenario["humidity"] = "200"
    assert 'role="alert"' in client.post("/", data=scenario).text

def test_data_split_integrity():
    frame = load_data()
    assert frame.iloc[6131].timestamp < frame.iloc[6132].timestamp
    assert frame.iloc[7439].timestamp < frame.iloc[7440].timestamp
    assert len(frame) == 8760

def _header_frame(columns):
    return pd.DataFrame({name: [0] for name in columns})

def test_original_columns_accepted():
    renamed = validate_and_rename(_header_frame(ORIGINAL_COLUMNS))
    assert list(renamed.columns) == ["date", "rentals", "hour", "temperature", "humidity", "wind_speed", "visibility", "dew_point", "solar", "rainfall", "snowfall", "season", "holiday", "operating"]

def test_original_columns_missing_rejected():
    columns = [c for c in ORIGINAL_COLUMNS if c != "Rainfall(mm)"]
    with pytest.raises(ValueError, match="Missing"):
        validate_and_rename(_header_frame(columns))

def test_original_columns_unexpected_rejected():
    with pytest.raises(ValueError, match="Unexpected"):
        validate_and_rename(_header_frame([*ORIGINAL_COLUMNS, "Extra column"]))

def test_original_columns_wrong_order_rejected():
    columns = list(ORIGINAL_COLUMNS)
    columns[0], columns[1] = columns[1], columns[0]
    with pytest.raises(ValueError, match="wrong order"):
        validate_and_rename(_header_frame(columns))

def test_initial_state_explains_submit():
    client = create_app().test_client()
    response = client.get("/")
    assert response.status_code == 200
    # "—" on first load is the intended empty state, not a failed prediction.
    assert 'class="number">—<' in response.text
    assert "No estimate yet" in response.text

def test_example_scenario_loads_without_predicting():
    client = create_app().test_client()
    response = client.get("/?example=1")
    assert response.status_code == 200
    assert "Example scenario loaded" in response.text
    assert 'class="number">—<' in response.text
    for field, value in EXAMPLE.items():
        assert f'value="{value}"' in response.text or f">{value}<" in response.text

def test_changed_inputs_change_prediction(scenario):
    client = create_app().test_client()
    first = client.post("/", data=scenario).text
    scenario["temperature"] = "30"
    scenario["hour"] = "8"
    second = client.post("/", data=scenario).text
    first_value = int(re.search(r'class="number">(\d+)<', first).group(1))
    second_value = int(re.search(r'class="number">(\d+)<', second).group(1))
    assert first_value != second_value
    # Submitted values stay visible after a prediction.
    assert 'value="30"' in second and 'value="8"' in second

def test_metrics_table_matches_saved_metrics():
    from bike.data import ROOT
    summary = json.loads((ROOT / "reports/metrics.json").read_text())
    response = create_app().test_client().get("/")
    for scores in summary["models"].values():
        assert f"{scores['test']['mae']:.1f}" in response.text
        assert f"{scores['test']['rmse']:.1f}" in response.text

def test_app_loads_saved_model_without_retraining():
    from bike.data import ROOT
    model_path = ROOT / "artifacts/model.joblib"
    before = hashlib.sha256(model_path.read_bytes()).hexdigest()
    client = create_app().test_client()
    client.get("/")
    client.post("/", data=dict(EXAMPLE))
    after = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert before == after
