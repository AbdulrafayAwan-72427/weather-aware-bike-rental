"""Local-only scenario interface using the saved training pipeline."""
from datetime import date
import json
import math
import joblib
import pandas as pd
from flask import Flask, render_template, request
from bike.data import ROOT, LIMITS, features
from bike.model import predict

def validate(payload):
    chosen_date = date.fromisoformat(payload["date"])
    hour = int(payload["hour"])
    if str(hour) != str(payload["hour"]) or not 0 <= hour <= 23:
        raise ValueError("Hour must be an integer from 0 to 23")
    row = {"date": chosen_date.isoformat(), "hour": hour}
    for field, (lower, upper) in LIMITS.items():
        value = float(payload[field])
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{field.replace('_', ' ').title()} must be between {lower} and {upper}")
        row[field] = value
    if row["dew_point"] > row["temperature"]:
        raise ValueError("Dew point cannot exceed air temperature")
    for field, choices in {"operating": ["Yes", "No"], "holiday": ["Holiday", "No Holiday"]}.items():
        if payload[field] not in choices:
            raise ValueError(f"Invalid {field}")
        row[field] = payload[field]
    row["season"] = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}[chosen_date.month]
    return features(pd.DataFrame([row]))

# Documented example scenario: a busy autumn evening hour (15 Oct 2018,
# 18:00, 18 °C, 55% humidity). The real hour that day had 2,635 observed
# rentals at 66% humidity; the saved model predicts 2,100 for this
# scenario (see docs/Technical-Report.md §9).
EXAMPLE = {"date": "2018-10-15", "hour": "18", "temperature": "18", "humidity": "55", "wind_speed": "1.5", "visibility": "1800", "dew_point": "9", "solar": "0.2", "rainfall": "0", "snowfall": "0", "operating": "Yes", "holiday": "No Holiday"}

def create_app():
    app = Flask(__name__, template_folder=str(ROOT / "templates"))
    model = joblib.load(ROOT / "artifacts/model.joblib")
    summary = json.loads((ROOT / "reports/metrics.json").read_text())

    @app.route("/", methods=["GET", "POST"])
    def index():
        values = dict(EXAMPLE)
        result, error, warnings, notice = None, None, [], None
        if request.method == "GET" and request.args.get("example") == "1":
            # Loads the example inputs only; it does not calculate a
            # prediction. The user still presses "Estimate hourly rentals".
            notice = "Example scenario loaded. Press “Estimate hourly rentals” to calculate a prediction."
        if request.method == "POST":
            values.update(request.form)
            try:
                frame = validate(values)
                result = round(float(predict(model, frame)[0]))
                for field, (lower, upper) in summary["training_weather_ranges"].items():
                    if not lower <= frame[field].iloc[0] <= upper:
                        warnings.append(f"{field.replace('_', ' ').title()} is outside the training range ({lower:g}–{upper:g}).")
            except (ValueError, KeyError, OverflowError) as exc:
                error = str(exc)
        return render_template("index.html", values=values, result=result, error=error, warnings=warnings, notice=notice, summary=summary, limits=LIMITS)

    return app

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
