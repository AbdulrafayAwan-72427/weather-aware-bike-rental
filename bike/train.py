"""Run a fixed chronological experiment; select by validation MAE only."""
import json
import time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from bike.data import ROOT, load_data, fingerprint, WEATHER
from bike.model import candidates, fit_model, predict

def metrics(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)), "rmse": float(np.sqrt(mean_squared_error(actual, predicted)))}

def main():
    frame = load_data()
    train = frame.iloc[:6132]
    validation = frame.iloc[6132:7440]
    test = frame.iloc[7440:]
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    results, models = {}, {}
    for name, estimator in candidates().items():
        started = time.perf_counter()
        models[name] = fit_model(name, estimator, train)
        results[name] = {"fit_seconds": time.perf_counter() - started, "validation": metrics(validation.rentals, predict(models[name], validation))}
    selected = min(results, key=lambda name: results[name]["validation"]["mae"])
    for name in models:
        predictions = predict(models[name], test)
        results[name]["test"] = metrics(test.rentals, predictions)
        active = test.operating.eq("Yes")
        results[name]["test_operating"] = metrics(test.loc[active, "rentals"], predictions[active])
    chosen = models[selected]
    joblib.dump(chosen, ROOT / "artifacts/model.joblib")
    predictions = predict(chosen, test)
    detail = test[["timestamp", "rentals", "hour", "season", "rainfall", "snowfall", "operating"]].copy()
    detail["prediction"] = predictions
    detail["absolute_error"] = abs(detail.rentals - predictions)
    detail.to_csv(output / "test_predictions.csv", index=False)
    groups = {"peak_hours": test.hour.isin([7, 8, 9, 17, 18, 19]), "off_peak": ~test.hour.isin([7, 8, 9, 17, 18, 19]), "wet": test.rainfall.gt(0) | test.snowfall.gt(0), "dry": test.rainfall.eq(0) & test.snowfall.eq(0)}
    groups.update({f"season_{season}": test.season.eq(season) for season in ["Winter", "Spring", "Summer", "Autumn"]})
    segments = {}
    for name, mask in groups.items():
        mask = mask & test.operating.eq("Yes")
        count = int(mask.sum())
        segments[name] = {"n": count, "metrics": metrics(test.loc[mask, "rentals"], predictions[mask]) if count >= 30 else None}
    summary = {"selected": selected, "selection": "lowest validation MAE; frozen training fit, no refit", "dataset_sha256": fingerprint(), "rows": len(frame), "missing_cells": int(frame.isna().sum().sum()), "closed_hours": int(frame.operating.eq("No").sum()), "splits": {name: {"n": len(part), "start": str(part.timestamp.min()), "end": str(part.timestamp.max()), "closed_hours": int(part.operating.eq("No").sum())} for name, part in [("train", train), ("validation", validation), ("test", test)]}, "models": results, "segments_operating_only": segments, "training_weather_ranges": {column: [float(train[column].min()), float(train[column].max())] for column in WEATHER}}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    train.groupby("hour").rentals.mean().plot(title="Training period: mean observed rentals by hour", ylabel="Rentals")
    plt.tight_layout(); plt.savefig(output / "hourly_profile.png"); plt.close()
    frame.assign(month_date=frame.timestamp.dt.to_period("M").astype(str)).groupby("month_date").rentals.mean().plot.bar(title="Observed monthly rentals (descriptive, all periods)", ylabel="Hourly mean rentals")
    plt.tight_layout(); plt.savefig(output / "monthly_profile.png"); plt.close()
    detail.iloc[:168].plot(x="timestamp", y=["rentals", "prediction"], title="First held-out week")
    plt.tight_layout(); plt.savefig(output / "test_week.png"); plt.close()
    # Actual vs predicted on the final test period (chosen model only).
    operating_test = detail[detail.operating.eq("Yes")]
    plt.figure()
    plt.scatter(operating_test.rentals, operating_test.prediction, s=8, alpha=0.45)
    limit = max(operating_test.rentals.max(), operating_test.prediction.max())
    plt.plot([0, limit], [0, limit], color="red", linewidth=1)
    plt.xlabel("Observed rentals / hour")
    plt.ylabel("Predicted rentals / hour")
    plt.title("Actual vs predicted — final test period (operating hours)")
    plt.tight_layout(); plt.savefig(output / "test_actual_vs_predicted.png"); plt.close()
    # Mean absolute error by hour of day on the test period.
    operating_test.assign(abs_error=operating_test.absolute_error).groupby("hour").abs_error.mean().plot.bar()
    plt.xlabel("Hour of day")
    plt.ylabel("Mean absolute error (rentals / hour)")
    plt.title("Test MAE by hour (operating hours)")
    plt.tight_layout(); plt.savefig(output / "test_error_by_hour.png"); plt.close()
    train[WEATHER + ["rentals"]].corr().to_csv(output / "training_correlations.csv")
    frame.describe(include="all").to_csv(output / "dataset_summary.csv")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
