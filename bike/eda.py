"""Exploratory analysis of the Seoul bike-sharing data.

Reads the validated frame from ``bike.data`` (no invented values, no
target-derived features) and writes charts plus a JSON summary to
``reports/``. Run with ``python -m bike.eda`` after the dataset download.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bike.data import ROOT, WEATHER, load_data


def _save(fig_path, title):
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def main():
    frame = load_data()
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    operating = frame[frame.operating.eq("Yes")]

    # 1. Rentals by hour of day (operating hours only, so closed hours
    #    do not drag the early-morning means down).
    hourly = operating.groupby("hour").rentals.agg(["mean", "count"])
    hourly["mean"].plot.bar()
    plt.ylabel("Mean rentals / hour")
    plt.xlabel("Hour of day (Seoul local time)")
    _save(output / "eda_hourly.png", "Mean rentals by hour (operating hours)")

    # 2. Rentals by season.
    order = ["Winter", "Spring", "Summer", "Autumn"]
    seasonal = operating.groupby("season").rentals.mean().reindex(order)
    seasonal.plot.bar()
    plt.ylabel("Mean rentals / hour")
    plt.xlabel("Season")
    _save(output / "eda_season.png", "Mean rentals by season (operating hours)")

    # 3. Rentals by temperature band.
    bins = pd.cut(operating.temperature, [-20, -10, 0, 10, 20, 30, 40])
    temp = operating.groupby(bins, observed=True).rentals.mean()
    temp.plot.bar()
    plt.ylabel("Mean rentals / hour")
    plt.xlabel("Temperature band (°C)")
    _save(output / "eda_temperature.png", "Mean rentals by temperature band")

    # 4. Target distribution (right-skewed: a few very busy hours).
    operating.rentals.plot.hist(bins=40)
    plt.xlabel("Rented bikes in one hour")
    plt.ylabel("Hours")
    _save(output / "eda_distribution.png", "Distribution of hourly rentals")

    # 5. Wet vs dry hours, and holiday vs non-holiday.
    wet = operating.rainfall.gt(0) | operating.snowfall.gt(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pd.Series(
        {"Dry": operating.loc[~wet, "rentals"].mean(),
         "Wet": operating.loc[wet, "rentals"].mean()}
    ).plot.bar(ax=axes[0])
    axes[0].set_title("Wet vs dry hours")
    axes[0].set_ylabel("Mean rentals / hour")
    operating.groupby("holiday").rentals.mean().plot.bar(ax=axes[1])
    axes[1].set_title("Holiday vs non-holiday")
    axes[1].set_ylabel("Mean rentals / hour")
    plt.tight_layout()
    plt.savefig(output / "eda_weather_holiday.png")
    plt.close()

    # 6. Monthly pattern (descriptive only — one year, so a month appears once).
    monthly = frame.assign(month=frame.timestamp.dt.to_period("M").astype(str))
    monthly.groupby("month").rentals.mean().plot.bar(figsize=(12, 4))
    plt.ylabel("Mean rentals / hour")
    plt.xlabel("Month")
    _save(output / "eda_monthly.png", "Mean rentals by month (one observation per month)")

    summary = {
        "rows": len(frame),
        "date_range": [str(frame.timestamp.min()), str(frame.timestamp.max())],
        "missing_cells": int(frame.isna().sum().sum()),
        "closed_hours": int(frame.operating.eq("No").sum()),
        "closed_rentals_all_zero": bool(
            frame.loc[frame.operating.eq("No"), "rentals"].eq(0).all()
        ),
        "target": {
            "mean": float(frame.rentals.mean()),
            "median": float(frame.rentals.median()),
            "min": int(frame.rentals.min()),
            "max": int(frame.rentals.max()),
            "std": float(frame.rentals.std()),
        },
        "hourly_mean": {str(k): float(v) for k, v in hourly["mean"].items()},
        "season_mean_operating": {k: float(v) for k, v in seasonal.items()},
        "season_counts": {k: int(v) for k, v in frame.season.value_counts().items()},
        "holiday_counts": {k: int(v) for k, v in frame.holiday.value_counts().items()},
        "wet_hours_operating": int(wet.sum()),
        "wet_mean": float(operating.loc[wet, "rentals"].mean()),
        "dry_mean": float(operating.loc[~wet, "rentals"].mean()),
        "holiday_mean": float(operating.loc[operating.holiday.eq("Holiday"), "rentals"].mean()),
        "non_holiday_mean": float(
            operating.loc[operating.holiday.eq("No Holiday"), "rentals"].mean()
        ),
        "correlations_with_rentals": {
            column: float(frame[column].corr(frame.rentals)) for column in WEATHER
        },
    }
    (output / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
