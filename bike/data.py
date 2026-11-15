"""Dataset acquisition, schema checks and deterministic calendar features."""
from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/SeoulBikeData.csv"
URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
WEATHER = ["temperature", "humidity", "wind_speed", "visibility", "dew_point", "solar", "rainfall", "snowfall"]
NUMERIC = WEATHER + ["hour_sin", "hour_cos"]
CATEGORICAL = ["hour", "weekday", "month", "season", "holiday"]
COLUMNS = ["date", "rentals", "hour", *WEATHER[:5], "solar", "rainfall", "snowfall", "season", "holiday", "operating"]
# Original header of SeoulBikeData.csv as supplied by UCI, read with
# encoding="cp1252" (the file is not UTF-8: the degree sign in
# "Temperature(°C)" is byte 0xB0). Renaming is only allowed after this
# exact list, in this order, has been verified — never positionally on
# an unchecked header.
ORIGINAL_COLUMNS = ["Date", "Rented Bike Count", "Hour", "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)", "Visibility (10m)", "Dew point temperature(°C)", "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)", "Seasons", "Holiday", "Functioning Day"]
LIMITS = {"temperature": (-40, 50), "humidity": (0, 100), "wind_speed": (0, 50), "visibility": (0, 5000), "dew_point": (-60, 50), "solar": (0, 10), "rainfall": (0, 200), "snowfall": (0, 100)}

def download():
    DATA.parent.mkdir(exist_ok=True)
    if not DATA.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        DATA.write_bytes(archive.read("SeoulBikeData.csv"))
    return DATA

def features(frame):
    result = frame.copy()
    dates = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="raise")
    result["weekday"] = dates.dt.dayofweek
    result["month"] = dates.dt.month
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    return result

def validate_and_rename(frame):
    """Check the original UCI header, then rename to internal names.

    Rejects a missing, unexpected, or misordered column with a message
    naming the problem, instead of silently renaming by position.
    """
    actual = list(frame.columns)
    if actual != ORIGINAL_COLUMNS:
        missing = [c for c in ORIGINAL_COLUMNS if c not in actual]
        unexpected = [c for c in actual if c not in ORIGINAL_COLUMNS]
        detail = f"Expected the 14 original UCI columns in order {ORIGINAL_COLUMNS}; got {actual}."
        if missing:
            detail += f" Missing: {missing}."
        if unexpected:
            detail += f" Unexpected: {unexpected}."
        if not missing and not unexpected:
            detail += " All expected columns are present but in the wrong order."
        raise ValueError(f"Unexpected dataset schema. {detail}")
    renamed = frame.copy()
    renamed.columns = COLUMNS
    return renamed

def load_data(path=None):
    frame = pd.read_csv(path or download(), encoding="cp1252")
    frame = validate_and_rename(frame)
    frame["date"] = pd.to_datetime(frame["date"], format="%d/%m/%Y").dt.strftime("%Y-%m-%d")
    frame["timestamp"] = pd.to_datetime(frame["date"]) + pd.to_timedelta(frame["hour"], unit="h")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if len(frame) != 8760 or frame.isna().any().any() or frame.timestamp.duplicated().any():
        raise ValueError("Dataset must contain 8760 unique complete hourly records")
    if not frame.timestamp.diff().dropna().dt.total_seconds().eq(3600).all():
        raise ValueError("Hourly sequence is incomplete")
    if not frame.operating.isin(["Yes", "No"]).all() or (frame.rentals < 0).any():
        raise ValueError("Invalid operation flag or target")
    if frame.loc[frame.operating.eq("No"), "rentals"].ne(0).any():
        raise ValueError("Closure rule conflicts with observed rentals")
    return features(frame)

def fingerprint():
    return hashlib.sha256(DATA.read_bytes()).hexdigest()
