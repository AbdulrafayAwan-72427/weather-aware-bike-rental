"""Shared training and inference preprocessing and closure policy."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from bike.data import NUMERIC, CATEGORICAL

def candidates():
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    from sklearn.base import clone
    return {
        "Hourly mean": None,
        "Linear Regression": make_pipeline(clone(preprocessing), LinearRegression()),
        "Random Forest": make_pipeline(clone(preprocessing), RandomForestRegressor(n_estimators=250, min_samples_leaf=3, max_features=0.8, random_state=42, n_jobs=-1)),
    }

def fit_model(name, estimator, train):
    active = train.loc[train.operating.eq("Yes")]
    if estimator is None:
        estimator = active.groupby("hour").rentals.mean().to_dict()
    else:
        estimator.fit(active, active.rentals)
    return {"name": name, "estimator": estimator}

def predict(model, frame):
    if isinstance(model["estimator"], dict):
        values = frame.hour.map(model["estimator"]).to_numpy(dtype=float)
    else:
        values = model["estimator"].predict(frame)
    return np.where(frame.operating.eq("No"), 0.0, np.maximum(values, 0.0))
