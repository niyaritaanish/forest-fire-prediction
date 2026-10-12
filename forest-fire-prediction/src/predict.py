"""Validation + prediction using the saved training pipeline (no re-implemented preprocessing)."""
import sys
from pathlib import Path
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import FEATURES, ROOT

MODEL_PATH = ROOT / "models" / "xgboost_pipeline.pkl"

# Validation limits: dataset range with a small margin.
LIMITS = {
    "Temperature": (15, 50), "RH": (10, 100), "Ws": (0, 40), "Rain": (0, 30),
    "FFMC": (20, 100), "DMC": (0, 80), "DC": (0, 300), "ISI": (0, 25), "BUI": (0, 90),
}
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model file missing. Run: python src/train_model.py")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def validate_inputs(form):
    """Return (values_dict, errors_list)."""
    values, errors = {}, []
    for name in FEATURES:
        raw = (form.get(name) or "").strip()
        if raw == "":
            errors.append(f"{name} is required."); continue
        try:
            v = float(raw)
        except ValueError:
            errors.append(f"{name} must be a number."); continue
        lo, hi = LIMITS[name]
        if not lo <= v <= hi:
            errors.append(f"{name} must be between {lo} and {hi}."); continue
        values[name] = v
    return values, errors


def risk_level(p):
    """UI interpretation of model probability, NOT a scientific fire-danger scale."""
    if p < 0.30: return "Low"
    if p < 0.60: return "Moderate"
    if p < 0.80: return "High"
    return "Very High"


def predict_one(values):
    pipe = get_pipeline()
    X = pd.DataFrame([values], columns=FEATURES)
    proba = float(pipe.predict_proba(X)[0, 1])
    return {"label": "FIRE" if proba >= 0.5 else "NOT FIRE",
            "probability": round(proba * 100, 1), "risk": risk_level(proba)}
