"""Flask app: loads the saved pipeline and serves the dashboard."""
import sys
from pathlib import Path
from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import LIMITS, predict_one, validate_inputs  # noqa: E402

app = Flask(__name__)

FIELDS = [  # (name, label, unit, step, placeholder, tooltip, group)
    ("Temperature", "Temperature", "°C", 0.1, "e.g. 32", "Noon air temperature.", "weather"),
    ("RH", "Relative Humidity", "%", 1, "e.g. 55", "Moisture in the air; low RH dries fuels.", "weather"),
    ("Ws", "Wind Speed", "km/h", 1, "e.g. 15", "Wind spreads fire faster.", "weather"),
    ("Rain", "Rain", "mm", 0.1, "e.g. 0", "Rainfall in the last 24 h.", "weather"),
    ("FFMC", "FFMC", "", 0.1, "e.g. 85", "Fine Fuel Moisture Code: litter dryness.", "fwi"),
    ("DMC", "DMC", "", 0.1, "e.g. 25", "Duff Moisture Code: moisture of loosely packed organic layers.", "fwi"),
    ("DC", "DC", "", 0.1, "e.g. 80", "Drought Code: deep, long-term dryness.", "fwi"),
    ("ISI", "ISI", "", 0.1, "e.g. 5", "Initial Spread Index: expected spread rate.", "fwi"),
    ("BUI", "BUI", "", 0.1, "e.g. 30", "Buildup Index: total fuel available to burn.", "fwi"),
]


def page(**ctx):
    return render_template("index.html", fields=FIELDS, limits=LIMITS, **ctx)


@app.route("/")
def home():
    return page(form={}, result=None, errors=[])


@app.route("/predict", methods=["POST"])
def predict():
    values, errors = validate_inputs(request.form)
    if errors:
        return page(form=request.form, result=None, errors=errors), 400
    try:
        result = predict_one(values)
    except FileNotFoundError as e:
        return page(form=request.form, result=None, errors=[str(e)]), 500
    except Exception:
        return page(form=request.form, result=None,
                    errors=["Prediction failed. Please check your inputs and try again."]), 500
    return page(form=request.form, result=result, errors=[])


if __name__ == "__main__":
    app.run(debug=True)
