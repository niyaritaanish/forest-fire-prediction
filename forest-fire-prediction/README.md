# Forest Fire Prediction System (XGBoost + Flask)

Predicts **Fire / Not Fire**, a fire probability and a UI risk level from weather + FWI-system inputs
(SDG 13 Climate Action, SDG 15 Life on Land).

Pipeline: Dataset → Cleaning → EDA → Split 70/15/15 (stratified) → StandardScaler → XGBoost → Evaluation → joblib → Flask dashboard.

## Setup
1. Download the CSV from https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset and save it as
   `data/Algerian_forest_fires_dataset.csv` (the file begins with a line "Bejaia Region Dataset"; the loader handles this).
2. Install and run:
```
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python src/eda.py              # figures -> outputs/figures/
python src/train_model.py      # trains, tunes, compares, saves models/xgboost_pipeline.pkl
python src/evaluate_model.py   # test-set report, confusion matrix, ROC curve
python app.py                  # open http://127.0.0.1:5000
```

## Feature selection and leakage
Features used: Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI.
- **FWI is excluded.** It is computed directly from ISI and BUI, which are already inputs, so it adds no new
  information and makes the model lean on a composite index. The fire label also correlates strongly with FWI
  (see the correlation heatmap), so keeping it would make the task artificially easy.
- **day / month / year excluded:** identifiers of the observation, not weather conditions.
- **Region excluded** to keep the form simple.
- **Remaining caveat:** FFMC, DMC, DC, ISI and BUI are themselves calculated from weather (ISI from FFMC + wind,
  BUI from DMC + DC). They are available from standard FWI calculations at prediction time, so they are not target
  leakage, but the features are correlated, and the target is labelled for the same day. Results show the
  weather–fire association in this dataset, not true forecasting skill.

## Results
**To be generated after running `train_model.py`.** See `outputs/model_comparison.csv`,
`outputs/metrics/final_metrics.json` and `outputs/figures/`. No numbers are hard-coded anywhere.

## Risk levels
0–30% Low, 30–60% Moderate, 60–80% High, 80–100% Very High. This is a UI interpretation of model probability,
**not** a scientifically validated fire-danger scale.

## Sample inputs (run them; the model generates the result)
1. Temp 35, RH 35, Ws 15, Rain 0, FFMC 90, DMC 40, DC 150, ISI 10, BUI 45
2. Temp 26, RH 80, Ws 14, Rain 8, FFMC 55, DMC 5, DC 15, ISI 1, BUI 6

## Limitations
Only 244 rows (37 in validation, 37 in test), so metrics are noisy and a single split can look better or worse
by chance. Only two Algerian regions, summer 2012, so it may not generalise elsewhere. Feature importance shows
model contribution, not causation. Not an official warning system.

## Future scope
Live weather API, GIS maps, satellite data, IoT sensors, real-time alerts, larger multi-region data,
explainable AI (SHAP), cloud deployment.

## Troubleshooting
- `Dataset not found`: check the file name and location in `data/`.
- `Model file missing`: run `python src/train_model.py` first.
- `ModuleNotFoundError`: activate the venv and run `pip install -r requirements.txt`.
