"""Evaluation helpers + standalone evaluation of the saved model on the test set."""
import json
import sys
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             RocCurveDisplay)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import ROOT, FEATURES, load_clean_data, split_data

FIG_DIR = ROOT / "outputs" / "figures"
MET_DIR = ROOT / "outputs" / "metrics"
MODEL_PATH = ROOT / "models" / "xgboost_pipeline.pkl"


def compute_metrics(pipe, X, y):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def save_confusion_matrix(cm, title, path):
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Not Fire", "Fire"], yticklabels=["Not Fire", "Fire"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def save_feature_importance(pipe, path):
    imp = pipe.named_steps["model"].feature_importances_
    order = imp.argsort()
    plt.figure(figsize=(6, 4))
    plt.barh([FEATURES[i] for i in order], imp[order], color="#2d6a4f")
    plt.xlabel("XGBoost importance (model contribution, not causation)")
    plt.title("Feature Importance"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    return dict(zip(FEATURES, map(float, imp)))


def main():
    pipe = joblib.load(MODEL_PATH)
    *_, X_test, _, _, y_test = split_data(load_clean_data())
    m = compute_metrics(pipe, X_test, y_test)
    print(classification_report(y_test, pipe.predict(X_test), target_names=["Not Fire", "Fire"]))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    save_confusion_matrix(m["confusion_matrix"], "Test Confusion Matrix", FIG_DIR / "confusion_matrix_test.png")
    RocCurveDisplay.from_estimator(pipe, X_test, y_test)
    plt.savefig(FIG_DIR / "roc_curve_test.png", dpi=150); plt.close()
    (MET_DIR / "test_metrics.json").write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
