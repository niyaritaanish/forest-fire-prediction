"""Train, compare, tune and save the models.  Run: python src/train_model.py"""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import SEED, ROOT, build_preprocessor, load_clean_data, split_data
from evaluate_model import (FIG_DIR, MET_DIR, compute_metrics, save_confusion_matrix,
                            save_feature_importance)


def make_pipe(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def main():
    for d in (FIG_DIR, MET_DIR, ROOT / "models"):
        d.mkdir(parents=True, exist_ok=True)
    df = load_clean_data()
    print(f"Loaded {len(df)} rows | Fire={df.target.sum()} NotFire={(df.target == 0).sum()}")
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(df)
    print(f"Split -> train {len(X_tr)}, val {len(X_val)}, test {len(X_te)}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Small randomized search (dataset is tiny)
    grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__gamma": [0, 0.1, 0.3],
    }
    search = RandomizedSearchCV(
        make_pipe(XGBClassifier(eval_metric="logloss", random_state=SEED)),
        grid, n_iter=40, cv=cv, scoring="roc_auc", random_state=SEED, n_jobs=-1)
    search.fit(X_tr, y_tr)  # tuning uses the training set only
    print("Best XGBoost params:", search.best_params_)
    final_pipe = search.best_estimator_

    candidates = {
        "Logistic Regression": make_pipe(LogisticRegression(max_iter=1000, random_state=SEED)),
        "Random Forest": make_pipe(RandomForestClassifier(n_estimators=200, random_state=SEED)),
        "SVM": make_pipe(SVC(probability=True, random_state=SEED)),
        "XGBoost (default)": make_pipe(XGBClassifier(eval_metric="logloss", random_state=SEED)),
        "XGBoost (tuned)": final_pipe,
    }
    rows = []
    for name, pipe in candidates.items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")  # clones internally
        pipe.fit(X_tr, y_tr)
        for split, X, y in (("validation", X_val, y_val), ("test", X_te, y_te)):
            m = compute_metrics(pipe, X, y)
            rows.append({"model": name, "split": split,
                         "cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
                         **{k: v for k, v in m.items() if k != "confusion_matrix"},
                         "tn_fp_fn_tp": str(np.array(m["confusion_matrix"]).ravel().tolist())})
    table = pd.DataFrame(rows).round(4)
    table.to_csv(ROOT / "outputs" / "model_comparison.csv", index=False)
    print("\n", table.to_string(index=False))

    # Final model = tuned XGBoost (project requirement). Compare honestly in the table above.
    test_m = compute_metrics(final_pipe, X_te, y_te)
    val_m = compute_metrics(final_pipe, X_val, y_val)
    (MET_DIR / "final_metrics.json").write_text(json.dumps(
        {"best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
         "validation": val_m, "test": test_m}, indent=2))
    save_confusion_matrix(test_m["confusion_matrix"], "Test Confusion Matrix (XGBoost)",
                          FIG_DIR / "confusion_matrix_test.png")
    imp = save_feature_importance(final_pipe, FIG_DIR / "feature_importance.png")
    (MET_DIR / "feature_importance.json").write_text(json.dumps(imp, indent=2))
    joblib.dump(final_pipe, ROOT / "models" / "xgboost_pipeline.pkl")
    print("\nSaved models/xgboost_pipeline.pkl")
    print("Test metrics:", {k: round(v, 4) for k, v in test_m.items() if k != "confusion_matrix"})
    print("Note: the test set has only ~37 rows, so these numbers are very noisy.")


if __name__ == "__main__":
    main()
