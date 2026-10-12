"""Data loading, cleaning, feature selection, splitting and preprocessing."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Algerian_forest_fires_dataset.csv"


FEATURES = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI"]
ALL_NUMERIC = FEATURES + ["FWI"]


def load_clean_data(path=DATA_PATH):
    """Load the UCI CSV and return a clean dataframe with a 0/1 'target' column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. See README for download steps.")
    df = pd.read_csv(path, skiprows=1, encoding='latin1')
    df.columns = df.columns.str.strip()
    df = df[pd.to_numeric(df["day"], errors="coerce").notna()].copy()
    for col in ALL_NUMERIC:  # dirty cells like '14.6 9' become NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Classes"] = df["Classes"].astype(str).str.strip().str.lower()
    df = df[df["Classes"].isin(["fire", "not fire"])]
    df["target"] = (df["Classes"] == "fire").astype(int)
    df = df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
    return df


def split_data(df):
    """Stratified 70/15/15 split. Returns (X_train, X_val, X_test, y_train, y_val, y_test)."""
    X, y = df[FEATURES], df["target"]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor():
    """Scale all numeric features. Saved inside the final Pipeline and reused by Flask."""
    return ColumnTransformer([("scale", StandardScaler(), FEATURES)])
