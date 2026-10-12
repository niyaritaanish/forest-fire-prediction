"""Exploratory data analysis. Run: python src/eda.py  (figures -> outputs/figures/)"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import ALL_NUMERIC, FEATURES, ROOT, load_clean_data

FIG = ROOT / "outputs" / "figures"
PAL = {"Not Fire": "#74c69d", "Fire": "#e76f51"}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_clean_data()
    df["Class"] = df["target"].map({1: "Fire", 0: "Not Fire"})
    print("Shape:", df.shape); print(df[ALL_NUMERIC].dtypes)
    print(df[ALL_NUMERIC].describe().T)
    print("Missing values:\n", df[ALL_NUMERIC].isna().sum())
    print("Class counts:\n", df["Class"].value_counts())

    sns.countplot(data=df, x="Class", hue="Class", palette=PAL, legend=False)
    plt.title("Class distribution"); plt.savefig(FIG / "class_distribution.png", dpi=150); plt.close()

    plt.figure(figsize=(9, 7))
    sns.heatmap(df[ALL_NUMERIC + ["target"]].corr(), annot=True, fmt=".2f", cmap="RdYlGn_r")
    plt.title("Correlation heatmap (note FWI vs ISI/BUI)")
    plt.tight_layout(); plt.savefig(FIG / "correlation_heatmap.png", dpi=150); plt.close()

    df[FEATURES].hist(bins=20, figsize=(11, 8), color="#2d6a4f")
    plt.tight_layout(); plt.savefig(FIG / "histograms.png", dpi=150); plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.boxplot(data=df, x="Class", y=col, hue="Class", palette=PAL, ax=ax, legend=False)
    plt.tight_layout(); plt.savefig(FIG / "boxplots_by_class.png", dpi=150); plt.close()

    for x, y in [("Temperature", "RH"), ("Ws", "Rain"), ("FFMC", "ISI"), ("DMC", "BUI"), ("DC", "BUI")]:
        sns.scatterplot(data=df, x=x, y=y, hue="Class", palette=PAL)
        plt.title(f"{x} vs {y}"); plt.savefig(FIG / f"scatter_{x}_{y}.png", dpi=150); plt.close()
    print("Figures saved to", FIG)


if __name__ == "__main__":
    main()
