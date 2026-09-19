"""Cardiovascular Disease (CVD) risk analysis.

Loads the Cleveland heart-disease dataset, cleans it, explores the
relationship between each clinical factor and the presence of CVD, and
trains a logistic regression classifier as a baseline predictive model.

Run from a VS Code terminal:
    python3 cvd_analysis.py

Figures are written to output/figures/ instead of being shown
interactively, so the script runs end-to-end without blocking.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "Dataset" / "data.xlsx"
FIGURES_DIR = ROOT_DIR / "output" / "figures"

CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
PARAM_GRID = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"],
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_excel(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing values/duplicates, then drop duplicate rows."""
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    df = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df.shape}")
    return df


def summarize(df: pd.DataFrame) -> None:
    print("\n--- Structure ---")
    df.info()

    print("\n--- Summary statistics ---")
    print(df.describe())

    print("\n--- Categorical variable cardinality ---")
    print(df[CATEGORICAL_COLUMNS].nunique())


def plot_categorical_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.countplot(x="cp", hue="cp", data=df, palette="Set2", legend=False, ax=axes[0, 0])
    axes[0, 0].set_title(
        "Chest Pain Type Distribution\n"
        "0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic"
    )

    sns.countplot(x="sex", hue="sex", data=df, palette="Set2", legend=False, ax=axes[0, 1])
    axes[0, 1].set_title("Gender Distribution\n0 = Female, 1 = Male")

    sns.countplot(x="restecg", hue="restecg", data=df, palette="Set2", legend=False, ax=axes[1, 0])
    axes[1, 0].set_title(
        "Resting ECG Distribution\n"
        "0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy"
    )

    sns.countplot(x="thal", hue="thal", data=df, palette="Set2", legend=False, ax=axes[1, 1])
    axes[1, 1].set_title(
        "Thalassemia Distribution\n3 = Normal, 6 = Fixed defect, 7 = Reversible defect"
    )

    fig.tight_layout()
    fig.savefig(out_dir / "01_categorical_distributions.png")
    plt.close(fig)


def plot_age_vs_cvd(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="target", y="age", hue="target", data=df, palette="Set3", legend=False, ax=ax)
    ax.set_title("CVD Occurrence Across Age")
    ax.set_xlabel("CVD (0 = No, 1 = Yes)")
    ax.set_ylabel("Age")
    fig.savefig(out_dir / "02_age_vs_cvd.png")
    plt.close(fig)


def plot_gender_vs_cvd(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x="sex", hue="target", data=df, palette="Set1", ax=ax)
    ax.set_title("Gender Distribution with CVD Occurrence")
    ax.set_xlabel("Gender (0 = Female, 1 = Male)")
    ax.set_ylabel("Count")
    fig.savefig(out_dir / "03_gender_vs_cvd.png")
    plt.close(fig)


def plot_resting_bp_vs_cvd(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="target", y="trestbps", hue="target", data=df, palette="Set3", legend=False, ax=ax)
    ax.set_title("Resting Blood Pressure vs CVD")
    ax.set_xlabel("CVD (0 = No, 1 = Yes)")
    ax.set_ylabel("Resting Blood Pressure (mm Hg)")
    fig.savefig(out_dir / "04_resting_bp_vs_cvd.png")
    plt.close(fig)


def plot_cholesterol_vs_cvd(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="target", y="chol", hue="target", data=df, palette="Set3", legend=False, ax=ax)
    ax.set_title("Cholesterol Levels vs CVD")
    ax.set_xlabel("CVD (0 = No, 1 = Yes)")
    ax.set_ylabel("Cholesterol (mg/dL)")
    fig.savefig(out_dir / "05_cholesterol_vs_cvd.png")
    plt.close(fig)


def plot_max_heart_rate_vs_cvd(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="target", y="thalach", hue="target", data=df, palette="Set3", legend=False, ax=ax)
    ax.set_title("Maximum Heart Rate Achieved vs CVD")
    ax.set_xlabel("CVD (0 = No, 1 = Yes)")
    ax.set_ylabel("Maximum Heart Rate (bpm)")
    fig.savefig(out_dir / "06_max_heart_rate_vs_cvd.png")
    plt.close(fig)


def plot_thalassemia_vs_cvd(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x="thal", hue="target", data=df, palette="Set1", ax=ax)
    ax.set_title("Thalassemia Distribution vs CVD")
    ax.set_xlabel("Thalassemia Type")
    ax.set_ylabel("Count")
    fig.savefig(out_dir / "07_thalassemia_vs_cvd.png")
    plt.close(fig)


def plot_other_factors_vs_cvd(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.countplot(x="fbs", hue="target", data=df, palette="Set1", ax=axes[0, 0])
    axes[0, 0].set_title("Fasting Blood Sugar vs CVD")

    sns.countplot(x="exang", hue="target", data=df, palette="Set1", ax=axes[0, 1])
    axes[0, 1].set_title("Exercise-Induced Angina vs CVD")

    sns.countplot(x="slope", hue="target", data=df, palette="Set1", ax=axes[1, 0])
    axes[1, 0].set_title("Slope of Peak Exercise ST Segment vs CVD")

    sns.countplot(x="ca", hue="target", data=df, palette="Set1", ax=axes[1, 1])
    axes[1, 1].set_title("Number of Major Vessels vs CVD")

    fig.tight_layout()
    fig.savefig(out_dir / "08_other_factors_vs_cvd.png")
    plt.close(fig)


def plot_pairplot(df: pd.DataFrame, out_dir: Path) -> None:
    grid = sns.pairplot(df, hue="target", palette="Set1")
    grid.fig.suptitle("Pair Plot of Variables with CVD Highlighted", y=1.02)
    grid.savefig(out_dir / "09_pairplot.png")
    plt.close(grid.fig)


def run_eda(df: pd.DataFrame, out_dir: Path) -> None:
    plot_categorical_distributions(df, out_dir)
    plot_age_vs_cvd(df, out_dir)
    plot_gender_vs_cvd(df, out_dir)
    plot_resting_bp_vs_cvd(df, out_dir)
    plot_cholesterol_vs_cvd(df, out_dir)
    plot_max_heart_rate_vs_cvd(df, out_dir)
    plot_thalassemia_vs_cvd(df, out_dir)
    plot_other_factors_vs_cvd(df, out_dir)
    plot_pairplot(df, out_dir)
    print(f"\nFigures written to {out_dir}")


def train_baseline_model(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def tune_model(X_train_scaled, y_train) -> GridSearchCV:
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=2000), PARAM_GRID, cv=5, scoring="accuracy"
    )
    grid_search.fit(X_train_scaled, y_train)
    return grid_search


def run_modeling(df: pd.DataFrame) -> None:
    X = df.drop(columns="target")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    baseline_model = train_baseline_model(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    print("\n--- Baseline Logistic Regression ---")
    print(classification_report(y_test, baseline_pred))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    grid_search = tune_model(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    tuned_pred = best_model.predict(X_test_scaled)

    print("\n--- Tuned Logistic Regression (scaled features) ---")
    print("Best hyperparameters:", grid_search.best_params_)
    print(classification_report(y_test, tuned_pred))

    baseline_acc = accuracy_score(y_test, baseline_pred)
    tuned_acc = accuracy_score(y_test, tuned_pred)
    print(f"\nBaseline accuracy: {baseline_acc:.2%}")
    print(f"Tuned accuracy:    {tuned_acc:.2%}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    heart = load_data()
    heart = clean_data(heart)
    summarize(heart)
    run_eda(heart, FIGURES_DIR)
    run_modeling(heart)


if __name__ == "__main__":
    main()
