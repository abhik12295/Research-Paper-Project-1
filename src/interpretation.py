import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR


def ridge_feature_importance(ridge_model, feature_names):
    coefs = ridge_model.named_steps["model"].coef_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_importance": np.abs(coefs)
    }).sort_values("abs_importance", ascending=False)

    return importance_df


def xgb_feature_importance(xgb_model, feature_names):
    importance = xgb_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    return importance_df


def plot_feature_importance(df, title, filename):
    plt.figure(figsize=(8, 5))

    plt.barh(df["feature"][:10][::-1], df.iloc[:10, 1][::-1])

    plt.title(title)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()

    print(f"Saved {filename}")