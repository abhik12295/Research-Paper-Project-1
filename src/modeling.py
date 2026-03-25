# from __future__ import annotations
# from pathlib import Path
# from typing import Dict, Tuple
# import numpy as np
# import pandas as pd
# from joblib import dump
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import Ridge
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBRegressor
# from src.config import OUTPUT_DIR, FIGURES_DIR, MODELS_DIR
# import matplotlib.pyplot as plt


# # metrics
# def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     denom = np.where(y_true == 0, np.nan, y_true)
#     return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

# def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#     mae = float(np.mean(np.abs(y_true - y_pred))) 
#     rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
#     return {"MAE":mae, "RMSE":rmse, "MAPE_%": mape(y_true, y_pred)}

# # train and evaluate
# def train_and_evaluate(
#         df: pd.DataFrame,
#         target_col: str,
#         test_horizon: int = 24,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline, XGBRegressor]:
    
#     df = df.sort_values("date").reset_index(drop=True)

#     # feature_cols = [c for c in df.columns if c not in ["date", "y_next"]]
#     # X = df[feature_cols]
#     # y = df["y_next"].astype(float)

#     selected_features = [
#         target_col,
#         f"{target_col}_lag1",
#         f"{target_col}_lag2",
#         f"{target_col}_lag3",
#         f"{target_col}_lag12",
#         f"{target_col}_rollmean3",
#         f"{target_col}_rollmean6",
#         "month_sin",
#         "month_cos",
#     ]

#     # Add FRED features
#     selected_features += [
#         "pcu484484",
#         "wpu057303",
#         "ces4348400001",
#     ]

#     # Keep only existing columns (safe guard)
#     selected_features = [c for c in selected_features if c in df.columns]

#     print(f"\n[INFO] Selected features ({len(selected_features)}): {selected_features}")

#     X = df[selected_features]
#     y = df["y_next"].astype(float)

#     split = len(df) - test_horizon

#     X_train, X_test = X.iloc[:split], X.iloc[split:]
#     y_train, y_test = y.iloc[:split], y.iloc[split:]
#     dates_test = df["date"].iloc[split:]

#     print(f"\nTrain size: {len(X_train)}")
#     print(f"Test size: {len(X_test)}")

#     # baseline
#     print("\nTraining baseline...")
#     pred_persistence = X_test[target_col].to_numpy()
#     pred_seasonal = df[target_col].shift(12).iloc[split:].to_numpy()

#     # ridge
#     print("Training Ridge model...")
#     ridge = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler()),
#         ("model", Ridge(alpha=1.0)),
#     ]) 

#     ridge.fit(X_train, y_train)
#     pred_ridge = np.asarray(ridge.predict(X_test), dtype=float)

#     # Xgboost
#     print("Training XGBoost model...")
#     xgb = XGBRegressor(
#         n_estimators=500,
#         learning_rate=0.03,
#         max_depth=2,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         objective="reg:squarederror",
#         random_state=0,
#         verbosity=0,
#     )

#     xgb.fit(X_train, y_train)
#     # pred_xgb = xgb.predict(X_test)
#     pred_xgb = np.asarray(xgb.predict(X_test), dtype=float)
#     train_pred = xgb.predict(X_train)

#     print("\n[DEBUG] XGBoost Overfitting Check:")
#     print("Train MAE:", np.mean(np.abs(y_train - train_pred)))
#     print("Test MAE :", np.mean(np.abs(y_test - pred_xgb)))

#     # metrics
#     y_true = np.asarray(y_test.to_numpy(), dtype=float)

#     rows = [
#         {"model": "Persistence", **compute_metrics(y_true, pred_persistence)},
#         {"model": "Seasonal", **compute_metrics(y_true, pred_seasonal)},
#         {"model": "Ridge", **compute_metrics(y_true, pred_ridge)},
#         {"model": "XGBoost", **compute_metrics(y_true, pred_xgb)},
#     ]

#     metrics_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
#     preds_df = pd.DataFrame({
#         "date": dates_test,
#         "actual": y_true,
#         "persistence": pred_persistence,
#         "seasonal": pred_seasonal,
#         "ridge": pred_ridge,
#         "xgboost": pred_xgb,
#     })
#     return metrics_df, preds_df, ridge, xgb


# # SAVE OUTPUTS
# def save_outputs(metrics_df, preds_df, ridge, xgb):

#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     FIGURES_DIR.mkdir(parents=True, exist_ok=True)
#     MODELS_DIR.mkdir(parents=True, exist_ok=True)

#     metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
#     preds_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

#     dump(ridge, MODELS_DIR / "ridge.joblib")
#     xgb.save_model(str(MODELS_DIR / "xgboost.json"))

#     print("\nSaved outputs and models.")

# # Plots per paper
# def create_plots(preds_df: pd.DataFrame, metrics_df: pd.DataFrame):

#     # Forecast plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(preds_df["date"], preds_df["actual"], label="Actual")
#     plt.plot(preds_df["date"], preds_df["ridge"], label="Ridge")
#     plt.plot(preds_df["date"], preds_df["xgboost"], label="XGBoost")
#     plt.legend()
#     plt.title("Forecast vs Actual")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("output/figures/forecast.png", dpi=200)
#     plt.close()

#     # Metrics plot
#     plt.figure(figsize=(8, 5))
#     x = np.arange(len(metrics_df))

#     plt.bar(x, metrics_df["MAE"])
#     plt.xticks(x, metrics_df["model"].tolist())
#     plt.title("Model Comparison (MAE)")
#     plt.tight_layout()
#     plt.savefig("output/figures/metrics.png", dpi=200)
#     plt.close()

#     print("Saved plots.")


from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import FIGURES_DIR, MODELS_DIR, OUTPUT_DIR


# =========================================================
# METRICS
# =========================================================
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_%": mape(y_true, y_pred),
    }


# =========================================================
# FEATURE SELECTION
# =========================================================
def select_features(df: pd.DataFrame, target_col: str) -> list[str]:
    selected_features: list[str] = []

    if target_col in df.columns:
        selected_features.append(target_col)

    for lag in [1, 2, 3, 6, 12]:
        col = f"{target_col}_lag{lag}"
        if col in df.columns:
            selected_features.append(col)

    for col in [
        f"{target_col}_rollmean3",
        f"{target_col}_rollmean6",
        f"{target_col}_rollstd3",
    ]:
        if col in df.columns:
            selected_features.append(col)

    for col in ["month_sin", "month_cos"]:
        if col in df.columns:
            selected_features.append(col)

    for col in ["pcu484484", "wpu057303"]:
        if col in df.columns:
            selected_features.append(col)

    if not selected_features:
        raise ValueError("No selected features found in dataframe.")

    return selected_features


# =========================================================
# TRAIN AND EVALUATE
# =========================================================
def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    test_horizon: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, XGBRegressor]:
    df = df.sort_values("date").reset_index(drop=True).copy()

    if "date" not in df.columns:
        raise ValueError("Input dataframe must contain 'date' column.")
    if "y_next" not in df.columns:
        raise ValueError("Input dataframe must contain 'y_next' column.")
    if len(df) <= test_horizon:
        raise ValueError(
            f"test_horizon={test_horizon} is too large for dataset size={len(df)}."
        )

    selected_features = select_features(df, target_col)
    print(f"\n[INFO] Selected features ({len(selected_features)}): {selected_features}")

    X = df[selected_features].copy()
    y = df["y_next"].astype(float)

    split = len(df) - test_horizon
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    dates_test = df["date"].iloc[split:].reset_index(drop=True)

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # =====================================================
    # BASELINES
    # =====================================================
    print("\nTraining baseline...")

    pred_persistence = np.asarray(X_test[target_col], dtype=float)

    seasonal_series = df[target_col].shift(12).iloc[split:]
    pred_seasonal = np.asarray(seasonal_series, dtype=float)

    # fallback in case any seasonal values are missing
    if np.isnan(pred_seasonal).any():
        pred_seasonal = np.where(np.isnan(pred_seasonal), pred_persistence, pred_seasonal)

    # =====================================================
    # RIDGE
    # =====================================================
    print("Training Ridge model...")

    ridge = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    ridge.fit(X_train, y_train)
    pred_ridge = np.asarray(ridge.predict(X_test), dtype=float)

    # =====================================================
    # XGBOOST
    # =====================================================
    print("Training XGBoost model...")

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=1.0,
        objective="reg:squarederror",
        random_state=0,
        verbosity=0,
    )

    xgb.fit(X_train, y_train)
    pred_xgb = np.asarray(xgb.predict(X_test), dtype=float)
    train_pred_xgb = np.asarray(xgb.predict(X_train), dtype=float)

    print("\n[DEBUG] XGBoost Overfitting Check:")
    print("Train MAE:", float(np.mean(np.abs(np.asarray(y_train) - train_pred_xgb))))
    print("Test MAE :", float(np.mean(np.abs(np.asarray(y_test) - pred_xgb))))

    # =====================================================
    # METRICS
    # =====================================================
    y_true = np.asarray(y_test, dtype=float)

    rows = [
        {"model": "Persistence", **compute_metrics(y_true, pred_persistence)},
        {"model": "Seasonal", **compute_metrics(y_true, pred_seasonal)},
        {"model": "Ridge", **compute_metrics(y_true, pred_ridge)},
        {"model": "XGBoost", **compute_metrics(y_true, pred_xgb)},
    ]

    metrics_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    preds_df = pd.DataFrame(
        {
            "date": dates_test,
            "actual": y_true,
            "persistence": pred_persistence,
            "seasonal": pred_seasonal,
            "ridge": pred_ridge,
            "xgboost": pred_xgb,
        }
    )

    return metrics_df, preds_df, ridge, xgb


# =========================================================
# SAVE OUTPUTS
# =========================================================
def save_outputs(
    metrics_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    ridge: Pipeline,
    xgb: XGBRegressor,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    preds_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    dump(ridge, MODELS_DIR / "ridge.joblib")
    xgb.save_model(str(MODELS_DIR / "xgboost.json"))

    print("\nSaved outputs and models.")


# =========================================================
# PLOTS
# =========================================================
def create_plots(preds_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Forecast plot
    plt.figure(figsize=(10, 5))
    plt.plot(preds_df["date"], preds_df["actual"], label="Actual")
    plt.plot(preds_df["date"], preds_df["ridge"], label="Ridge")
    plt.plot(preds_df["date"], preds_df["xgboost"], label="XGBoost")
    plt.legend()
    plt.title("Forecast vs Actual")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "forecast.png", dpi=200)
    plt.close()

    # Metrics plot
    plt.figure(figsize=(8, 5))
    x = np.arange(len(metrics_df))
    plt.bar(x, metrics_df["MAE"])
    plt.xticks(x, metrics_df["model"].tolist())
    plt.title("Model Comparison (MAE)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metrics.png", dpi=200)
    plt.close()

    print("Saved plots.")