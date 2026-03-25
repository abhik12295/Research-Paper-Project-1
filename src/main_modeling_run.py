import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.modeling import train_and_evaluate, save_outputs, create_plots
from src.interpretation import (
    ridge_feature_importance,
    xgb_feature_importance,
    plot_feature_importance,
)

from src.interpretation import (
    ridge_feature_importance,
    xgb_feature_importance,
    plot_feature_importance,
)

def main():
    path = PROCESSED_DATA_DIR / "model_dataset.csv"

    df = pd.read_csv(path, parse_dates=["date"])

    metrics_df, preds_df, ridge, xgb = train_and_evaluate(
        df,
        target_col="cass_expenditures",
        test_horizon=24,
    )

    print("\n=== MODEL RESULTS ===")
    print(metrics_df.to_string(index=False))

    save_outputs(metrics_df, preds_df, ridge, xgb)
    create_plots(preds_df, metrics_df)

    # FEATURE Importance
    feature_names = [
        "cass_expenditures",
        "cass_expenditures_lag1",
        "cass_expenditures_lag2",
        "cass_expenditures_lag3",
        "cass_expenditures_lag6",
        "cass_expenditures_lag12",
        "cass_expenditures_rollmean3",
        "cass_expenditures_rollmean6",
        "cass_expenditures_rollstd3",
        "month_sin",
        "month_cos",
        "pcu484484",
        "wpu057303",
    ]

    # keep only existing columns
    feature_names = [c for c in feature_names if c in df.columns]

    ridge_imp = ridge_feature_importance(ridge, feature_names)
    xgb_imp = xgb_feature_importance(xgb, feature_names)

    print("\nTop Ridge Features:")
    print(ridge_imp.head(10))

    print("\nTop XGBoost Features:")
    print(xgb_imp.head(10))

    plot_feature_importance(
        ridge_imp,
        "Ridge Feature Importance",
        "ridge_importance.png"
    )

    plot_feature_importance(
        xgb_imp,
        "XGBoost Feature Importance",
        "xgb_importance.png"
    )


if __name__ == "__main__":
    main()