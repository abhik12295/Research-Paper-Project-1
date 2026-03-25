import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.features import (
    build_feature_dataset,
    validate_feature_dataset,
    save_feature_dataset,
)


def main() -> None:
    input_path = PROCESSED_DATA_DIR / "base_with_fred.csv"

    if not input_path.exists():
        raise FileNotFoundError("Run previous step (FRED merge) first.")

    print("\nLoading dataset...")
    df = pd.read_csv(input_path, parse_dates=["date"])

    print(f"Input rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    print("\nBuilding feature dataset...")
    feature_df = build_feature_dataset(df, target_col="cass_expenditures")

    print("\nValidating feature dataset...")
    summary = validate_feature_dataset(feature_df)

    print("\nValidation summary:")
    print(f"Rows       : {summary['row_count']}")
    print(f"Columns    : {summary['column_count']}")
    print(f"Start date : {summary['start_date']}")
    print(f"End date   : {summary['end_date']}")

    output_path = save_feature_dataset(feature_df)

    print(f"\nSaved model dataset to: {output_path}")

    print("\nPreview:")
    print(feature_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()