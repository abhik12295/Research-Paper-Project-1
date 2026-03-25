# 1.
# from data_ingestion import (
#     fetch_base_bts_file,
#     fetch_border_crossings_file,
#     fetch_bts_monthly_teu_file,
#     fetch_fred_bundle,
#     )

# def main() -> None:
#     print("Fetching freight analytics datasets...")

#     fetch_base_bts_file()
#     fetch_border_crossings_file()
#     fetch_bts_monthly_teu_file()
#     fetch_fred_bundle()

#     print("Done..")

# if __name__ == "__main__":
#     main()

# # 2
# from src.config import RAW_DATA_DIR
# from src.data_validation import inspect_excel_workbook, load_bts_figure4_base, validate_base_dataset, save_base_dataset

# def main() -> None:
#     xlsx_path = RAW_DATA_DIR / "Figure4_1.xlsx"
#     print("\n Inspecting workbook...")
#     workbook_info = inspect_excel_workbook(xlsx_path)
#     print(f"Sheet names: {workbook_info['sheet_names']}")
    
#     print("\nLoading BTS base dataset....")
#     base_df = load_bts_figure4_base(xlsx_path)

#     print("\nValidating base dataset....")
#     summary = validate_base_dataset(base_df)

#     print("Validation Summart:")
#     print(f"Rows       : {summary['row_count']}")
#     print(f"Columns    : {summary['column_count']}")
#     print(f"Start Date : {summary['start_date']}")
#     print(f"End Date   : {summary['end_date']}")
#     print(f"Columns    : {summary['columns']}")
#     print(f"Null counts: {summary['null_counts']}")

#     output_path = save_base_dataset(base_df)
#     print(f"\nSaved cleaned base dataset to: {output_path}")

#     print("\nPreview:")
#     print(base_df.head(10).to_string(index=False))

# if __name__ == "__main__":
#     main()

# 3.
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data_merge import (
    load_fred_bundle,
    merge_base_with_fred,
    save_processed_dataset,
    validate_merged_panel
)

def main() -> None:
    base_path = PROCESSED_DATA_DIR / "base_freight_index.csv"

    if not base_path.exists():
        raise FileNotFoundError(
            f"Base dataset not found at {base_path}. Run the BTS base parsing step first."
        )
    
    print("\nLoading base dataset...")
    base_df = __import__("pandas").read_csv(base_path, parse_dates=["date"])
    print(f"Base rows: {len(base_df)}")
    print(f"Base columns: {base_df.columns.tolist()}")

    print("\nLoading FRED bundle...")
    fred_df = load_fred_bundle(RAW_DATA_DIR)
    print(f"FRED rows: {len(fred_df)}")
    print(f"FRED columns: {fred_df.columns.tolist()}")

    print("\nMerging base dataset with FRED features...")
    merged_df = merge_base_with_fred(base_df, fred_df)

    print("\nValidating merged panel...")
    summary = validate_merged_panel(merged_df)
    print("Validation summary:")
    print(f"Rows       : {summary['row_count']}")
    print(f"Columns    : {summary['column_count']}")
    print(f"Start date : {summary['start_date']}")
    print(f"End date   : {summary['end_date']}")
    print(f"Columns    : {summary['columns']}")
    print(f"Null counts: {summary['null_counts']}")

    output_path = save_processed_dataset(merged_df, "base_with_fred.csv")
    print(f"\nSaved merged dataset to: {output_path}")

    print("\nPreview:")
    print(merged_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main() 