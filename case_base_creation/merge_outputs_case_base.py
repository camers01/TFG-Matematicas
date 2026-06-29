import pandas as pd
import os

def merge_outputs_case_base():

    # 1. Define paths relative to the src/ directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    CASE_BASE_PREV_CSV = os.path.join(RAW_DATA_DIR, "case_base_prev.csv")
    RAW_OUTPUTS_CSV = os.path.join(RAW_DATA_DIR, "raw_case_outputs.csv")
    CASE_BASE_CSV = os.path.join(PROCESSED_DATA_DIR, "case_base.csv")

    # 2. Load the CSVs, enforcing string type for IDs to preserve leading zeros
    try:
        df_base = pd.read_csv(CASE_BASE_PREV_CSV, dtype={'id': str})
        df_raw = pd.read_csv(RAW_OUTPUTS_CSV, dtype={'case_id': str})
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 3. Pivot the raw outputs dataframe
    # This transforms the 'model' rows into columns containing their respective 'insight' values
    df_pivot = df_raw.pivot(index='case_id', columns='model', values='insight').reset_index()

    # 4. Rename the columns to match your exact requested format
    rename_mapping = {
        'Qwen-3-VL-8B': 'qwen_insight',
        'Pixtral-12B': 'pixtral_insight',
        'Idefics3-8B': 'idefics_insight'
    }
    df_pivot = df_pivot.rename(columns=rename_mapping)

    # Determine which columns were actually successfully generated
    available_cols = [col for col in rename_mapping.values() if col in df_pivot.columns]

    # 5. Fill existing columns in df_base with the new insights

    # Pandas .update() replaces NaNs, but ignores empty strings (""), so we temporarily convert 
    # any empty string to NaN in the base DataFrame so it knows what to overwrite
    for col in available_cols:
        if col in df_base.columns:
            df_base[col] = df_base[col].astype(object)
            df_base[col] = df_base[col].replace("", pd.NA)

    # Set indices to align the data perfectly ('id' matches 'case_id')
    df_base.set_index('id', inplace=True)
    df_pivot.set_index('case_id', inplace=True)

    # Update df_base in-place, strictly filling the corresponding columns without altering the order
    df_base.update(df_pivot[available_cols])

    # Reset the index to turn 'id' back into a normal column
    df_base.reset_index(inplace=True)

    # Convert any remaining NaNs back to empty strings
    for col in available_cols:
        if col in df_base.columns:
            df_base[col] = df_base[col].fillna("")

    # 6. Save to a new copy, leaving the original intact
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_base['id'] = df_base['id'].astype(str).str.zfill(6) # Force the 'id' format
    df_base.to_csv(CASE_BASE_CSV, index=False)

    print(f"Successfully created {CASE_BASE_CSV} with the insight columns filled.")

if __name__ == "__main__":
    merge_outputs_case_base()