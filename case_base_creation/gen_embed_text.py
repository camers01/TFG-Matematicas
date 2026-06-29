import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. PATH SETUP

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# Case base CSV file
TARGET_CSV = os.path.join(RAW_DATA_DIR, "case_base_prev.csv")

# Directories for the new embeddings
DOMAIN_EMB_DIR = os.path.join(PROCESSED_DATA_DIR, "embeddings_domain")
TASK_EMB_DIR = os.path.join(PROCESSED_DATA_DIR, "embeddings_task")


# 2. EXECUTION

def main():

    print(f"Loading data from: {TARGET_CSV}")
    try:
        # Enforce 'id' as a string
        df = pd.read_csv(TARGET_CSV, dtype={'id': str}) 
    except FileNotFoundError:
        print("ERROR: Target CSV not found.")
        return

    print("Loading the MiniLM model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract text as lists
    domain_texts = df['domain'].astype(str).tolist()
    task_texts = df['analytical_task'].astype(str).tolist()

    print(f"Computing embeddings for {len(domain_texts)} domains...")
    domain_embeddings = model.encode(domain_texts, show_progress_bar=True, normalize_embeddings=True) # Batch encoding is significantly faster than looping row by row

    print(f"Computing embeddings for {len(task_texts)} tasks...")
    task_embeddings = model.encode(task_texts, show_progress_bar=True, normalize_embeddings=True) # Batch encoding is significantly faster than looping row by row

    print("Saving .npy files")

    os.makedirs(DOMAIN_EMB_DIR, exist_ok=True)
    os.makedirs(TASK_EMB_DIR, exist_ok=True)

    for i, row in df.iterrows():

        # Ensure the ID is exactly 6 digits (e.g., '000001')
        case_id = str(row['id']).zfill(6) 

        # Define absolute paths for saving the files
        domain_abs_path = os.path.join(DOMAIN_EMB_DIR, f"{case_id}.npy")
        task_abs_path = os.path.join(TASK_EMB_DIR, f"{case_id}.npy")

        # Save the numpy arrays to the respective directories
        np.save(domain_abs_path, domain_embeddings[i])
        np.save(task_abs_path, task_embeddings[i])

    print("Process Completed.")

if __name__ == "__main__":
    main()