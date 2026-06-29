import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
from tqdm import tqdm


# 1. PATH SETUP

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Case base CSV file
TARGET_CSV = os.path.join(RAW_DATA_DIR, "case_base_prev.csv")

# Directory for visual embeddings
VISUAL_EMB_DIR = os.path.join(PROCESSED_DATA_DIR, "embeddings_visual")


# 2. CONFIGURATION

BATCH_SIZE = 4
MODEL_NAME = "ahmed-masry/chartgemma"

# Use GPU if available, otherwise fallback to CPU safely
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 3. EXECUTION

def main():

    print(f"Loading data from: {TARGET_CSV}")
    try:
        df = pd.read_csv(TARGET_CSV, dtype={'id': str})
    except FileNotFoundError:
        print("ERROR: Target CSV not found.")
        return

    print(f"Loading Vision Model onto {DEVICE.upper()}...")
    
    # Load the specific PaliGemma architecture for ChartGemma
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    os.makedirs(VISUAL_EMB_DIR, exist_ok=True)

    # Process in chunks to prevent memory leaks
    for start_idx in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing Image Batches"):
        
        batch_df = df.iloc[start_idx:start_idx + BATCH_SIZE]
        images = []
        valid_indices = []

        # 1. Safely load images
        for idx, row in batch_df.iterrows():
            img_abs_path = os.path.join(RAW_DATA_DIR, row['img_path'])
            try:
                img = Image.open(img_abs_path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"\nWARNING: Could not load image {img_abs_path}. Error: {e}")
                continue
        
        if not images:
            continue

        # 2. Process through the Vision Tower
        with torch.no_grad():

            # Prepare pixels (PaliGemma format) - only using image_processor (no text inputs here)
            inputs = processor.image_processor(images=images, return_tensors="pt").to(DEVICE)
            
            # Extract the vector from the vision tower (inputs directly contains 'pixel_values', instead of inputs.pixel_values))
            vision_outputs = model.vision_tower(inputs["pixel_values"])
            
            # Take the mean across the sequence dimension (dim=1) 
            raw_embeddings = vision_outputs.last_hidden_state.mean(dim=1)

            # L2 Normalization to save computation time in Phase 2
            normalized_embeddings = F.normalize(raw_embeddings, p=2, dim=1)

            # Move to CPU and convert to Numpy for saving
            pooled_embeddings = normalized_embeddings.cpu().numpy()

        # 3. Save the embeddings
        for i, df_idx in enumerate(valid_indices):
            case_id = str(df.at[df_idx, 'id']).zfill(6)
            save_path = os.path.join(VISUAL_EMB_DIR, f"{case_id}.npy")
            np.save(save_path, pooled_embeddings[i])
        
        # Clear GPU cache to prevent out-of-memory errors on long runs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("Process Completed.")

if __name__ == "__main__":
    main()