import os
import numpy as np
import pandas as pd
import joblib

# =========================
# CONFIGURATION
# =========================
PROJECT_ROOT = r"D:\BCI\MILimbEEG"
METADATA_PATH = os.path.join(PROJECT_ROOT, "data2", "metadata.xlsx")
FIR_ROOT_DIR = os.path.join(PROJECT_ROOT, "fir_dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ml_dataset")

# Output files for RAW data (CSP requires time-series, not features)
X_RAW_PATH = os.path.join(OUTPUT_DIR, "X_raw.npy")
Y_PATH = os.path.join(OUTPUT_DIR, "y_csp.npy")

# Target Classes: 3=DLF (Dorsiflexion Left Foot), 5=DRF (Dorsiflexion Right Foot)
TARGET_LABELS = [3, 5]

# Sampling Rate
FS = 125 

def main():
    print("Starting Raw Data Extraction for CSP (Leg Movements)...")
    
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata not found at {METADATA_PATH}")
        return
        
    df_meta = pd.read_excel(METADATA_PATH)
    print(f"Loaded {len(df_meta)} rows from metadata.")
    
    # Filter for DLF and DRF
    df_meta = df_meta[df_meta['task_label_encoded'].isin(TARGET_LABELS)]
    print(f"Filtered to {len(df_meta)} rows (DLF + DRF only).")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_list = []
    y_list = []
    
    processed_count = 0
    missing_count = 0
    
    # We need to ensure all files have the same length. 
    # MNE epochs usually require fixed length. 
    # Let's track lengths and truncate/pad if necessary, or just stack if they represent same duration trials.
    # Assuming trials are roughly same length, we'll check the first one.
    target_length = None

    for idx, row in df_meta.iterrows():
        # Path Resolution
        local_url = row['local_url']
        filename = os.path.basename(local_url)
        patient_num = row['patient_number']
        fir_file_path = os.path.join(FIR_ROOT_DIR, f"S{patient_num}", f"f_{filename}")
        
        if not os.path.exists(fir_file_path):
            missing_count += 1
            if missing_count <= 5:
                print(f"Warning: FIR file not found: {fir_file_path}")
            continue

        try:
            # Read CSV
            df_fir = pd.read_csv(fir_file_path)
            
            # Extract ALL channels (no heuristic selection for FBCSP)
            # Column 0 is Index/Time, Columns 1..N are channels
            data = df_fir.iloc[:, 1:].values.T  # (n_channels, n_times)
            
            # Check/Set Length (use full trial)
            if target_length is None:
                target_length = data.shape[1]
                print(f"Target trial length set to: {target_length} samples ({target_length/FS:.2f}s)")
            
            # Handle length mismatch
            if data.shape[1] > target_length:
                data = data[:, :target_length]
            elif data.shape[1] < target_length:
                pad_width = target_length - data.shape[1]
                data = np.pad(data, ((0,0), (0, pad_width)), mode='constant')
            
            X_list.append(data)
            y_list.append(row['task_label_encoded'])
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files...")
                
        except Exception as e:
            print(f"Error reading {fir_file_path}: {e}")
            continue

    if processed_count == 0:
        print("No files processed. Exiting.")
        return

    # Convert to Numpy Arrays
    # Format: (n_epochs, n_channels, n_times)
    X = np.stack(X_list)
    y = np.array(y_list)
    
    print(f"Raw Data Shape: {X.shape}") # Expect (N, Channels, Time)
    print(f"Labels Shape: {y.shape}")
    
    print(f"Saving to {OUTPUT_DIR}...")
    np.save(X_RAW_PATH, X)
    np.save(Y_PATH, y)
    print("Done.")

if __name__ == "__main__":
    main()
