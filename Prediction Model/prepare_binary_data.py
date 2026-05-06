import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import joblib
from sklearn.preprocessing import StandardScaler
import json

# =========================
# CONFIGURATION
# =========================

PROJECT_ROOT = r"D:\BCI\MILimbEEG"
METADATA_PATH = os.path.join(PROJECT_ROOT, "data2", "metadata.xlsx")
FIR_ROOT_DIR = os.path.join(PROJECT_ROOT, "fir_dataset")

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "binary_ml_dataset")
X_FILE = os.path.join(OUTPUT_DIR, "X_binary.npy")
Y_FILE = os.path.join(OUTPUT_DIR, "y_binary.npy")
SCALER_FILE = os.path.join(OUTPUT_DIR, "scaler_binary.joblib")
FEATURE_NAMES_FILE = os.path.join(OUTPUT_DIR, "feature_names_binary.json")

FS = 125  # Sampling Frequency (Hz)

# Binary Classes
# 1: CLH (Closed Left Hand) -> Movement
# 7: Rest -> Rest
TARGET_LABELS = [1, 7]

# =========================
# FEATURE CALCULATION FUNCTIONS
# =========================

def calculate_power(x):
    return np.mean(x**2)

def calculate_variance(x):
    return np.var(x)

def calculate_rms(x):
    return np.sqrt(np.mean(x**2))

def calculate_spectral_entropy(x, fs):
    try:
        f, psd = signal.welch(x, fs=fs, nperseg=min(len(x), 256))
        psd_sum = np.sum(psd)
        if psd_sum == 0:
            return 0
        psd_norm = psd / psd_sum
        se = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        return se
    except Exception:
        return 0

def calculate_band_power(x, fs, band):
    try:
        f, psd = signal.welch(x, fs=fs, nperseg=min(len(x), 256))
        idx_band = np.logical_and(f >= band[0], f <= band[1])
        band_power = np.mean(psd[idx_band])
        return band_power
    except Exception:
        return 0

# =========================
# MAIN PROCESSING LOOP
# =========================

def main():
    print("Starting Binary Feature Extraction (Movement vs Rest)...")
    
    # 1. Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata not found at {METADATA_PATH}")
        return
        
    df_meta = pd.read_excel(METADATA_PATH)
    print(f"Loaded {len(df_meta)} rows from metadata.")

    # Filter for target labels
    df_meta = df_meta[df_meta['task_label_encoded'].isin(TARGET_LABELS)]
    print(f"Filtered to {len(df_meta)} rows (Movement and Rest only).")

    # 2. Check Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_list = []
    y_list = []
    feature_names = []
    
    processed_count = 0
    missing_count = 0

    print("Extracting features...")

    for idx, row in df_meta.iterrows():
        local_url = row['local_url']
        filename = os.path.basename(local_url)
        patient_num = row['patient_number']
        fir_file_path = os.path.join(FIR_ROOT_DIR, f"S{patient_num}", f"f_{filename}")
        
        if not os.path.exists(fir_file_path):
            missing_count += 1
            continue

        try:
            df_fir = pd.read_csv(fir_file_path)
        except Exception as e:
            print(f"Error reading {fir_file_path}: {e}")
            continue

        feature_cols = df_fir.columns[1:] 
        file_features = []
        capture_names = (len(feature_names) == 0)

        for col in feature_cols:
            signal_data = pd.to_numeric(df_fir[col], errors='coerce').fillna(0).values
            
            p = calculate_power(signal_data)
            v = calculate_variance(signal_data)
            r = calculate_rms(signal_data)
            se = calculate_spectral_entropy(signal_data, FS)
            mu = calculate_band_power(signal_data, FS, (8, 13))
            beta = calculate_band_power(signal_data, FS, (13, 30))
            
            file_features.extend([p, v, r, se, mu, beta])
            
            if capture_names:
                feature_names.extend([
                    f"{col}_Power", f"{col}_Variance", f"{col}_RMS",
                    f"{col}_Entropy", f"{col}_Mu", f"{col}_Beta"
                ])
        
        X_list.append(file_features)
        y_list.append(row['task_label_encoded'])
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} files...")

    if processed_count == 0:
        print("Error: No files were processed.")
        return

    print(f"\nProcessing complete. Processed {processed_count} files.")

    X = np.array(X_list)
    y = np.array(y_list)

    # Normalize
    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save
    print(f"Saving binary dataset to {OUTPUT_DIR}...")
    np.save(X_FILE, X_scaled)
    np.save(Y_FILE, y)
    joblib.dump(scaler, SCALER_FILE)
    
    with open(FEATURE_NAMES_FILE, 'w') as f:
        json.dump(feature_names, f, indent=4)

    print("[Done] Binary data preparation complete.")

if __name__ == "__main__":
    main()
