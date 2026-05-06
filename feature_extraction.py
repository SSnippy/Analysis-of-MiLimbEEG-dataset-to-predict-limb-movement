
import os
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.stats import entropy
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ml_dataset")
X_FILE = os.path.join(OUTPUT_DIR, "X.npy")
Y_FILE = os.path.join(OUTPUT_DIR, "y.npy")
SCALER_FILE = os.path.join(OUTPUT_DIR, "scaler.joblib")
FEATURE_NAMES_FILE = os.path.join(OUTPUT_DIR, "feature_names.json")

FS = 125  # Sampling Frequency (Hz)

# =========================
# FEATURE CALCULATION FUNCTIONS
# =========================

def calculate_power(x):
    """Mean Square Power (Time-domain)"""
    return np.mean(x**2)

def calculate_variance(x):
    """Variance (Time-domain)"""
    return np.var(x)

def calculate_rms(x):
    """Root Mean Square (Time-domain)"""
    return np.sqrt(np.mean(x**2))

def calculate_spectral_entropy(x, fs):
    """Spectral Entropy (Frequency-domain: via Welch's PSD)"""
    try:
        # Compute PSD
        f, psd = signal.welch(x, fs=fs, nperseg=min(len(x), 256))
        # Normalize PSD to get probability distribution
        psd_sum = np.sum(psd)
        if psd_sum == 0:
            return 0
        psd_norm = psd / psd_sum
        # Compute Entropy (base 2 for bits)
        se = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        return se
    except Exception:
        return 0

# =========================
# MAIN PROCESSING LOOP
# =========================

def main():
    print("Starting Feature Extraction for Machine Learning...")
    
    # 1. Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata not found at {METADATA_PATH}")
        return
        
    df_meta = pd.read_excel(METADATA_PATH)
    print(f"Loaded {len(df_meta)} rows from metadata.")

    # 2. Check Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_list = []
    y_list = []
    feature_names = []
    
    # Track processed count
    processed_count = 0
    missing_count = 0

    print("Extracting features...")

    # Iterate through metadata rows
    for idx, row in df_meta.iterrows():
        # --- PATH RESOLUTION ---
        # 1. Start with the original raw filename from metadata
        local_url = row['local_url']
        filename = os.path.basename(local_url)
        
        # 2. Construct the expected FIR file path
        # fir_main.py saved to: FIR_ROOT_DIR / S{patient_num} / f_{filename}
        patient_num = row['patient_number']
        fir_file_path = os.path.join(FIR_ROOT_DIR, f"S{patient_num}", f"f_{filename}")
        
        # 3. Validation
        if not os.path.exists(fir_file_path):
            # Try checking without subject folder if fir_main structure differs?
            # But we implemented it with S folders.
            missing_count += 1
            if missing_count <= 5: # Limit spam
                print(f"Warning: FIR file not found: {fir_file_path}")
            continue

        # --- LOAD DATA ---
        try:
            df_fir = pd.read_csv(fir_file_path)
        except Exception as e:
            print(f"Error reading {fir_file_path}: {e}")
            continue

        # --- FEATURE EXTRACTION ---
        # We process ALL numerical columns except the first one (typically index/time)
        # Assuming format: Index, Ch1_a, Ch1_b, Ch1_g, Ch2_a...
        
        # Identify columns to process (all except first)
        feature_cols = df_fir.columns[1:] 
        
        file_features = []
        
        # For the FIRST file, we capture feature names
        capture_names = (len(feature_names) == 0)

        for col in feature_cols:
            signal_data = pd.to_numeric(df_fir[col], errors='coerce').fillna(0).values
            
            # Calculate features
            p = calculate_power(signal_data)
            v = calculate_variance(signal_data)
            r = calculate_rms(signal_data)
            se = calculate_spectral_entropy(signal_data, FS)
            
            file_features.extend([p, v, r, se])
            
            if capture_names:
                feature_names.extend([
                    f"{col}_Power",
                    f"{col}_Variance",
                    f"{col}_RMS",
                    f"{col}_Entropy"
                ])
        
        # Append to X and y
        X_list.append(file_features)
        y_list.append(row['task_label_encoded'])
        
        processed_count += 1
        if processed_count % 500 == 0:
            print(f"Processed {processed_count}/{len(df_meta)} files...")

    # =========================
    # POST-PROCESSING
    # =========================
    
    if processed_count == 0:
        print("Error: No files were processed. Check FIR directory paths.")
        return

    print(f"\nProcessing complete. Processed {processed_count} files.")
    if missing_count > 0:
        print(f"Warning: {missing_count} files were missing.")

    # Convert to Numpy Arrays
    X = np.array(X_list)
    y = np.array(y_list)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Normalize (StandardScaler)
    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save
    print(f"Saving dataset to {OUTPUT_DIR}...")
    np.save(X_FILE, X_scaled)
    np.save(Y_FILE, y)
    joblib.dump(scaler, SCALER_FILE)
    
    with open(FEATURE_NAMES_FILE, 'w') as f:
        json.dump(feature_names, f, indent=4)

    print("[Done] Feature Extraction Complete. Ready for ML training.")
    print(f"Files saved:\n- {X_FILE}\n- {Y_FILE}\n- {SCALER_FILE}\n- {FEATURE_NAMES_FILE}")

if __name__ == "__main__":
    main()
