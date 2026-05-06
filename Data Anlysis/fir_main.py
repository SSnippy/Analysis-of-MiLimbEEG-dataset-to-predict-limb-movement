import os
import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt

# =========================
# CONFIGURATION
# =========================

PROJECT_ROOT = "/home/snippy/Desktop/Projects/Milimb_eeg"
RAW_DIR = os.path.join(PROJECT_ROOT, "datapoints", "raw")
FIR_DIR = os.path.join(PROJECT_ROOT, "datapoints", "fir")

FS = 125  # sampling frequency (Hz)

# Frequency bands (Hz)
BANDS = {
    "a": (8, 12),    # Alpha
    "b": (12, 30),   # Beta
    "g": (30, 50)    # Gamma
}

FILTER_ORDER = 101  # odd → linear-phase FIR

# Create top-level FIR directory if it doesn't exist
os.makedirs(FIR_DIR, exist_ok=True)

# =========================
# FIR FILTER FUNCTION
# =========================

def bandpass_fir(signal, lowcut, highcut, fs, order):
    """
    Linear-phase FIR bandpass filter applied in zero-phase manner.
    """
    taps = firwin(
        numtaps=order,
        cutoff=[lowcut / (fs / 2), highcut / (fs / 2)],
        pass_zero=False
    )
    return filtfilt(taps, [1.0], signal)

# =========================
# FULL DATASET PROCESSING
# =========================

# Loop over all patient folders (S1 ... S60)
patient_folders = sorted(
    d for d in os.listdir(RAW_DIR)
    if os.path.isdir(os.path.join(RAW_DIR, d))
)

for patient in patient_folders:
    raw_patient_path = os.path.join(RAW_DIR, patient)

    # Create corresponding FIR patient folder: f_Sx
    fir_patient_name = f"f_{patient}"
    fir_patient_path = os.path.join(FIR_DIR, fir_patient_name)
    os.makedirs(fir_patient_path, exist_ok=True)

    # Get all CSV files for this patient
    csv_files = sorted(
        f for f in os.listdir(raw_patient_path)
        if f.endswith(".csv")
    )

    for csv_file in csv_files:
        input_path = os.path.join(raw_patient_path, csv_file)

        # Load raw CSV
        df = pd.read_csv(input_path)

        # Initialize output DataFrame with sample index column
        output_df = pd.DataFrame({
            df.columns[0]: df.iloc[:, 0].values
        })

        # Process each electrode independently
        for col in df.columns[1:]:
            signal = df[col].values.astype(float)

            for band_label, (low, high) in BANDS.items():
                filtered_signal = bandpass_fir(
                    signal,
                    lowcut=low,
                    highcut=high,
                    fs=FS,
                    order=FILTER_ORDER
                )

                output_df[f"{col}_{band_label}"] = filtered_signal

        # Save output file with f_ prefix
        output_filename = f"f_{csv_file}"
        output_path = os.path.join(fir_patient_path, output_filename)

        output_df.to_csv(output_path, index=False)

        print(f"[{patient}] Processed: {csv_file}")

print("✅ All patients and files processed successfully.")
