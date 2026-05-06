import os
import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt

# =========================
# CONFIGURATION
# =========================

PROJECT_ROOT = "/home/snippy/Desktop/Projects/MiLimb EEG"
RAW_DIR = os.path.join(PROJECT_ROOT, "datapoints", "raw")
FIR_DIR = os.path.join(PROJECT_ROOT, "datapoints", "fir")

FS = 125  # sampling frequency (Hz)
NYQUIST = FS / 2

# Frequency bands (Hz)
BANDS = {
    "a": (8, 12),    # Alpha
    "b": (12, 30),   # Beta
    "g": (30, 50)    # Gamma
}

FILTER_ORDER = 101  # odd number → linear phase FIR

# Create output directory if it doesn't exist
os.makedirs(FIR_DIR, exist_ok=True)

# =========================
# FIR FILTER FUNCTION
# =========================

def bandpass_fir(signal, lowcut, highcut, fs, order):
    """
    Applies a linear-phase FIR bandpass filter to a 1D signal.
    Uses zero-phase filtering to avoid time shifts.
    """
    taps = firwin(
        numtaps=order,
        cutoff=[lowcut / (fs / 2), highcut / (fs / 2)],
        pass_zero=False
    )
    return filtfilt(taps, [1.0], signal)

# =========================
# PROCESSING PIPELINE
# =========================

# Get first patient folder
patient_folders = sorted(
    [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
)

first_patient_path = os.path.join(RAW_DIR, patient_folders[0])

# Get first 5 CSV files
csv_files = sorted(
    [f for f in os.listdir(first_patient_path) if f.endswith(".csv")]
)[:5]

for csv_file in csv_files:
    input_path = os.path.join(first_patient_path, csv_file)

    # Load CSV
    df = pd.read_csv(input_path)

    # First column = sample index
    # Initialize output dataframe with the sample index column
    output_df = pd.DataFrame({
        df.columns[0]: df.iloc[:, 0].values
    })


    # Process each electrode column
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

            new_col_name = f"{col}_{band_label}"
            output_df[new_col_name] = filtered_signal

    # Save output
    output_filename = os.path.splitext(csv_file)[0] + "_fir.csv"
    output_path = os.path.join(FIR_DIR, output_filename)

    output_df.to_csv(output_path, index=False)

    print(f"Processed and saved: {output_filename}")

print("All files processed successfully.")
