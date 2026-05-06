import os
import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================

PROJECT_ROOT = r"D:\BCI\MILimbEEG"
RAW_DIR =os.path.join(PROJECT_ROOT, "data")
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
    
    # =========================
    # PLOTTING
    # =========================
    PLOT_DIR = os.path.join(FIR_DIR, "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Plotting one representative channel (e.g., first channel processed)
    # We take the first column name from the original df (excluding time/index)
    first_col = df.columns[1] 
    
    plt.figure(figsize=(10, 8))
    
    # 1. Raw Signal
    plt.subplot(4, 1, 1)
    plt.plot(df[first_col].values, label='Raw', color='black', linewidth=0.8)
    plt.title(f"Signal Analysis: {first_col}")
    plt.ylabel("Raw (uV)")
    plt.legend(loc='upper right')
    
    # 2. Alpha
    plt.subplot(4, 1, 2)
    plt.plot(output_df[f"{first_col}_a"].values, label='Alpha (8-12Hz)', color='blue', linewidth=0.8)
    plt.ylabel("Alpha")
    plt.legend(loc='upper right')
    
    # 3. Beta
    plt.subplot(4, 1, 3)
    plt.plot(output_df[f"{first_col}_b"].values, label='Beta (12-30Hz)', color='green', linewidth=0.8)
    plt.ylabel("Beta")
    plt.legend(loc='upper right')
    
    # 4. Gamma
    plt.subplot(4, 1, 4)
    plt.plot(output_df[f"{first_col}_g"].values, label='Gamma (30-50Hz)', color='red', linewidth=0.8)
    plt.ylabel("Gamma")
    plt.xlabel("Sample Index")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plot_filename = os.path.splitext(csv_file)[0] + "_plot.png"
    plt.savefig(os.path.join(PLOT_DIR, plot_filename))
    plt.close()

print("All files processed successfully.")
