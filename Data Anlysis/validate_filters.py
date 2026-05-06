import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, welch
import os

# =========================
# CONFIGURATION
# =========================
FS = 125  # Hz
DURATION = 5  # seconds
FILTER_ORDER = 101
FIR_DIR = r"D:\BCI\MILimbEEG\datapoints\fir"

# =========================
# HELPER FUNCTIONS
# =========================

def bandpass_fir(signal, lowcut, highcut, fs, order):
    """
    Applies a linear-phase FIR bandpass filter to a 1D signal.
    """
    taps = firwin(
        numtaps=order,
        cutoff=[lowcut / (fs / 2), highcut / (fs / 2)],
        pass_zero=False
    )
    return filtfilt(taps, [1.0], signal)

def generate_synthetic_data(fs, duration):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Mix of 10Hz (Alpha), 20Hz (Beta), 40Hz (Gamma)
    # Each has amplitude 1.0
    sig = (1.0 * np.sin(2 * np.pi * 10 * t) +
           1.0 * np.sin(2 * np.pi * 20 * t) +
           1.0 * np.sin(2 * np.pi * 40 * t))
    return t, sig

# =========================
# 1. SYNTHETIC DATA TEST
# =========================
print("Running Synthetic Data Test...")
t, synthetic_signal = generate_synthetic_data(FS, DURATION)

# Filter
alpha = bandpass_fir(synthetic_signal, 8, 12, FS, FILTER_ORDER)
beta = bandpass_fir(synthetic_signal, 12, 30, FS, FILTER_ORDER)
gamma = bandpass_fir(synthetic_signal, 30, 50, FS, FILTER_ORDER)

# Output directory
VALIDATION_DIR = r"D:\BCI\MILimbEEG\validation_plots"
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Plot Synthetic Results
plt.figure(figsize=(10, 8))
plt.suptitle("Synthetic Data Test", fontsize=16)

plt.subplot(4, 1, 1)
plt.plot(t[:250], synthetic_signal[:250], color='black')
plt.title("Synthetic Input (10Hz + 20Hz + 40Hz)")
plt.ylabel("Amp")

plt.subplot(4, 1, 2)
plt.plot(t[:250], alpha[:250], color='blue')
plt.title("Extracted Alpha (Target: 10Hz)")
plt.ylabel("Amp")

plt.subplot(4, 1, 3)
plt.plot(t[:250], beta[:250], color='green')
plt.title("Extracted Beta (Target: 20Hz)")
plt.ylabel("Amp")

plt.subplot(4, 1, 4)
plt.plot(t[:250], gamma[:250], color='red')
plt.title("Extracted Gamma (Target: 40Hz)")
plt.ylabel("Amp")
plt.xlabel("Time (s)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
plt.savefig(os.path.join(VALIDATION_DIR, "validation_synthetic.png"))
print(f"Saved validation_synthetic.png to {VALIDATION_DIR}")
plt.close()

# =========================
# 2. REAL DATA SPECTRAL CHECK
# =========================
print("\nRunning Real Data Spectral Check...")

# Find a processed file
processed_files = [f for f in os.listdir(FIR_DIR) if f.endswith("_fir.csv")]
if not processed_files:
    print("No processed files found in FIR_DIR. Run fir_test_6.py first.")
    exit()

test_file = processed_files[0]
print(f"Analyzing file: {test_file}")
df = pd.read_csv(os.path.join(FIR_DIR, test_file))

# Pick the first channel available
# We look for columns ending in '_a'
alpha_cols = [c for c in df.columns if c.endswith('_a')]
if not alpha_cols:
    print("No alpha band columns found in the file.")
    exit()

first_a_col = alpha_cols[0]
target_col = first_a_col[:-2] # remove "_a" suffix
print(f"Targeting base column: {target_col}")

col_a = f"{target_col}_a"
col_b = f"{target_col}_b"
col_g = f"{target_col}_g"

# Compute PSD
f_a, Pxx_a = welch(df[col_a].values, fs=FS, nperseg=256)
f_b, Pxx_b = welch(df[col_b].values, fs=FS, nperseg=256)
f_g, Pxx_g = welch(df[col_g].values, fs=FS, nperseg=256)

# Plot PSD
plt.figure(figsize=(10, 6))
plt.semilogy(f_a, Pxx_a, label='Alpha Output', color='blue')
plt.semilogy(f_b, Pxx_b, label='Beta Output', color='green')
plt.semilogy(f_g, Pxx_g, label='Gamma Output', color='red')

# Add band markers
plt.axvspan(8, 12, color='blue', alpha=0.1, label='Alpha Band')
plt.axvspan(12, 30, color='green', alpha=0.1, label='Beta Band')
plt.axvspan(30, 50, color='red', alpha=0.1, label='Gamma Band')

plt.title("Power Spectral Density")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (dB/Hz)")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.xlim(0, 60)

plt.tight_layout()
plt.savefig(os.path.join(VALIDATION_DIR, "validation_real_psd.png"))
print(f"Saved validation_real_psd.png to {VALIDATION_DIR}")
plt.close()

print("Validation complete.")
