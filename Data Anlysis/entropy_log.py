import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

# =========================
# PATHS
# =========================
PROJECT_ROOT = "/home/snippy/Desktop/Projects/Milimb_eeg"
DATA_DIR = os.path.join(PROJECT_ROOT, "datapoints")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
output_path=os.path.join(PLOT_DIR , "entropy" , "moving_average_entropy.png")
os.makedirs(PLOT_DIR, exist_ok=True)

METADATA_PATH = os.path.join(DATA_DIR, "metadata.xlsx")

# =========================
# PARAMETERS
# =========================
FS = 125
NUM_ELECTRODES = 16

BANDS = ["alpha", "beta", "gamma"]
BAND_SUFFIX = {"alpha": "_a", "beta": "_b", "gamma": "_g"}

WINDOW_SEC = 0.5
STEP_SEC = 0.1

WINDOW_SAMPLES = int(WINDOW_SEC * FS)
STEP_SAMPLES = int(STEP_SEC * FS)

EPSILON = 1e-12

MA_WINDOW = 5  # moving average window (in entropy samples)

# =========================
# SPECTRAL ENTROPY FUNCTION
# =========================
def spectral_entropy(signal, fs):
    _, psd = welch(signal, fs=fs, nperseg=len(signal))
    psd = psd + EPSILON
    psd_norm = psd / np.sum(psd)
    return -np.sum(psd_norm * np.log2(psd_norm))

# =========================
# MOVING AVERAGE FUNCTION
# =========================
def moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode="same")


# =========================
# LOAD METADATA
# =========================
metadata = pd.read_excel(METADATA_PATH)
filtered_relative_path = metadata.loc[64, "filtered_url"]
filtered_path = os.path.join(PROJECT_ROOT, filtered_relative_path)

# =========================
# LOAD FILTERED EEG
# =========================
df = pd.read_csv(filtered_path)

df = df.iloc[1:, :]      # drop electrode number row
df = df.iloc[:, 1:]      # drop serial-number column
df = df.astype(float)

num_samples = len(df)

# =========================
# TIME AXIS
# =========================
window_centers = np.arange(
    WINDOW_SAMPLES // 2,
    num_samples - WINDOW_SAMPLES // 2,
    STEP_SAMPLES
)
time_axis = window_centers / FS

# =========================
# COMPUTE ENTROPY
# =========================
entropy_time = {
    band: np.zeros((NUM_ELECTRODES, len(time_axis)))
    for band in BANDS
}

for band in BANDS:
    suffix = BAND_SUFFIX[band]

    for elec in range(NUM_ELECTRODES):
        signal = df[f"{elec}{suffix}"].values
        idx = 0

        for start in range(0, num_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            window = signal[start:start + WINDOW_SAMPLES]
            entropy_time[band][elec, idx] = spectral_entropy(window, FS)
            idx += 1

        # =========================
        # APPLY MOVING AVERAGE (SMOOTHING)
        # =========================
        entropy_time[band][elec] = moving_average(
            entropy_time[band][elec],
            MA_WINDOW
        )

# =========================
# COLOR MAP
# =========================
cmap = plt.get_cmap("tab20")
electrode_colors = {e: cmap(e) for e in range(NUM_ELECTRODES)}

# =========================
# PLOTTING
# =========================
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for ax, band in zip(axes, BANDS):
    for elec in range(NUM_ELECTRODES):
        ax.plot(
            time_axis,
            entropy_time[band][elec],
            color=electrode_colors[elec],
            linewidth=1.3
        )

    ax.set_title(f"{band.capitalize()} Band Spectral Entropy (MA smoothed)")
    ax.set_ylabel("Spectral Entropy")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")

# =========================
# LEGEND
# =========================
handles = [
    plt.Line2D([0], [0], color=electrode_colors[e], lw=2, label=f"E{e}")
    for e in range(NUM_ELECTRODES)
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.91),
    ncol=8,
    fontsize=9,
    frameon=False
)

fig.suptitle(
    "Time-Resolved Spectral Entropy (0.5 s Window + Moving Average)",
    fontsize=14,
    y=0.97
)

plt.tight_layout(rect=[0, 0.05, 1, 0.86])
plt.savefig(output_path, dpi=300)
plt.close()
plt.close()
