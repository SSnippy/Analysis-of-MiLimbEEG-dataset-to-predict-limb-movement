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

os.makedirs(PLOT_DIR, exist_ok=True)

METADATA_PATH = os.path.join(DATA_DIR, "metadata.xlsx")

# =========================
# PARAMETERS
# =========================
FS = 125
NUM_ELECTRODES = 16

WINDOW_SEC = 0.5
STEP_SEC = 0.1

WINDOW_SAMPLES = int(WINDOW_SEC * FS)
STEP_SAMPLES = int(STEP_SEC * FS)

EPSILON = 1e-12

# =========================
# SPECTRAL ENTROPY FUNCTION
# =========================
def spectral_entropy(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=len(signal))
    psd = psd + EPSILON
    psd_norm = psd / np.sum(psd)
    return -np.sum(psd_norm * np.log2(psd_norm))

# =========================
# LOAD METADATA
# =========================
metadata = pd.read_excel(METADATA_PATH)

# First movement instance
local_relative_path = metadata.loc[0, "local_url"]
local_path = os.path.join(PROJECT_ROOT, local_relative_path)

# =========================
# LOAD RAW EEG DATA
# =========================
df = pd.read_csv(local_path)

# Drop first row (electrode numbers)
df = df.iloc[1:, :]

# Drop serial-number column
df = df.iloc[:, 1:]

df = df.astype(float)
num_samples = len(df)  # should be 500

# =========================
# TIME AXIS (WINDOW CENTERS)
# =========================
window_centers = np.arange(
    WINDOW_SAMPLES // 2,
    num_samples - WINDOW_SAMPLES // 2,
    STEP_SAMPLES
)
time_axis = window_centers / FS

# =========================
# COMPUTE TIME-RESOLVED ENTROPY (RAW EEG)
# =========================
entropy_time = np.zeros((NUM_ELECTRODES, len(time_axis)))

for elec in range(NUM_ELECTRODES):
    signal = df.iloc[:, elec].values

    idx = 0
    for start in range(0, num_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        window = signal[start:start + WINDOW_SAMPLES]
        entropy_time[elec, idx] = spectral_entropy(window, FS)
        idx += 1

# =========================
# COLOR MAP (16 ELECTRODES)
# =========================
cmap = plt.get_cmap("tab20")
electrode_colors = {e: cmap(e) for e in range(NUM_ELECTRODES)}

# =========================
# PLOTTING
# =========================
fig, ax = plt.subplots(figsize=(14, 6))

for elec in range(NUM_ELECTRODES):
    ax.plot(
        time_axis,
        entropy_time[elec],
        color=electrode_colors[elec],
        linewidth=1.1,
        label=f"E{elec}"
    )

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Spectral Entropy")
ax.set_title("Raw EEG Time-Resolved Spectral Entropy (0.5 s Window)")
ax.grid(True, alpha=0.3)

# =========================
# LEGEND (BETWEEN TITLE AND PLOT)
# =========================
handles = [
    plt.Line2D([0], [0], color=electrode_colors[e], lw=2, label=f"E{e}")
    for e in range(NUM_ELECTRODES)
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.88),
    ncol=8,
    fontsize=9,
    frameon=False
)

plt.tight_layout(rect=[0, 0.05, 1, 0.82])

# =========================
# SAVE FIGURE
# =========================
output_path = os.path.join(
    PLOT_DIR,
    "time_entropy_raw_eeg_0p5s_colored.png"
)

plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved raw EEG entropy plot to:\n{output_path}")
