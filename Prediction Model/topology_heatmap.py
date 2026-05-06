import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib.colors import LinearSegmentedColormap

neon_gyr = LinearSegmentedColormap.from_list(
    "neon_gyr",
    [
        (0.0, "#39FF14"),  # neon/light green
        (0.5, "yellow"),
        (1.0, "red"),
    ]
)

# =========================
# PATHS
# =========================
PROJECT_ROOT = r"D:\BCI"
META_PATH = os.path.join(PROJECT_ROOT, "MILimbEEG", "data2", "metadata.xlsx")
PLOT_DIR = os.path.join(PROJECT_ROOT, "MILimbEEG", "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# =========================
# ELECTRODE MAP
# =========================
electrode_map = {
    0: "FC5",
    1: "F3",
    2: "Fz",
    3: "F4",
    4: "FC6",
    5: "FC1",
    6: "FC2",
    7: "Cz",
    8: "T7",
    9: "CP5",
    10: "C3",
    11: "CP1",
    12: "CP2",
    13: "C4",
    14: "CP6",
    15: "T8",
}

bands = ["a", "b", "g"]
band_titles = {"a": "Alpha", "b": "Beta", "g": "Gamma"}

# =========================
# LOAD DATA
# =========================
meta = pd.read_excel(META_PATH)

# Derive filtered path correctly
first_row = meta.iloc[0]
patient = f"S{first_row['patient_number']}"
filename = os.path.basename(first_row['local_url'])
csv_path = os.path.join(PROJECT_ROOT, "MILimbEEG", "fir_dataset", patient, f"f_{filename}")

df = pd.read_csv(csv_path)

# =========================
# MNE INFO (16 channels)
# =========================
ch_names = list(electrode_map.values())

info = mne.create_info(
    ch_names=ch_names,
    sfreq=125,
    ch_types="eeg"
)

montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage, on_missing="ignore")


# =========================
# SHIFT ELECTRODES DOWN
# =========================
SHIFT_Y = -0.02  # 👈 adjust if needed (-0.015 to -0.03 typical)

montage = info.get_montage()
pos = montage.get_positions()["ch_pos"]

new_pos = {}
for ch, xyz in pos.items():
    x, y, z = xyz
    new_pos[ch] = np.array([x, y + SHIFT_Y, z])

montage = mne.channels.make_dig_montage(
    ch_pos=new_pos,
    coord_frame="head"
)

info.set_montage(montage)

# =========================
# NORMALIZATION FUNCTION
# =========================
def minmax_norm(x):
    x = np.asarray(x, dtype=float)
    xmin, xmax = x.min(), x.max()
    if np.isclose(xmax - xmin, 0):
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

# =========================
# PLOT
# =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, band in zip(axes, bands):

    raw_vals = np.array([
        df[f"{idx}_{band}"].mean()
        for idx in electrode_map.keys()
    ])

    # 🔥 THIS IS THE KEY FIX
    values = minmax_norm(raw_vals)

    im, _ = mne.viz.plot_topomap(
        values,
        info,
        axes=ax,
        show=False,
        cmap=neon_gyr,   # blue → green
        vlim=(0, 1),
        contours=0,
        sensors=True,
        sphere=(0., 0., 0., 0.095)
    )

    ax.set_title(f"{band_titles[band]} Band")

# =========================
# COLORBAR (moved down)
# =========================
cbar = fig.colorbar(
    im,
    ax=axes,
    orientation="horizontal",
    fraction=0.045,
    pad=0.18   # ⬅️ pushes legend down
)
cbar.set_label("Normalized Power (0 → 1)")

# Make room at bottom
plt.subplots_adjust(bottom=0.25)

to_save = os.path.join(PLOT_DIR, "topology_heatmap.png")

plt.suptitle("EEG Topographic Heatmaps (Min–Max Normalized)", fontsize=14)

plt.savefig(
    to_save,
    dpi=300,
    bbox_inches="tight"
)

