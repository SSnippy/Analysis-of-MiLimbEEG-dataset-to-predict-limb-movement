import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# =========================
# PATHS
# =========================
PROJECT_ROOT = "/home/snippy/Desktop/Projects/Milimb_eeg"
DATAPOINTS_DIR = os.path.join(PROJECT_ROOT, "datapoints")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots", "patient_1_connectivity")

os.makedirs(PLOTS_DIR, exist_ok=True)

METADATA_PATH = os.path.join(DATAPOINTS_DIR, "metadata.xlsx")

# =========================
# PARAMETERS
# =========================
PATIENT_ID = 1
TASK_TYPE_REQUIRED = 1
TARGET_SAMPLES = 500
ELECTRODES = list(range(16))

BANDS = {
    "Original": None,   # special case (local_url)
    "Alpha": "_a",
    "Beta": "_b",
    "Gamma": "_g"
}

# =========================
# LOAD METADATA
# =========================
metadata = pd.read_excel(METADATA_PATH)

# Filter metadata
metadata = metadata[
    (metadata["patient_number"] == PATIENT_ID) &
    (metadata["task_type"] == TASK_TYPE_REQUIRED)
]

# =========================
# HELPER FUNCTIONS
# =========================
def load_eeg_matrix(csv_path, suffix=None):
    """
    Returns electrode x time matrix (16 x 500)
    """
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, csv_path)

    df = pd.read_csv(csv_path)
    df = df.iloc[:, 1:]  # drop serial column

    data = []
    for e in ELECTRODES:
        col = f"{e}{suffix}" if suffix else str(e)
        data.append(df[col].values[:TARGET_SAMPLES])

    return np.array(data)


def spearman_fisher_z(data):
    """
    data: electrodes x time
    returns Fisher-z transformed correlation matrix
    """
    r, _ = spearmanr(data, axis=1)
    r = np.clip(r, -0.999, 0.999)
    return np.arctanh(r)

# =========================
# MAIN LOOP: PER MOVEMENT
# =========================
for task_label in metadata["task_label"].unique():

    task_rows = metadata[metadata["task_label"] == task_label]

    # store Fisher-z matrices per band
    z_matrices = {band: [] for band in BANDS}

    for _, row in task_rows.iterrows():

        # --- Original EEG ---
        orig_data = load_eeg_matrix(row["local_url"], suffix=None)
        z_matrices["Original"].append(spearman_fisher_z(orig_data))

        # --- Filtered bands ---
        filtered_path = row["filtered_url"]
        for band, suffix in BANDS.items():
            if suffix is None:
                continue
            band_data = load_eeg_matrix(filtered_path, suffix)
            z_matrices[band].append(spearman_fisher_z(band_data))

    # =========================
    # AVERAGE ACROSS REPETITIONS
    # =========================
    avg_corr = {}
    for band, mats in z_matrices.items():
        z_mean = np.mean(np.stack(mats), axis=0)
        avg_corr[band] = np.tanh(z_mean)  # back to [-1, 1]

    # =========================
    # PLOTTING
    # =========================
    fig, axes = plt.subplots(
        1, 4,
        figsize=(22, 6),
        constrained_layout=True
    )

    for ax, (band, corr) in zip(axes, avg_corr.items()):
        im = ax.imshow(
            corr,
            cmap="seismic",
            vmin=-1,
            vmax=1
        )
        ax.set_title(f"{band}")
        ax.set_xticks(range(16))
        ax.set_yticks(range(16))
        ax.set_xlabel("Electrode")
        ax.set_ylabel("Electrode")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Patient {PATIENT_ID} – Movement: {task_label}\n"
        f"Spearman Correlation (Averaged Across Repetitions)",
        fontsize=14
    )

    # =========================
    # SAVE
    # =========================
    output_path = os.path.join(
        PLOTS_DIR,
        f"patient_{PATIENT_ID}_movement_{task_label}_connectivity.png"
    )

    plt.savefig(output_path, dpi=400)
    plt.close()

    print(f"Saved: {output_path}")
