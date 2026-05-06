import numpy as np
import os
import matplotlib.pyplot as plt
import mne

# =========================
# CONFIGURATION
# =========================
DATASET_DIR = r"D:\BCI\MILimbEEG\ml_dataset"
X_RAW_PATH = os.path.join(DATASET_DIR, "X_raw.npy")
Y_PATH = os.path.join(DATASET_DIR, "y_csp.npy")
FS = 125

def visualize_psd():
    print("Loading Raw Data...")
    if not os.path.exists(X_RAW_PATH):
        print("Error: X_raw.npy not found. Run prepare_csp_data.py first.")
        return

    X = np.load(X_RAW_PATH) # (n_epochs, n_channels, n_times)
    y = np.load(Y_PATH)
    
    print(f"Data Loaded: {X.shape}")
    
    # Separate Classes
    # Label 1: CLH, Label 2: CRH
    X_clh = X[y == 1]
    X_crh = X[y == 2]
    
    print(f"CLH Trials: {X_clh.shape[0]}")
    print(f"CRH Trials: {X_crh.shape[0]}")
    
    # Compute PSD using Welch's method (Channel-wise average)
    print("Computing PSD...")
    
    # Helper to compute mean PSD across all trials and channels for a class
    def compute_mean_psd(data, fs):
        # data: (n_trials, n_channels, n_times)
        # Flatten trials and channels essentially, or average over them
        # Let's average over trials first
        # mne.time_frequency.psd_array_welch input: (n_epochs, n_channels, n_times)
        psds, freqs = mne.time_frequency.psd_array_welch(
            data, sfreq=fs, fmin=4, fmax=40, n_fft=256, verbose=False
        )
        # psds shape: (n_epochs, n_channels, n_freqs)
        # 1. Average over trials
        psds_mean_epoch = np.mean(psds, axis=0) # (n_channels, n_freqs)
        # 2. Average over channels (Global Field Power approximation)
        psds_mean_global = np.mean(psds_mean_epoch, axis=0) # (n_freqs,)
        return freqs, psds_mean_global

    freqs, psd_clh = compute_mean_psd(X_clh, FS)
    _, psd_crh = compute_mean_psd(X_crh, FS)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, 10 * np.log10(psd_clh), label='CLH (Left Hand)', color='blue', linewidth=2)
    plt.plot(freqs, 10 * np.log10(psd_crh), label='CRH (Right Hand)', color='red', linewidth=2, linestyle='--')
    
    # Highlight Mu and Beta bands
    plt.axvspan(8, 13, color='gray', alpha=0.2, label='Mu Band (8-13 Hz)')
    plt.axvspan(13, 30, color='yellow', alpha=0.1, label='Beta Band (13-30 Hz)')
    
    plt.title("Global Power Spectral Density (Left vs Right)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB)")
    plt.legend()
    plt.grid(True)
    
    output_img = "psd_comparison.png"
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")
    # plt.show() # Uncomment if running locally with display

if __name__ == "__main__":
    visualize_psd()
