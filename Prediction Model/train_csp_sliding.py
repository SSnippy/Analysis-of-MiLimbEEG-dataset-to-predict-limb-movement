import numpy as np
import os
import joblib
import mne
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import mode

# =========================
# CONFIGURATION
# =========================
DATASET_DIR = r"D:\BCI\MILimbEEG\ml_dataset"
X_RAW_PATH = os.path.join(DATASET_DIR, "X_raw.npy")
Y_PATH = os.path.join(DATASET_DIR, "y_csp.npy")
MODEL_PATH = os.path.join(DATASET_DIR, "csp_sliding_model.joblib")

FS = 125
WINDOW_SIZE_SEC = 1.0 # 1 second window
STEP_SIZE_SEC = 0.1   # 0.1 second step

def train_sliding_window():
    print("Running Sliding Window Classification...")
    
    # 1. Load Data
    if not os.path.exists(X_RAW_PATH):
        print("Error: X_raw.npy not found.")
        return

    X_full = np.load(X_RAW_PATH) # (n_epochs, n_channels, n_times)
    y_full = np.load(Y_PATH)
    
    print(f"Original Data: X={X_full.shape}, y={y_full.shape}")
    
    # Apply Band-Pass Filter (8-30 Hz) globally first
    print("Filtering (8-30 Hz)...")
    X_full = mne.filter.filter_data(X_full, FS, 8, 30, verbose=False)
    
    # 2. Windowing Logic
    # We need to slice X_full into smaller windows.
    # Output: X_windows (n_total_windows, n_channels, n_window_samples)
    # y_windows (n_total_windows,)
    
    n_epochs, n_channels, n_times = X_full.shape
    window_samples = int(WINDOW_SIZE_SEC * FS)
    step_samples = int(STEP_SIZE_SEC * FS)
    
    X_windows = []
    y_windows = []
    trial_ids = [] # To map windows back to original trial for voting
    
    print(f"Slicing into {WINDOW_SIZE_SEC}s windows with {STEP_SIZE_SEC}s step...")
    
    for i in range(n_epochs):
        # Sliding window over the time dimension
        start = 0
        while start + window_samples <= n_times:
            end = start + window_samples
            segment = X_full[i, :, start:end]
            
            X_windows.append(segment)
            y_windows.append(y_full[i])
            trial_ids.append(i)
            
            start += step_samples
            
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    trial_ids = np.array(trial_ids)
    
    print(f"Windowed Data: X={X_windows.shape}, y={y_windows.shape}")
    
    # 3. Split Data (Group Split to keep trials together)
    # We must split by TRIAL, not by window, to avoid data leakage.
    unique_trials = np.unique(trial_ids)
    train_trials, test_trials = train_test_split(unique_trials, test_size=0.2, random_state=42)
    
    # Create masks
    train_mask = np.isin(trial_ids, train_trials)
    test_mask = np.isin(trial_ids, test_trials)
    
    X_train = X_windows[train_mask]
    y_train = y_windows[train_mask]
    
    X_test = X_windows[test_mask]
    y_test = y_windows[test_mask]
    trial_ids_test = trial_ids[test_mask]
    
    print(f"Train Windows: {X_train.shape[0]}, Test Windows: {X_test.shape[0]}")

    # 4. Train CSP + SVM on Windows
    print("Training CSP on windows...")
    csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)
    svc = SVC(kernel='rbf', class_weight='balanced', probability=True)
    pipeline = Pipeline([('csp', csp), ('svc', svc)])
    
    pipeline.fit(X_train, y_train)
    
    # 5. Evaluate (Voting)
    print("\n--- Evaluating with Majority Voting per Trial ---")
    
    # Predict on all test windows
    y_pred_windows = pipeline.predict(X_test)
    
    # Aggregate votes
    final_preds = []
    final_truth = []
    
    for trial in test_trials:
        # Get windows for this trial
        mask = (trial_ids_test == trial)
        votes = y_pred_windows[mask]
        
        if len(votes) == 0:
            continue
            
        # Majority Vote
        prediction = mode(votes, keepdims=True).mode[0]
        
        # Ground Truth (all windows have same label)
        truth = y_test[mask][0]
        
        final_preds.append(prediction)
        final_truth.append(truth)
        
    acc = accuracy_score(final_truth, final_preds)
    print(f"Final Trial Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(final_truth, final_preds))
    
    print("Done.")

if __name__ == "__main__":
    train_sliding_window()
