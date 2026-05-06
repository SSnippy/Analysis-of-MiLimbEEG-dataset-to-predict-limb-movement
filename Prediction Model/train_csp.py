import numpy as np
import os
import joblib
import mne
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# CONFIGURATION
# =========================
DATASET_DIR = r"D:\BCI\MILimbEEG\ml_dataset"
X_RAW_PATH = os.path.join(DATASET_DIR, "X_raw.npy")
Y_PATH = os.path.join(DATASET_DIR, "y_csp.npy")
MODEL_PATH = os.path.join(DATASET_DIR, "csp_svm_model.joblib")

# MNE Info (Required for CSP plotting, optional for math but good practice)
FS = 125
# We don't have channel names easily available, creating dummy names
# If you know them, replace here.
msg = "Training CSP..."

def train_csp():
    print(msg)
    
    # 1. Load Data
    if not os.path.exists(X_RAW_PATH) or not os.path.exists(Y_PATH):
        print("Raw data not found. Run prepare_csp_data.py first.")
        return

    X = np.load(X_RAW_PATH) # (n_epochs, n_channels, n_times)
    y = np.load(Y_PATH)
    
    print(f"Data Loaded: X={X.shape}, y={y.shape}")
    
    # --- CRITICAL: FILTER DATA (8-30 Hz) ---
    print("Applying 8-30 Hz Band-Pass Filter (Mu/Beta)...")
    X = mne.filter.filter_data(X, FS, 8, 30, verbose=False)
    
    # Check classes
    classes = np.unique(y)
    print(f"Classes: {classes}")
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Define CSP + SVM Pipeline
    print("Setting up CSP pipeline...")
    
    # CSP Parameters:
    # n_components: Number of spatial filters to keep (usually 4-8)
    # reg: Regularization (helps if covariance matrix is ill-conditioned)
    # log: Apply log transform (essential for variance features)
    csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)
    
    svc = SVC(kernel='rbf', class_weight='balanced', probability=True)
    
    pipeline = Pipeline([
        ('csp', csp),
        ('svc', svc)
    ])
    
    # 4. Grid Search Optimization
    # We optimize both CSP columns and SVM hyperparameters
    param_grid = {
        'csp__n_components': [4, 6, 8],
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.01, 0.1]
    }
    
    print("Running GridSearchCV (this may take time)...")
    clf = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    
    print(f"Best Params: {clf.best_params_}")
    print(f"Best CV Score: {clf.best_score_:.4f}")
    
    best_model = clf.best_estimator_
    
    # 5. Evaluate
    print("\n--- Test Set Evaluation ---")
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Map labels for report
    unique_labels = np.unique(y_test)
    if set(unique_labels) == {3, 5}:
        target_names = ["DLF", "DRF"]
    elif set(unique_labels) == {1, 2}:
        target_names = ["CLH", "CRH"]
    else:
        target_names = [str(l) for l in unique_labels]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 6. Save
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train_csp()
