import numpy as np
import os
import joblib
import mne
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# =========================
# CONFIGURATION
# =========================
DATASET_DIR = r"D:\BCI\MILimbEEG\ml_dataset"
X_RAW_PATH = os.path.join(DATASET_DIR, "X_raw.npy")
Y_PATH = os.path.join(DATASET_DIR, "y_csp.npy")
MODEL_PATH = os.path.join(DATASET_DIR, "fbcsp_model.joblib")

FS = 125

# Filter Bank: multiple sub-bands
FILTER_BANK = [
    (4, 8),    # Theta
    (8, 12),   # Mu/Alpha
    (12, 16),  # Low Beta
    (16, 20),  # Mid Beta
    (20, 24),  # High Beta
    (24, 30),  # Beta-Gamma border
    (30, 40),  # Low Gamma
]

N_CSP_COMPONENTS = 4  # Per sub-band

def train_fbcsp():
    print("=" * 60)
    print("  FBCSP (Filter Bank Common Spatial Patterns)")
    print("=" * 60)
    
    # 1. Load Data
    if not os.path.exists(X_RAW_PATH):
        print("Error: X_raw.npy not found. Run prepare_csp_data.py first.")
        return

    X = np.load(X_RAW_PATH)  # (n_epochs, n_channels, n_times)
    y = np.load(Y_PATH)
    
    print(f"Data Loaded: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)}, Counts: {np.bincount(y)[1:]}")
    
    # 2. Split Data FIRST (to avoid data leakage in feature selection)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # 3. Filter Bank CSP Feature Extraction
    print(f"\nApplying {len(FILTER_BANK)} sub-band filters + CSP...")
    
    all_csp_features_train = []
    all_csp_features_test = []
    csp_filters = []
    
    for i, (fmin, fmax) in enumerate(FILTER_BANK):
        print(f"  Band {i+1}/{len(FILTER_BANK)}: {fmin}-{fmax} Hz", end=" -> ")
        
        # Band-pass filter
        X_train_filt = mne.filter.filter_data(
            X_train.astype(np.float64), FS, fmin, fmax, verbose=False
        )
        X_test_filt = mne.filter.filter_data(
            X_test.astype(np.float64), FS, fmin, fmax, verbose=False
        )
        
        # Apply CSP
        csp = CSP(n_components=N_CSP_COMPONENTS, reg='ledoit_wolf', 
                  log=True, norm_trace=False)
        
        try:
            csp.fit(X_train_filt, y_train)
            
            features_train = csp.transform(X_train_filt)
            features_test = csp.transform(X_test_filt)
            
            all_csp_features_train.append(features_train)
            all_csp_features_test.append(features_test)
            csp_filters.append(csp)
            
            print(f"OK ({features_train.shape[1]} features)")
        except Exception as e:
            print(f"FAILED ({e})")
            continue
    
    if len(all_csp_features_train) == 0:
        print("No bands produced valid CSP features. Exiting.")
        return
    
    # Concatenate all sub-band features
    X_train_fbcsp = np.hstack(all_csp_features_train)
    X_test_fbcsp = np.hstack(all_csp_features_test)
    
    print(f"\nTotal FBCSP Features: {X_train_fbcsp.shape[1]}")
    
    # 4. Feature Selection (Mutual Information)
    print("Selecting best features (Mutual Information)...")
    n_best = min(10, X_train_fbcsp.shape[1])
    selector = SelectKBest(mutual_info_classif, k=n_best)
    X_train_selected = selector.fit_transform(X_train_fbcsp, y_train)
    X_test_selected = selector.transform(X_test_fbcsp)
    
    selected_indices = selector.get_support(indices=True)
    print(f"Selected {n_best} features from indices: {selected_indices}")
    
    # 5. Train Classifiers (Try both SVM and LDA)
    # Define Labels mapping
    unique_labels = np.unique(y_test)
    if set(unique_labels) == {3, 5}:
        target_names = ["DLF", "DRF"]
    elif set(unique_labels) == {1, 2}:
        target_names = ["CLH", "CRH"]
    else:
        target_names = [str(l) for l in unique_labels]

    print("\n--- Training SVM ---")
    svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
    svm.fit(X_train_selected, y_train)
    y_pred_svm = svm.predict(X_test_selected)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Accuracy: {acc_svm:.4f}")
    print(classification_report(y_test, y_pred_svm, target_names=target_names))
    print(confusion_matrix(y_test, y_pred_svm))
    
    print("\n--- Training LDA ---")
    lda = LDA()
    lda.fit(X_train_selected, y_train)
    y_pred_lda = lda.predict(X_test_selected)
    acc_lda = accuracy_score(y_test, y_pred_lda)
    print(f"LDA Accuracy: {acc_lda:.4f}")
    print(classification_report(y_test, y_pred_lda, target_names=target_names))
    print(confusion_matrix(y_test, y_pred_lda))
    
    # 6. Cross-Validation on best model
    best_acc = max(acc_svm, acc_lda)
    best_name = "SVM" if acc_svm >= acc_lda else "LDA"
    best_clf = svm if acc_svm >= acc_lda else lda
    
    print(f"\nBest Classifier: {best_name} ({best_acc:.4f})")
    
    # CV on full FBCSP features
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_clf, X_train_selected, y_train, cv=cv, scoring='accuracy')
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 7. Save
    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump({
        'classifier': best_clf,
        'csp_filters': csp_filters,
        'selector': selector,
        'filter_bank': FILTER_BANK,
    }, MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train_fbcsp()
