import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# CONFIGURATION
# =========================
DATASET_DIR = r"D:\BCI\MILimbEEG\ml_dataset"
X_PATH = os.path.join(DATASET_DIR, "X.npy")
Y_PATH = os.path.join(DATASET_DIR, "y.npy")
MODEL_PATH = os.path.join(DATASET_DIR, "best_binary_svm_pca.joblib")

# Internal Label Mapping for training
LABEL_NAMES = {1: "Movement", 0: "Rest"}

def train_hypertuned_svm():
    print("Loading existing dataset...")
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise FileNotFoundError(f"Dataset files not found in {DATASET_DIR}")
        
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    
    print(f"Full Dataset Shape: X={X.shape}, y={y.shape}")

    # --- FILTER AND GROUP LABELS ---
    print(f"Grouping All Movements (1-6) and Rest (7)...")
    movement_labels = [1, 2, 3, 4, 5, 6]
    rest_label = 7
    
    mask_mov = np.isin(y, movement_labels)
    mask_rest = (y == rest_label)
    
    X_mov = X[mask_mov]
    y_mov = np.ones(len(X_mov)) 
    
    X_rest = X[mask_rest]
    y_rest = np.zeros(len(X_rest))
    
    # --- BALANCE CLASSES (Downsample) ---
    print(f"Original counts - Movement: {len(X_mov)}, Rest: {len(X_rest)}")
    min_samples = min(len(X_mov), len(X_rest))
    
    np.random.seed(42)
    idx_mov = np.random.choice(len(X_mov), size=min_samples, replace=False)
    idx_rest = np.random.choice(len(X_rest), size=min_samples, replace=False)
    
    X_bin = np.concatenate([X_mov[idx_mov], X_rest[idx_rest]])
    y_bin = np.concatenate([y_mov[idx_mov], y_rest[idx_rest]])
    
    print(f"Balanced Data Shape: X={X_bin.shape}, y={y_bin.shape}")

    # --- SPLIT DATA ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_bin, y_bin, test_size=0.2, stratify=y_bin, random_state=42
    )
    
    # --- PIPELINE: PCA + SVM ---
    pipeline = Pipeline([
        ('pca', PCA()),
        ('svm', SVC(class_weight='balanced', probability=True, random_state=42))
    ])
    
    # --- EXTENSIVE GRID SEARCH ---
    print("\nStarting extensive grid search (this may take several minutes)...")
    param_grid = {
        'pca__n_components': [0.90, 0.95, 0.99, None],
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.1, 0.01],
        'svm__kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy', n_jobs=-1, verbose=1)
    
    clf.fit(X_train, y_train)
    
    print(f"\nBest Params: {clf.best_params_}")
    print(f"Best Validation Accuracy: {clf.best_score_:.4f}")
    
    best_model = clf.best_estimator_
    
    # --- EVALUATION ---
    print("\n--- Final Test Set Evaluation ---")
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    target_names = ["Rest", "Movement"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # --- SAVE MODEL ---
    print(f"\nSaving best model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)
    print("Model saved successfully.")

if __name__ == "__main__":
    try:
        train_hypertuned_svm()
    except Exception as e:
        print(f"An error occurred: {e}")
