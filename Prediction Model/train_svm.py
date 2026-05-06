import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# CONFIGURATION
# =========================
DATASET_DIR = r"D:\BCI\MILimbEEG\ml_dataset"
X_PATH = os.path.join(DATASET_DIR, "X.npy")
Y_PATH = os.path.join(DATASET_DIR, "y.npy")
SCALER_PATH = os.path.join(DATASET_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(DATASET_DIR, "svm_model.joblib")

# Label Mapping
LABEL_MAP = {
    1: "CLH",
    2: "CRH"
}

# CONFIGURATION FOR FILTERING
TARGET_LABELS = [1, 2]

# =========================
# PREDICTION FUNCTION
# =========================
def predict_movement(feature_vector, model, scaler):
    """
    Predicts movement label from a single feature vector.
    
    Args:
        feature_vector (np.array): 1D array of features (raw, unscaled).
        model: Trained SVM model.
        scaler: Fitted StandardScaler.
        
    Returns:
        str: Decoded movement label (e.g., "DLF").
    """
    # Reshape if 1D
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)
        
    # Apply scaling
    scaled_features = scaler.transform(feature_vector)
    
    # Predict
    prediction_idx = model.predict(scaled_features)[0]
    return LABEL_MAP.get(prediction_idx, "Unknown")

# =========================
# MAIN TRAINING FUNCTIONS
# =========================
def load_data():
    print("Loading dataset...")
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise FileNotFoundError(f"Dataset files not found in {DATASET_DIR}")
        
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    print(f"Data Loaded: X={X.shape}, y={y.shape}")

    # --- FILTER FOR TARGET LABELS ---
    print(f"Filtering data for labels {TARGET_LABELS}...")
    mask = np.isin(y, TARGET_LABELS)
    X = X[mask]
    y = y[mask]
    print(f"Filtered Data: X={X.shape}, y={y.shape}")

    return X, y, scaler

def train_and_evaluate():
    # 1. Load Data
    X, y, scaler = load_data()
    
    # 2. Split Data (Stratified 80/20)
    print("\nSplitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Initialize Model (Grid Search)
    print("Initializing SVM with GridSearchCV...")
    
    # Define Parameter Grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
    
    svc = SVC(class_weight='balanced', probability=True, random_state=42)
    clf = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    
    # 4. Train
    print("Tuning hyperparameters (this may take a minute)...")
    clf.fit(X_train, y_train)
    
    print(f"Training complete. Best Params: {clf.best_params_}")
    print(f"Best CV Score: {clf.best_score_:.4f}")
    
    # Use best estimator for evaluations
    best_model = clf.best_estimator_
    
    # 5. Evaluate on Test Set
    print("\n--- Test Set Evaluation ---")
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=list(LABEL_MAP.values()), zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 6. Validation on Full Dataset
    print("\n--- Validation on Full Dataset ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # 7. Save Model
    print(f"\nSaving best model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)
    print("Model saved successfully.")
    
    return best_model, scaler

if __name__ == "__main__":
    try:
        model, scaler = train_and_evaluate()
        
        # Demo Prediction with a dummy vector
        # Note: Feature size is now larger (192 base + bands). 
        # But dummy vector size needs to match. 
        # We don't know exact size until runtime, but assuming standard channels (64?), 
        # previously it was 192 (64 * 3). Now it's 64 * 5 = 320.
        # Let's dynamically check model n_features_in_ if possible, or just skip demo for now to avoid crash.
        
        print("\n--- Model Ready ---")
        
    except Exception as e:
        print(f"An error occurred: {e}")

