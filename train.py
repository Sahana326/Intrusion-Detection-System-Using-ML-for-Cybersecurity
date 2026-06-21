import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from preprocess import load_and_preprocess_data, FEATURES

def train_models(dataset_path="data/raw/network_traffic.csv", models_dir="models"):
    """
    Trains the double-stage intrusion detection system models:
    Stage 1: Isolation Forest (Anomaly Detection)
    Stage 2: Random Forest Classifier (Multi-class attack classification)
    """
    os.makedirs(models_dir, exist_ok=True)
    
    print("--- STEP 1: Preprocessing Data ---")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        dataset_path=dataset_path, 
        scaler_path=os.path.join(models_dir, "scaler.pkl")
    )
    
    print("\n--- STEP 2: Training Stage 1 (Anomaly Detection - Isolation Forest) ---")
    # isolation forest predicts 1 for inliers (normal) and -1 for outliers (anomalous)
    # We train it on the training set. Ideally it learns the profile of benign traffic, but 
    # even when trained on mixed traffic it learns to separate low-density attack flows from high-density normal flows.
    anomaly_detector = IsolationForest(contamination=0.4, random_state=42, n_jobs=-1)
    anomaly_detector.fit(X_train)
    
    # Save Isolation Forest
    anomaly_path = os.path.join(models_dir, "anomaly_detector.pkl")
    joblib.dump(anomaly_detector, anomaly_path)
    print(f"Stage 1 Anomaly Detector saved to: {anomaly_path}")
    
    # Evaluate Isolation Forest
    # Let's map true labels: BENIGN -> 1, Attack -> -1
    y_test_anomaly_true = y_test.apply(lambda label: 1 if label == "BENIGN" else -1)
    y_test_anomaly_pred = anomaly_detector.predict(X_test)
    
    anomaly_acc = accuracy_score(y_test_anomaly_true, y_test_anomaly_pred)
    print(f"Anomaly Detection (Normal vs. Attack) Accuracy: {anomaly_acc * 100:.2f}%")
    
    print("\n--- STEP 3: Training Stage 2 (Attack Classifier - Random Forest) ---")
    # Train multi-class classifier on all labels to label specific attacks
    classifier = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    classifier.fit(X_train, y_train)
    
    # Save Classifier
    classifier_path = os.path.join(models_dir, "classifier.pkl")
    joblib.dump(classifier, classifier_path)
    print(f"Stage 2 Classifier saved to: {classifier_path}")
    
    # Evaluate Classifier
    y_pred = classifier.predict(X_test)
    print("\nStage 2 Classifier Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance analysis
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 5 Most Important Features for Intrusion Detection:")
    for f in range(min(5, len(FEATURES))):
        print(f"{f+1}. {FEATURES[indices[f]]} ({importances[indices[f]]*100:.2f}%)")
        
    # Save training metadata
    metadata = {
        "classes": list(classifier.classes_),
        "features": FEATURES,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "anomaly_accuracy": float(anomaly_acc)
        }
    }
    metadata_path = os.path.join(models_dir, "metadata.pkl")
    joblib.dump(metadata, metadata_path)
    print(f"\nModel metadata saved to: {metadata_path}")
    print("Training process completed successfully.")

if __name__ == "__main__":
    train_models()
