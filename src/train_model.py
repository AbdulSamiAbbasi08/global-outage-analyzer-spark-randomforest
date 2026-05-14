import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report
)
import joblib
import os

print("Starting Random Forest model training...")

# Load cleaned data
df = pd.read_csv("data/cleaned_outage_data.csv")
print(f"Records loaded: {len(df)}")

# Features and target
features = [
    "hour",
    "day_of_week",
    "month",
    "bgp_signal",
    "active_probing",
    "traffic_drop_pct",
    "latency_ms",
    "weather_index",
    "country_index",
    "base_outage_prob"
]

X = df[features]
y = df["outage"]

print(f"Features: {features}")
print(f"Target distribution:\n{y.value_counts()}")

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size:  {len(X_test)}")

# Train Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("Training complete!")

# Evaluate model
y_pred = rf_model.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
print("===== FEATURE IMPORTANCE =====")
importance_df = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance_df.to_string(index=False))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/rf_model.pkl")

# Save evaluation metrics
metrics = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
}
pd.DataFrame([metrics]).to_csv("models/metrics.csv", index=False)
importance_df.to_csv("models/feature_importance.csv", index=False)

print("\nModel saved to models/rf_model.pkl")
print("Metrics saved to models/metrics.csv")
print("Feature importance saved to models/feature_importance.csv")