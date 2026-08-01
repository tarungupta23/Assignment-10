"""
Task 1: Data Understanding & Preprocessing
Task 2: Model Development (Logistic Regression)
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Task 1: Data Understanding and Preprocessing
# ---------------------------------------------------------------------------
print("=" * 60)
print("TASK 1: Data Understanding and Preprocessing")
print("=" * 60)

df = pd.read_csv("heart.csv")

print("\nFirst five records:")
print(df.head())

target_col = "target"
numerical_features = [c for c in df.columns if c != target_col]
print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
print(f"Target variable: '{target_col}'")

print("\nMissing values per column:")
print(df.isnull().sum())


n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nRemoved {n_before - len(df)} duplicate rows ({n_before} -> {len(df)} unique patient records)")

X = df[numerical_features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

# ---------------------------------------------------------------------------
# Task 2: Model Development (Logistic Regression)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 2: Model Development")
print("=" * 60)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Logistic Regression -> Accuracy: {acc:.4f}")

# Save the trained model with Pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Also persist feature order, for the API layer
with open("model_meta.pkl", "wb") as f:
    pickle.dump({"algorithm": "Logistic Regression", "features": numerical_features}, f)

print("\nSaved trained model to 'model.pkl'")
print("Saved model metadata to 'model_meta.pkl'")
