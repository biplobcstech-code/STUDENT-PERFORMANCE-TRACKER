# =====================================================
# STUDENT PERFORMANCE TRACKING SYSTEM USING XGBOOST
# =====================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# =====================================================
# LOAD DATASET
# =====================================================

file_path = r"D:\New folder (2)\student_data.csv"

df = pd.read_csv(file_path)

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

# =====================================================
# HANDLE MISSING VALUES
# =====================================================

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# =====================================================
# ENCODE CATEGORICAL FEATURES
# =====================================================

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# =====================================================
# CREATE PERFORMANCE COLUMN
# PASS = G3 >= 10
# FAIL = G3 < 10
# =====================================================

df["Performance"] = np.where(df["G3"] >= 10, 1, 0)

# =====================================================
# FEATURES AND TARGET
# =====================================================

X = df.drop(["Performance", "G3"], axis=1)

y = df["Performance"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# =====================================================
# BUILD XGBOOST MODEL
# =====================================================

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Training Completed!")

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# EVALUATION
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"\nAccuracy : {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 50)
print("TOP 10 IMPORTANT FEATURES")
print("=" * 50)

print(importance_df.head(10))

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(model, "student_performance_model.pkl")

print("\nModel Saved Successfully!")
print("File: student_performance_model.pkl")

# =====================================================
# SAMPLE PREDICTION
# =====================================================

sample_student = X.iloc[[0]]

prediction = model.predict(sample_student)

print("\n" + "=" * 50)
print("SAMPLE STUDENT PREDICTION")
print("=" * 50)

if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")

# =====================================================
# PREDICTION PROBABILITY
# =====================================================

probability = model.predict_proba(sample_student)

print("\nProbability of Passing:",
      round(probability[0][1] * 100, 2), "%")
