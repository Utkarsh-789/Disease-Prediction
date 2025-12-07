"""
train_model.py

Usage:
    python train_model.py /path/to/diabetes_prediction_dataset.csv

What it does:
 - Loads CSV
 - Auto-detects target column (looks for 'diabetes','outcome','target' etc; falls back to last column)
 - Preprocesses numeric and categorical features (median imputation + scaling, most-frequent + one-hot)
 - Trains LogisticRegression and RandomForest
 - Evaluates on stratified 80/20 split
 - Saves best model (by accuracy) to ./best_model.pkl
 - Prints metrics and confusion matrix
"""
import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(path):
    df = pd.read_csv(path)
    return df

def find_target_column(df):
    # common target names
    for name in df.columns:
        if name.lower() in ('outcome','target','label','diabetes','diabetic'):
            return name
    # fallback to last column
    return df.columns[-1]

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # universal OneHotEncoder (no sparse/sparse_output)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    return preprocessor, numeric_cols, categorical_cols

def main(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = load_data(csv_path)
    print("Loaded dataset. Shape:", df.shape)

    target_col = find_target_column(df)
    print("Using target column:", target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # Pipelines
    pipe_logreg = Pipeline(steps=[('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
    pipe_rf = Pipeline(steps=[('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])

    print("Training Logistic Regression...")
    pipe_logreg.fit(X_train, y_train)

    print("Training Random Forest...")
    pipe_rf.fit(X_train, y_train)

    # Evaluate
    y_pred_log = pipe_logreg.predict(X_test)
    y_pred_rf = pipe_rf.predict(X_test)

    acc_log = accuracy_score(y_test, y_pred_log)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Logistic Regression accuracy: {acc_log:.4f}")
    print(f"Random Forest accuracy:   {acc_rf:.4f}\n")

    print("=== Logistic Regression classification report ===")
    print(classification_report(y_test, y_pred_log))
    print("=== Random Forest classification report ===")
    print(classification_report(y_test, y_pred_rf))

    # select best model
    best_pipe = pipe_rf if acc_rf >= acc_log else pipe_logreg
    best_name = "RandomForest" if acc_rf >= acc_log else "LogisticRegression"
    print("Selected best model:", best_name)

    cm = confusion_matrix(y_test, best_pipe.predict(X_test))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    # Save model
    out_path = "best_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "model_pipeline": best_pipe,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "target_column": target_col
        }, f)
    print("Saved best model to:", out_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_model.py /path/to/dataset.csv")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
