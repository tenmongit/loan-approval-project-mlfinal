#!/usr/bin/env python3
"""
Test script to verify all notebooks work correctly regardless of execution location.
This simulates running the notebooks from any directory.
"""

import os
import sys
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

def test_baseline_model():
    """Test the baseline notebook functionality"""
    print("=== Testing Baseline Model (03_baseline.ipynb) ===")
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Load data
    data_path = os.path.join(project_root, 'data', 'processed', 'cleaned.csv')
    df = pd.read_csv(data_path)
    X, y = df.drop('loan_approved', axis=1), df['loan_approved']
    
    print(f"SUCCESS: Data loaded: {df.shape}")
    
    # Create preprocessing pipeline
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    
    # Create and train model
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    pipe.fit(X, y)
    pred = pipe.predict(X)
    
    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    
    print(f"SUCCESS: Baseline Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Save model
    model_path = os.path.join(project_root, 'models', 'baseline.pkl')
    joblib.dump(pipe, model_path)
    print(f"SUCCESS: Model saved to {model_path}")
    
    return accuracy, f1

def test_random_forest_model():
    """Test the Random Forest notebook functionality"""
    print("\n=== Testing Random Forest Model (04_advanced.ipynb) ===")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Load data
    data_path = os.path.join(project_root, 'data', 'processed', 'cleaned.csv')
    df = pd.read_csv(data_path)
    X, y = df.drop('loan_approved', axis=1), df['loan_approved']
    
    # Create preprocessing pipeline
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    
    # Create and train Random Forest model
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    rf_pipeline.fit(X, y)
    pred = rf_pipeline.predict(X)
    
    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    
    print(f"SUCCESS: Random Forest Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Save model
    model_path = os.path.join(project_root, 'models', 'randomforest.pkl')
    joblib.dump(rf_pipeline, model_path)
    print(f"SUCCESS: Model saved to {model_path}")
    
    return accuracy, f1

def test_evaluation():
    """Test the evaluation notebook functionality"""
    print("\n=== Testing Evaluation (05_evaluation.ipynb) ===")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Load data and model
    data_path = os.path.join(project_root, 'data', 'processed', 'cleaned.csv')
    model_path = os.path.join(project_root, 'models', 'randomforest.pkl')
    
    df = pd.read_csv(data_path)
    X, y = df.drop('loan_approved', axis=1), df['loan_approved']
    rf = joblib.load(model_path)
    
    # Generate predictions
    y_pred = rf.predict(X)
    
    # Create confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"SUCCESS: Confusion matrix calculated: {cm.shape}")
    
    # Create and save visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Approved', 'Approved'],
                yticklabels=['Not Approved', 'Approved'])
    plt.title('Confusion Matrix - Random Forest Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    report_path = os.path.join(project_root, 'reports', 'confusion_matrix.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SUCCESS: Confusion matrix saved to {report_path}")
    
    # Classification report
    report = classification_report(y, y_pred, target_names=['Not Approved', 'Approved'])
    print("SUCCESS: Classification report generated")
    
    return cm, report

def main():
    """Run all tests"""
    print("Testing All Notebooks - File Path Independent")
    print("=" * 50)
    
    try:
        # Test all components
        baseline_acc, baseline_f1 = test_baseline_model()
        rf_acc, rf_f1 = test_random_forest_model()
        cm, report = test_evaluation()
        
        print("\n" + "=" * 50)
        print("SUCCESS: ALL TESTS PASSED!")
        print(f"Baseline Model: Accuracy={baseline_acc:.4f}, F1={baseline_f1:.4f}")
        print(f"Random Forest: Accuracy={rf_acc:.4f}, F1={rf_f1:.4f}")
        print(f"Confusion Matrix Shape: {cm.shape}")
        print("\nThe notebooks should work correctly from any location!")
        
    except Exception as e:
        print(f"\nFAILED: TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()