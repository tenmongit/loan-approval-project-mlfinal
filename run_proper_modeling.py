#!/usr/bin/env python3
"""
Proper modeling script with data leakage fixes and realistic evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import json
import os

def main():
    print("=== Proper Modeling with Data Leakage Fixes ===")
    
    # Load data
    project_root = '.'
    raw_data_path = 'data/raw/loan_data.csv'
    df_raw = pd.read_csv(raw_data_path)

    print('Dataset loaded:', df_raw.shape)
    print('Target distribution:', df_raw['loan_approved'].value_counts(normalize=True).round(3).to_dict())

    # Check correlations
    df_corr = df_raw.copy()
    df_corr['loan_approved_num'] = df_corr['loan_approved'].astype(int)
    numeric_cols = ['income', 'credit_score', 'loan_amount', 'years_employed', 'points']
    correlations = df_corr[numeric_cols + ['loan_approved_num']].corr()['loan_approved_num'].sort_values(ascending=False)
    print('Correlations with loan approval:', correlations.round(3).to_dict())

    # Create clean dataset (remove data leakage)
    df_clean = df_raw.copy()
    df_clean = df_clean.drop(columns=['name', 'points'])  # Remove leakage features
    df_clean['loan_approved'] = df_clean['loan_approved'].astype(int)
    print('Clean dataset shape:', df_clean.shape)

    # Train/test split
    X = df_clean.drop('loan_approved', axis=1)
    y = df_clean['loan_approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f'Data split - Train: {X_train.shape}, Test: {X_test.shape}')

    # Feature engineering
    def engineer_features(X_train, X_test):
        X_train_eng = X_train.copy()
        X_test_eng = X_test.copy()
        
        # Frequency encoding for city
        city_freq = X_train['city'].value_counts().to_dict()
        X_train_eng['city_frequency'] = X_train_eng['city'].map(city_freq)
        X_test_eng['city_frequency'] = X_test_eng['city'].map(city_freq)
        X_test_eng['city_frequency'] = X_test_eng['city_frequency'].fillna(0)
        
        # City size categories
        freq_thresholds = X_train_eng['city_frequency'].quantile([0.25, 0.5, 0.75])
        def categorize_city_freq(freq):
            if freq >= freq_thresholds[0.75]:
                return 'large'
            elif freq >= freq_thresholds[0.5]:
                return 'medium'
            elif freq >= freq_thresholds[0.25]:
                return 'small'
            else:
                return 'rare'
        
        X_train_eng['city_size'] = X_train_eng['city_frequency'].apply(categorize_city_freq)
        X_test_eng['city_size'] = X_test_eng['city_frequency'].apply(categorize_city_freq)
        
        # Financial ratios
        X_train_eng['income_to_loan_ratio'] = X_train_eng['income'] / (X_train_eng['loan_amount'] + 1)
        X_test_eng['income_to_loan_ratio'] = X_test_eng['income'] / (X_test_eng['loan_amount'] + 1)
        
        # Employment stability
        X_train_eng['employment_stability'] = np.clip(X_train_eng['years_employed'] / 30, 0, 1)
        X_test_eng['employment_stability'] = np.clip(X_test_eng['years_employed'] / 30, 0, 1)
        
        # Drop original city
        X_train_eng = X_train_eng.drop(columns=['city'])
        X_test_eng = X_test_eng.drop(columns=['city'])
        
        return X_train_eng, X_test_eng

    X_train_engineered, X_test_engineered = engineer_features(X_train, X_test)
    print('Feature engineering complete. Shape:', X_train_engineered.shape)

    # Prepare features for modeling
    numerical_features = ['income', 'credit_score', 'loan_amount', 'years_employed', 
                         'city_frequency', 'income_to_loan_ratio', 'employment_stability']
    categorical_features = ['city_size']

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train_engineered.copy()
    X_test_scaled = X_test_engineered.copy()
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train_engineered[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test_engineered[numerical_features])

    # Encode categorical
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(X_train_engineered[categorical_features])
    X_test_cat = encoder.transform(X_test_engineered[categorical_features])

    # Combine features
    X_train_final = np.hstack([X_train_scaled[numerical_features], X_train_cat])
    X_test_final = np.hstack([X_test_scaled[numerical_features], X_test_cat])
    print('Final feature matrices:', X_train_final.shape, X_test_final.shape)

    # Train models with cross-validation
    models = {
        'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision_Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random_Forest': RandomForestClassifier(random_state=42, n_estimators=50),
        'Gradient_Boosting': GradientBoostingClassifier(random_state=42, n_estimators=50)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    final_results = {}

    print('\n=== Cross-Validation Results ===')
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_final, y_train, cv=cv, scoring='f1')
        cv_results[name] = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
        print(f'{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

    # Train final models
    print('\n=== Final Model Training ===')
    for name, model in models.items():
        model.fit(X_train_final, y_train)
        
        y_test_pred = model.predict(X_test_final)
        y_test_proba = model.predict_proba(X_test_final)[:, 1]
        
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        final_results[name] = {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'cv_f1_mean': cv_results[name]['cv_mean'],
            'cv_f1_std': cv_results[name]['cv_std']
        }
        print(f'{name}: Accuracy={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}')

    # Find best model
    best_model_name = max(final_results.keys(), key=lambda x: final_results[x]['test_f1'])
    print(f'\\nBest Model: {best_model_name}')
    print(f'Test F1-Score: {final_results[best_model_name]["test_f1"]:.4f}')
    print(f'Cross-validation F1: {final_results[best_model_name]["cv_f1_mean"]:.4f}')

    # Save results
    results_dict = {
        'best_model': best_model_name,
        'results': final_results,
        'feature_names': numerical_features + list(encoder.get_feature_names_out(categorical_features)),
        'n_features': X_train_final.shape[1],
        'n_samples': {'train': len(X_train), 'test': len(X_test), 'total': len(df_clean)}
    }

    with open('reports/final_results_proper.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print('\nResults saved to reports/final_results_proper.json')
    print('Proper modeling complete with realistic performance!')
    
    # Print summary
    print('\n=== SUMMARY ===')
    print(f'Best Model: {best_model_name}')
    print(f'Test Accuracy: {final_results[best_model_name]["test_accuracy"]:.4f}')
    print(f'Test F1-Score: {final_results[best_model_name]["test_f1"]:.4f}')
    print(f'Cross-validation F1: {final_results[best_model_name]["cv_f1_mean"]:.4f} (+/- {final_results[best_model_name]["cv_f1_std"]:.4f})')
    print(f'Number of features: {len(results_dict["feature_names"])}')
    print(f'Training samples: {len(X_train)}')
    print(f'Test samples: {len(X_test)}')
    
    return final_results, best_model_name

if __name__ == "__main__":
    main()