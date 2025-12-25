#!/usr/bin/env python3
"""
Create evaluation visualizations for the proper modeling results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

def create_evaluation_plots():
    """Create comprehensive evaluation visualizations"""
    
    print("=== Creating Evaluation Visualizations ===")
    
    # Load the proper results
    with open('reports/final_results_proper.json', 'r') as f:
        results = json.load(f)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Model Performance Comparison
    ax1 = plt.subplot(2, 3, 1)
    models = list(results['results'].keys())
    accuracies = [results['results'][model]['test_accuracy'] for model in models]
    f1_scores = [results['results'][model]['test_f1'] for model in models]
    auc_scores = [results['results'][model]['test_auc'] for model in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    ax1.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
    ax1.bar(x, f1_scores, width, label='F1-Score', alpha=0.8)
    ax1.bar(x + width, auc_scores, width, label='ROC-AUC', alpha=0.8)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', '\n') for m in models], rotation=0, ha='center')
    ax1.legend()
    ax1.set_ylim(0.9, 1.0)
    ax1.grid(True, alpha=0.3)
    
    # 2. Cross-Validation vs Test Performance
    ax2 = plt.subplot(2, 3, 2)
    cv_means = [results['results'][model]['cv_f1_mean'] for model in models]
    cv_stds = [results['results'][model]['cv_f1_std'] for model in models]
    test_f1s = [results['results'][model]['test_f1'] for model in models]
    
    ax2.errorbar(range(len(models)), cv_means, yerr=cv_stds, fmt='o-', label='Cross-Validation', capsize=5)
    ax2.plot(range(len(models)), test_f1s, 's-', label='Test Set', color='red')
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Cross-Validation vs Test Performance')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([m.replace('_', '\n') for m in models], rotation=0, ha='center')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.85, 1.0)
    
    # 3. Feature Importance (for best model)
    ax3 = plt.subplot(2, 3, 3)
    
    # Recreate best model to get feature importance
    try:
        # Load data and recreate model quickly
        df_raw = pd.read_csv('data/raw/loan_data.csv')
        df_clean = df_raw.drop(columns=['name', 'points'])
        df_clean['loan_approved'] = df_clean['loan_approved'].astype(int)
        
        X = df_clean.drop('loan_approved', axis=1)
        y = df_clean['loan_approved']
        
        # Quick feature engineering
        def quick_engineer(X):
            city_freq = X['city'].value_counts().to_dict()
            X_eng = X.copy()
            X_eng['city_frequency'] = X_eng['city'].map(city_freq)
            X_eng['income_to_loan_ratio'] = X_eng['income'] / (X_eng['loan_amount'] + 1)
            X_eng['employment_stability'] = np.clip(X_eng['years_employed'] / 30, 0, 1)
            
            # City size categories
            freq_thresholds = X_eng['city_frequency'].quantile([0.25, 0.5, 0.75])
            def categorize_city_freq(freq):
                if freq >= freq_thresholds[0.75]:
                    return 'large'
                elif freq >= freq_thresholds[0.5]:
                    return 'medium'
                elif freq >= freq_thresholds[0.25]:
                    return 'small'
                else:
                    return 'rare'
            
            X_eng['city_size'] = X_eng['city_frequency'].apply(categorize_city_freq)
            return X_eng.drop(columns=['city'])
        
        # Split and engineer
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_eng = quick_engineer(X_train)
        X_test_eng = quick_engineer(X_test)
        
        # Prepare features
        numerical_features = ['income', 'credit_score', 'loan_amount', 'years_employed', 
                             'city_frequency', 'income_to_loan_ratio', 'employment_stability']
        categorical_features = ['city_size']
        
        scaler = StandardScaler()
        X_train_scaled = X_train_eng.copy()
        X_test_scaled = X_test_eng.copy()
        X_train_scaled[numerical_features] = scaler.fit_transform(X_train_eng[numerical_features])
        X_test_scaled[numerical_features] = scaler.transform(X_test_eng[numerical_features])
        
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_train_cat = encoder.fit_transform(X_train_eng[categorical_features])
        X_test_cat = encoder.transform(X_test_eng[categorical_features])
        
        X_train_final = np.hstack([X_train_scaled[numerical_features], X_train_cat])
        
        # Train best model
        best_model = RandomForestClassifier(random_state=42, n_estimators=50)
        best_model.fit(X_train_final, y_train)
        
        # Get feature importance
        feature_names = numerical_features + list(encoder.get_feature_names_out(categorical_features))
        importances = best_model.feature_importances_
        
        # Plot feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        ax3.barh(importance_df['feature'], importance_df['importance'])
        ax3.set_xlabel('Importance')
        ax3.set_title('Feature Importance (Random Forest)')
        ax3.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"Could not create feature importance plot: {e}")
        ax3.text(0.5, 0.5, 'Feature Importance\nData Not Available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Feature Importance (Random Forest)')
    
    # 4. Target Distribution
    ax4 = plt.subplot(2, 3, 4)
    target_dist = results['n_samples']
    labels = ['Training', 'Test']
    sizes = [target_dist['train'], target_dist['test']]
    colors = ['lightblue', 'lightcoral']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
    ax4.set_title('Dataset Split Distribution')
    
    # 5. Cross-Validation Stability
    ax5 = plt.subplot(2, 3, 5)
    cv_means = [results['results'][model]['cv_f1_mean'] for model in models]
    cv_stds = [results['results'][model]['cv_f1_std'] for model in models]
    
    ax5.bar(range(len(models)), cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
    ax5.set_xlabel('Models')
    ax5.set_ylabel('CV F1-Score')
    ax5.set_title('Cross-Validation Stability')
    ax5.set_xticks(range(len(models)))
    ax5.set_xticklabels([m.replace('_', '\n') for m in models], rotation=0, ha='center')
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create text summary
    best_model = results['best_model']
    best_results = results['results'][best_model]
    
    summary_text = f"""
    BEST MODEL: {best_model.replace('_', ' ')}
    
    Test Accuracy: {best_results['test_accuracy']:.1%}
    Test F1-Score: {best_results['test_f1']:.3f}
    Test ROC-AUC: {best_results['test_auc']:.3f}
    
    Cross-Validation:
    F1-Score: {best_results['cv_f1_mean']:.3f}
    Std Dev: {best_results['cv_f1_std']:.3f}
    
    Features: {results['n_features']}
    Training Samples: {results['n_samples']['train']}
    Test Samples: {results['n_samples']['test']}
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('reports/evaluation_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("SUCCESS: Evaluation dashboard created: reports/evaluation_dashboard.png")
    
    # Create individual confusion matrix for best model
    plt.figure(figsize=(8, 6))
    
    # Recreate predictions for confusion matrix
    try:
        y_test_pred = best_model.predict(X_test_final)
        cm = confusion_matrix(y_test, y_test_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Rejected', 'Approved'],
                   yticklabels=['Rejected', 'Approved'])
        plt.title(f'Confusion Matrix - {best_model.replace("_", " ")}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('reports/confusion_matrix_proper.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("SUCCESS: Confusion matrix created: reports/confusion_matrix_proper.png")
        
        # Print classification report
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_test_pred, target_names=['Rejected', 'Approved']))
        
    except Exception as e:
        print(f"Could not create confusion matrix: {e}")
    
    print("\n=== Visualization Complete ===")

if __name__ == "__main__":
    create_evaluation_plots()