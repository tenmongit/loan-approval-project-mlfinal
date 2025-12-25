# Loan Approval Prediction Using Machine Learning

**Course**: SIS-2203 - Machine Learning Final Project  
**Semester**: Fall 2025  
**Team**: SIS-2203  

##  Project Overview

This project develops machine learning models to predict loan approval decisions, achieving **98.5% accuracy** with proper validation techniques. We identified and resolved critical data leakage issues, implemented robust cross-validation, and created interpretable models suitable for business deployment.

**Key Achievements:**
-  Fixed data leakage issues (removed 'points' feature with 0.82 target correlation)
-  Implemented proper train/test split and 5-fold cross-validation  
-  Reduced feature dimensionality from 1,000+ to 7 meaningful features
-  Achieved realistic performance: 98.3% F1-score with Random Forest
-  Delivered interpretable models with clear feature importance rankings

##  Quick Start

### Option 1: Run the Complete Pipeline (Recommended)
```bash
# Run the complete modeling pipeline with proper validation
python run_proper_modeling.py

# Create evaluation visualizations  
python create_evaluation_plots.py
```

### Option 2: Jupyter Notebook (Interactive)
```bash
# Launch Jupyter and open the complete modeling notebook
jupyter notebook notebooks/01_complete_modeling.ipynb
```

### Option 3: Google Colab (No Local Setup)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tenmongit/loan-approval-project-mlfinal/blob/main/notebooks/01_complete_modeling.ipynb)

##  Final Results

**Best Model: Random Forest**
- **Test Accuracy**: 98.5%
- **Test F1-Score**: 0.983
- **Cross-validation F1**: 0.975 (±0.021)
- **ROC-AUC**: 0.999

**Key Features (in order of importance):**
1. **Credit Score** (68% importance) - Primary creditworthiness indicator
2. **Income** (12% importance) - Repayment capacity
3. **City Frequency** (7% importance) - Geographic economic factors  
4. **Employment Stability** (5% importance) - Job security indicator
5. **Income-to-Loan Ratio** (4% importance) - Debt service capability

##  Project Structure

```
loan-approval-project-mlfinal/
├──  notebooks/
│   └── 01_complete_modeling.ipynb    # Complete modeling pipeline with proper validation
├──  src/
│   └── preprocess.py                  # Data preprocessing utilities
├──  data/
│   ├── raw/
│   │   └── loan_data.csv             # Original dataset (2,000 records)
│   ├── processed/
│   │   └── cleaned_no_leakage.csv    # Cleaned dataset (data leakage removed)
│   └── README.md                     # Dataset documentation
├──  models/
│   ├── best_model_proper.pkl         # Final Random Forest model
│   ├── scaler_proper.pkl             # Feature scaling transformer
│   └── encoder_proper.pkl            # Categorical encoding transformer
├──  reports/
│   ├── final_results_proper.json     # Model performance metrics
│   ├── evaluation_dashboard.png      # Comprehensive evaluation visualizations
│   └── confusion_matrix_proper.png   # Confusion matrix for best model
├──  documentation/
│   ├── PROJECT_PROPOSAL.md           # 12-page project proposal
│   └── FINAL_REPORT.md               # Comprehensive technical report
├──  run_proper_modeling.py         # Complete modeling pipeline script
├──  create_evaluation_plots.py     # Evaluation visualization script
└──  requirements.txt               # Python dependencies
```

##  Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/tenmongit/loan-approval-project-mlfinal.git
cd loan-approval-project-mlfinal

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python run_proper_modeling.py
```

##  Requirements

```
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.3
matplotlib==3.8.2
seaborn==0.13.0
joblib==1.3.2
gradio==4.8.0
```

##  Methodology

### 1. Data Quality Assessment
- **Dataset**: 2,000 loan applications from Kaggle
- **Target**: Binary classification (Approved/Rejected)
- **Key Finding**: Identified data leakage in 'points' feature (0.82 correlation)

### 2. Data Preprocessing
- **Data Leakage Removal**: Eliminated 'points' feature and applicant names
- **Feature Engineering**: Smart city encoding, financial ratios, employment stability
- **Train/Test Split**: 80%/20% with stratified sampling

### 3. Model Development
- **Algorithms**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Validation**: 5-fold stratified cross-validation
- **Evaluation**: Accuracy, F1-score, ROC-AUC, confusion matrix analysis

### 4. Key Improvements Over Original Approach
-  **Fixed Data Leakage**: Removed problematic features
-  **Proper Validation**: Implemented train/test split before preprocessing
-  **Feature Reduction**: From 1,000+ to 7 meaningful features
-  **Realistic Performance**: 98.5% accuracy vs. impossible 100%
-  **Cross-Validation**: Consistent performance across folds

##  Evaluation Results

| Model | Test Accuracy | Test F1-Score | CV F1-Score | Stability |
|-------|---------------|---------------|-------------|-----------|
| Logistic Regression | 95.5% | 0.949 | 0.888 | Good |
| Decision Tree | 98.5% | 0.983 | 0.964 | Excellent |
| **Random Forest** | **98.5%** | **0.983** | **0.975** | **Excellent** |
| Gradient Boosting | 98.2% | 0.980 | 0.977 | Excellent |

##  Visualizations

- **Model Comparison Dashboard**: Performance metrics across all models
- **Feature Importance**: Key factors influencing loan decisions
- **Confusion Matrix**: Detailed error analysis
- **Cross-Validation Stability**: Consistency across validation folds

##  Team

**Group:** SIS-2203  

| Name | GitHub | Role | Contributions |
|------|--------|------|---------------|
| Madiyar Mustafin | [@tenmongit](https://github.com/tenmongit) | Team Lead & ML Engineer | Model development, validation, deployment |
| Alisher Toleubay | [@sweetssymphony](https://github.com/sweetssymphony) | Data Scientist | Data preprocessing, feature engineering, EDA |
| Damir Izenbayev | [@unlessyoung](https://github.com/unlessyoung) | Research Analyst | Literature review, documentation, presentation |

##  Documentation

- **[Project Proposal](documentation/PROJECT_PROPOSAL.md)** - Comprehensive 12-page proposal
- **[Final Report](documentation/FINAL_REPORT.md)** - Detailed technical report with methodology
- **[Dataset Documentation](data/README.md)** - Data source and characteristics

##  Key Insights

1. **Credit Score Dominance**: 68% of model decisions based on credit score
2. **Income Matters**: Higher income significantly increases approval probability  
3. **Geographic Factors**: City size/frequency plays a role in decisions
4. **Employment Stability**: Longer employment history improves chances
5. **Debt Ratio**: Lower loan-to-income ratios favor approval

##  Important Notes

- **Data Leakage Fixed**: Original 'points' feature (0.82 correlation) removed
- **Realistic Performance**: 98.5% accuracy is excellent and achievable
- **Proper Validation**: Cross-validation confirms model stability
- **Business Ready**: Models are interpretable and deployment-ready

##  Future Enhancements

- **Advanced Algorithms**: XGBoost, LightGBM implementation
- **Real-time API**: Deploy as web service for instant predictions
- **Risk Scoring**: Predict default probability, not just approval
- **Continuous Learning**: Online model updates with new data

##  License

This project is part of academic coursework. Dataset license: Public Domain / CC0.

---

** If you use this project, please cite the original dataset source and give credit to the team.**
