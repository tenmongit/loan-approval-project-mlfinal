# Commit Message Guidelines

## Standardized Commit Message Format

### Structure:
```
type: brief description in lowercase
```

### Types:
- `created model:` - For new model implementations
- `trained model:` - For model training results
- `evaluated model:` - For model evaluation and metrics
- `processed data:` - For data preprocessing and cleaning
- `fixed:` - For bug fixes
- `docs:` - For documentation updates
- `refactor:` - For code restructuring
- `test:` - For adding or updating tests

## Examples by Category

### Model Development
```bash
# Creating new models
created model: baseline logistic regression with categorical encoding
created model: random forest classifier with hyperparameter tuning
created model: xgboost with feature selection

# Training results
trained model: random forest achieving 0.85 accuracy on validation set
trained model: neural network with early stopping and 92% accuracy

# Evaluation
evaluated model: baseline with confusion matrix and roc curve
evaluated model: ensemble with cross-validation and feature importance
```

### Data Processing
```bash
processed data: cleaned missing values and encoded categorical variables
processed data: normalized numerical features and removed outliers
processed data: created stratified train-test split with 80-20 ratio
```

### Bug Fixes
```bash
fixed: file path resolution for cross-platform compatibility
fixed: data leakage in feature engineering pipeline
fixed: unicode encoding errors in console output
```

### Documentation
```bash
docs: updated model performance comparison in readme
docs: added installation instructions for required packages
docs: created api documentation for preprocessing functions
```

## Best Practices

1. **Keep it concise**: 50 characters or less for the subject line
2. **Use lowercase**: Consistent with conventional commits
3. **Be specific**: Mention the model type, metrics, or specific changes
4. **Use present tense**: "created" not "created" or "creating"
5. **Avoid generic messages**: No "updated files" or "fixed stuff"

## Your Recent Work - Proper Structure

For the work we just completed, here's how the commits should have been structured:

```bash
# Instead of one big commit:
created model: baseline logistic regression with preprocessing

# It could have been:
created model: baseline logistic regression with categorical encoding
created model: random forest classifier with hyperparameter tuning
evaluated model: baseline vs random forest with confusion matrix
docs: added kimi cli quick guide for project usage
test: added verification script for notebook functionality
fixed: implemented robust file path handling for cross-platform use
```

## Future Commits

For your next Model Master tasks, use these patterns:

```bash
# When creating new models:
created model: [algorithm] with [preprocessing/features]

# When training:
trained model: [algorithm] achieving [metric] performance

# When evaluating:
evaluated model: [algorithm] with [evaluation methods]

# When fixing issues:
fixed: [specific problem and solution]
```

This makes your git history much more professional and easier to track progress!