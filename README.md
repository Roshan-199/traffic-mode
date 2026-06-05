# XGBoost with Random Forest Feature Selection and SHAP Analysis

## Overview

This repository contains a research notebook implementing a machine learning pipeline for mode choice prediction using a combination of Random Forest feature selection and an XGBoost classifier. The workflow is designed from the perspective of a data science researcher, emphasizing feature relevance, model explainability, and robust evaluation.

## Objective

- Predict `travel_mode` using a tabular dataset.
- Use Random Forest feature importance to select the most predictive features.
- Train an XGBoost classifier on the selected features.
- Evaluate performance using test-set accuracy.
- Interpret the final model using SHAP values and feature importance.

## Dataset

- Input file: `mode_choice_data.csv`
- Target column: `travel_mode`
- Data preprocessing:
  - Trim whitespace and remove tab characters in column names.
  - Convert target labels to 0-based indexing.
  - Remove identifier column `id` from the features.

## Feature Selection

- A `RandomForestClassifier` is trained on the full training set.
- Feature importance from the Random Forest model is extracted.
- The top 30 features by importance are selected for the final model.
- This approach reduces dimensionality and improves interpretability while preserving predictive power.

## Model and Hyperparameters

### Model used
- `XGBClassifier` from the XGBoost library

### Hyperparameters
- `objective='multi:softprob'`
- `num_class=5`
- `eval_metric='mlogloss'`
- `random_state=42`
- `n_jobs=-1`

### Why XGBoost
- XGBoost is chosen for its proven efficiency on structured data.
- It handles multi-class classification naturally.
- It supports fast training with parallel processing through `n_jobs=-1`.
- `mlogloss` is appropriate for multi-class probability output and robust model fitting.

## Training and Evaluation

- Data is split using `train_test_split` with `test_size=0.25` and `stratify=y`.
- The final classifier is trained on the top 30 features selected earlier.
- Model performance is measured using `accuracy_score` on the held-out test set.
- The notebook prints the final accuracy as:
  - `XGBoost Accuracy with Top 30 Features: {acc*100:.2f}%`

## SHAP and Feature Importance

### SHAP analysis
- `shap.TreeExplainer` is used to compute SHAP values for the XGBoost model.
- SHAP values explain the contribution of each feature to individual predictions.
- A `shap.summary_plot` visualizes feature impact across the test set.

### Feature importance
- The notebook also plots XGBoost’s internal feature importance.
- Comparing Random Forest and XGBoost importance helps confirm which attributes are consistently predictive.

## Conclusions

- The combined approach of Random Forest feature selection followed by XGBoost classification is effective for this travel mode prediction task.
- Feature selection improves model focus on the most relevant attributes and reduces noise.
- SHAP analysis provides transparent model explainability, revealing which attributes drive predictions.
- The model is strong because it:
  - uses an ensemble-based feature selection method,
  - leverages XGBoost for robust multi-class learning,
  - and includes explainability through SHAP.
- The most important attributes are those consistently ranked highest by both Random Forest and XGBoost importance measures.

## Notes for Reproduction

1. Install dependencies:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `xgboost`
   - `shap`
   - `matplotlib`
2. Place `mode_choice_data.csv` in the same folder.
3. Run the notebook cells sequentially.

## Research Perspective

As a researcher in data science and machine learning, this pipeline demonstrates a principled methodology:
- start with careful data preparation,
- apply interpretable feature selection,
- choose a strong predictive model,
- and validate results with both quantitative metrics and explainability tools.
# XGBoost with Random Forest Feature Selection and SHAP Analysis

## Overview

This repository contains a research notebook implementing a machine learning pipeline for mode choice prediction using a combination of Random Forest feature selection and an XGBoost classifier. The workflow is designed from the perspective of a data science researcher, emphasizing feature relevance, model explainability, and robust evaluation.

## Objective

- Predict `travel_mode` using a tabular dataset.
- Use Random Forest feature importance to select the most predictive features.
- Train an XGBoost classifier on the selected features.
- Evaluate performance using test-set accuracy.
- Interpret the final model using SHAP values and feature importance.

## Dataset

- Input file: `mode_choice_data.csv`
- Target column: `travel_mode`
- Data preprocessing:
  - Trim whitespace and remove tab characters in column names.
  - Convert target labels to 0-based indexing.
  - Remove identifier column `id` from the features.

## Feature Selection

- A `RandomForestClassifier` is trained on the full training set.
- Feature importance from the Random Forest model is extracted.
- The top 30 features by importance are selected for the final model.
- This approach reduces dimensionality and improves interpretability while preserving predictive power.

## Model and Hyperparameters

### Model used
- `XGBClassifier` from the XGBoost library

### Hyperparameters
- `objective='multi:softprob'`
- `num_class=5`
- `eval_metric='mlogloss'`
- `random_state=42`
- `n_jobs=-1`

### Why XGBoost
- XGBoost is chosen for its proven efficiency on structured data.
- It handles multi-class classification naturally.
- It supports fast training with parallel processing through `n_jobs=-1`.
- `mlogloss` is appropriate for multi-class probability output and robust model fitting.

## Training and Evaluation

- Data is split using `train_test_split` with `test_size=0.25` and `stratify=y`.
- The final classifier is trained on the top 30 features selected earlier.
- Model performance is measured using `accuracy_score` on the held-out test set.
- The notebook prints the final accuracy as:
  - `XGBoost Accuracy with Top 30 Features: {acc*100:.2f}%`

## SHAP and Feature Importance

### SHAP analysis
- `shap.TreeExplainer` is used to compute SHAP values for the XGBoost model.
- SHAP values explain the contribution of each feature to individual predictions.
- A `shap.summary_plot` visualizes feature impact across the test set.

### Feature importance
- The notebook also plots XGBoost’s internal feature importance.
- Comparing Random Forest and XGBoost importance helps confirm which attributes are consistently predictive.

## Conclusions

- The combined approach of Random Forest feature selection followed by XGBoost classification is effective for this travel mode prediction task.
- Feature selection improves model focus on the most relevant attributes and reduces noise.
- SHAP analysis provides transparent model explainability, revealing which attributes drive predictions.
- The model is strong because it:
  - uses an ensemble-based feature selection method,
  - leverages XGBoost for robust multi-class learning,
  - and includes explainability through SHAP.
- The most important attributes are those consistently ranked highest by both Random Forest and XGBoost importance measures.

## Notes for Reproduction

1. Install dependencies:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `xgboost`
   - `shap`
   - `matplotlib`
2. Place `mode_choice_data.csv` in the same folder.
3. Run the notebook cells sequentially.

## Research Perspective

As a researcher in data science and machine learning, this pipeline demonstrates a principled methodology:
- start with careful data preparation,
- apply interpretable feature selection,
- choose a strong predictive model,
- and validate results with both quantitative metrics and explainability tools.
