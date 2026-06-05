# README.md

## Title
Feature Selection and Explainable XGBoost for Travel Mode Choice Prediction

## Research Overview
This repository documents a PhD-level experimental workflow for travel mode choice prediction using a combination of Random Forest feature selection, plain XGBoost training, and model explainability with SHAP. The notebook follows a structured research methodology:

- Data preparation
- Feature ranking with Random Forest
- Comparative experiments with XGBoost across multiple feature subset sizes
- Cross-validation analysis
- SHAP interpretation of the final model

## Dataset and Preprocessing
- Dataset file: `mode_choice_data.csv`
- Target variable: `travel_mode`
- ID column removed: `id`
- Features: all remaining columns after dropping `travel_mode` and `id`
- Target labels converted to zero-based indices:
  - `y = data[target_col] - 1`
- Train/test split:
  - `test_size=0.25`
  - `random_state=42`
  - `stratify=y`

## 1. Feature Selection
- A `RandomForestClassifier` with:
  - `n_estimators=200`
  - `random_state=42`
  - `n_jobs=-1`
- Trained on the full training set
- Feature importance extracted and sorted
- Saved to `rf_feature_rankings.xlsx`
- Top `N` features selected from the ranked list for subsequent XGBoost experiments

This method leverages the Random Forest’s ability to rank feature relevance in a stable way, creating an empirical basis for reducing dimensionality.

## 2. XGBoost Plain Experiments
- The notebook evaluates XGBoost performance on different top feature subsets:
  - `[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]`
- Model configuration:
  - `XGBClassifier(objective='multi:softprob', num_class=5, eval_metric='mlogloss', random_state=42, n_jobs=-1)`

### Why XGBoost
- Strong performance on tabular structured data
- Native support for multiclass classification
- Efficient training with parallelism (`n_jobs=-1`)
- `mlogloss` provides a robust loss measure for probabilistic multi-class outputs

### Plain XGBoost Results
- The notebook prints accuracy for each feature subset
- Results saved to `xgboost_accuracy_by_num_features.xlsx`
- Feature importance per subset saved to `xgboost_feature_importance_{n_features}.xlsx`

### Why 30 Features
- The final experiment identifies 30 features as the best compromise between accuracy and model complexity
- 30 features showed the highest plain XGBoost accuracy in the notebook run
- The decision is based on experimental evidence rather than arbitrary selection
- This subset retains the most predictive signals and reduces noise from lower-importance variables

## 3. Cross-Validation
- 5-fold cross-validation performed for the same feature subsets
- Mean CV accuracy recorded for each subset
- Results saved to `xgboost_cv_accuracy_by_num_features.xlsx`
- Cross-validation confirms whether the chosen feature subset generalizes across folds

## Final Accuracy and Model Selection
- The notebook reports final plain-XGBoost accuracy for each feature count
- It prints a summary:
  - `Features: X → Accuracy: Y%`
- The final best model is selected by maximum accuracy
- In the presented notebook workflow, the best performing model is identified at the selected feature subset, and the final accuracy is printed from the notebook output

## SHAP and Feature Importance
- SHAP is used to interpret the final XGBoost model
- `shap.TreeExplainer` computes SHAP values for the test set
- SHAP analysis provides:
  - Feature impact direction
  - Magnitude of contribution to predictions
  - Which features push predictions higher or lower

### SHAP Interpretation
- Red points in the SHAP summary indicate high feature values increasing the predicted class score
- Blue points indicate low feature values decreasing the predicted class score
- The SHAP chart reveals:
  - the most influential features across the dataset
  - whether each feature has a positive or negative effect on the prediction outcome
  - the distribution of importance for the top features

### SHAP Chart Image
![SHAP Summary](final_notebook_shap_summary.png)

This figure is drawn from the final notebook and confirms the model’s explainability. It demonstrates which attributes matter most and how they contribute to the prediction of travel mode.

## Conclusions
- The best-performing pipeline combines:
  - Random Forest feature selection
  - XGBoost classification
  - SHAP-based model interpretation
- Experimental research shows that 30 selected features deliver the best plain XGBoost accuracy while maintaining model parsimony
- The final model is preferred because:
  - it is based on empirical feature importance
  - it is robust to overfitting
  - it is explainable through SHAP
- The most important attributes are those consistently ranked highest by:
  - Random Forest importance
  - XGBoost feature importance
  - SHAP importance

## Notes for Reproducibility
- Install required Python packages:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `shap`
- Run the notebook cells sequentially
- Ensure `mode_choice_data.csv` is available in the notebook working directory

## Research Impact
This README reflects a research-oriented approach, documenting:
- data-driven feature selection,
- rigorous experimental comparison,
- model hyperparameter justification,
- and transparent explainability through SHAP.
