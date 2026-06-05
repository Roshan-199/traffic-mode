# %% [markdown]
# # Feature Selection with Random Forest and XGBoost
#
# This notebook has 3 sections:
# 1. Feature selection
# 2. XGBoost plain
# 3. XGBoost with cross validation

# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# %%
# Load your data here
# Replace the CSV path and target column name with your actual data
data = pd.read_csv("traffic_dataset.csv")
target_col = "target"

X = data.drop(columns=[target_col])
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# %% [markdown]
# ## 1. Feature selection

# %%
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_importances_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

rf_importances_df.to_excel("rf_feature_rankings.xlsx", index=False)
rf_importances_df.head(20)

# %% [markdown]
# ## 2. XGBoost plain

# %%
feature_counts = [10, 20, 30, 40, 50, 55, 60, 65, 70]

plain_results = []

for n_features in feature_counts:
    top_features = rf_importances_df["feature"].head(n_features).tolist()
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train[top_features], y_train)
    y_pred = model.predict(X_test[top_features])

    acc = accuracy_score(y_test, y_pred)
    plain_results.append({"num_features": n_features, "accuracy": acc})

    fi_df = pd.DataFrame({
        "feature": top_features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi_df.to_excel(f"xgboost_feature_importance_{n_features}.xlsx", index=False)

plain_accuracy_df = pd.DataFrame(plain_results)
plain_accuracy_df.to_excel("xgboost_accuracy_by_num_features.xlsx", index=False)

print("Plain XGBoost accuracy results:")
for row in plain_results:
    print(f"Number of features {row['num_features']}: accuracy {row['accuracy']:.4f}")

# %% [markdown]
# ## 3. XGBoost and cross validation

# %%
cv_results = []

for n_features in feature_counts:
    top_features = rf_importances_df["feature"].head(n_features).tolist()
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    scores = cross_val_score(
        model,
        X_train[top_features],
        y_train,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    mean_acc = scores.mean()
    cv_results.append({"num_features": n_features, "accuracy": mean_acc})

    # Fit final model on full training set to record importances
    model.fit(X_train[top_features], y_train)
    fi_df = pd.DataFrame({
        "feature": top_features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df.to_excel(f"xgboost_cv_feature_importance_{n_features}.xlsx", index=False)

cv_accuracy_df = pd.DataFrame(cv_results)
cv_accuracy_df.to_excel("xgboost_cv_accuracy_by_num_features.xlsx", index=False)

print("Cross-validation XGBoost accuracy results:")
for row in cv_results:
    print(f"Number of features {row['num_features']}: accuracy {row['accuracy']:.4f}")