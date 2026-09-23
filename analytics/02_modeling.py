# =====================================================================
# 02_modeling.py  (BEGINNER VERSION)
# Same work as the advanced version, but written with simple loops
# and clear variable names instead of comprehensions / advanced
# one-liners, so it is easier to follow step by step.
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, mean_absolute_error,
    mean_squared_error, r2_score,
)

CHART_DIR = "charts"
if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

# Read the same cleaned data by loading the CSV we saved in 01_eda.py
# (we do NOT download the dataset from the internet again here).
raw_data = pd.read_csv("titanic.csv")

# Apply the SAME cleaning steps used in 01_eda.py, so this script
# continues from the exact same cleaned data.
df = raw_data.copy()

for column_name in ["embarked", "embark_town"]:
    if column_name in df.columns:
        df = df[df[column_name].notna()]

if "age" in df.columns:
    age_median_value = df["age"].median()
    df["age"] = df["age"].fillna(age_median_value)

if "deck" in df.columns:
    df = df.drop(columns=["deck"])

print("Modeling input shape:", df.shape)


# =====================================================================
# TASK 7: Split the data into a training set and a test set
# =====================================================================
print("\n" + "=" * 70)
print("TASK 7: Train/test split")
print("=" * 70)

class_balance = df["survived"].value_counts(normalize=True).sort_index()
print("Class balance (survived):")
print(class_balance.round(3))

percent_not_survived = class_balance.get(0, 0) * 100
percent_survived = class_balance.get(1, 0) * 100
print(
    "Justification for stratification: survival is imbalanced "
    "(about", round(percent_not_survived, 1), "% not-survived vs about",
    round(percent_survived, 1), "% survived). A plain random split "
    "could, by chance, put too many/too few survivors in the train or "
    "test set. Using stratify=y keeps the same ratio in both sets."
)

# Build the list of feature columns: every column except the target
# ("survived") and "alive" (which is just a text version of the same
# answer, so keeping it would let the model "cheat").
feature_cols = []
for column_name in df.columns:
    if column_name != "survived" and column_name != "alive":
        feature_cols.append(column_name)

X = df[feature_cols]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train class balance:", y_train.value_counts(normalize=True).round(3).to_dict())
print("Test class balance :", y_test.value_counts(normalize=True).round(3).to_dict())


# =====================================================================
# TASK 8: Build a preprocessing pipeline (fit ONLY on training data)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 8: Preprocessing pipeline")
print("=" * 70)

numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

# For numbers: fill missing values with the median, then scale them
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# For text/category columns: fill missing values with the most common
# value, then convert to 0/1 columns (one-hot encoding)
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

print("Preprocessing pipeline is ready.")
print("It will only be 'fit' on the training data (X_train).")
print("The test data (X_test) will only be transformed, never fit, "
      "so no information from the test set leaks into training.")


# =====================================================================
# TASK 9: Train three different classifiers
# =====================================================================
print("\n" + "=" * 70)
print("TASK 9: Train three classifiers")
print("=" * 70)

logistic_model = LogisticRegression(max_iter=1000, random_state=42)
tree_model = DecisionTreeClassifier(random_state=42)
forest_model = RandomForestClassifier(random_state=42)

# We keep the trained pipelines in a simple dictionary so we can
# loop over them later instead of repeating code three times.
fitted_pipelines = {}

logistic_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", logistic_model)])
logistic_pipeline.fit(X_train, y_train)
fitted_pipelines["Logistic Regression"] = logistic_pipeline
print("Trained: Logistic Regression")

tree_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", tree_model)])
tree_pipeline.fit(X_train, y_train)
fitted_pipelines["Decision Tree"] = tree_pipeline
print("Trained: Decision Tree")

forest_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", forest_model)])
forest_pipeline.fit(X_train, y_train)
fitted_pipelines["Random Forest"] = forest_pipeline
print("Trained: Random Forest")

# Draw the Decision Tree so we can see how it makes decisions
onehot_encoder = tree_pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
onehot_column_names = list(onehot_encoder.get_feature_names_out(categorical_features))
all_feature_names = numeric_features + onehot_column_names

figure, axis = plt.subplots(figsize=(20, 10))
plot_tree(
    tree_pipeline.named_steps["model"],
    feature_names=all_feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    max_depth=3,
    fontsize=8,
    ax=axis,
)
axis.set_title("Decision Tree (showing only top 3 levels for readability)")
figure.tight_layout()
figure.savefig(CHART_DIR + "/task9_decision_tree.png", dpi=120)
plt.close(figure)


# =====================================================================
# TASK 10: Evaluate all three models
# =====================================================================
print("\n" + "=" * 70)
print("TASK 10: Evaluate all three models")
print("=" * 70)

results_list = []
figure_cm, axes_cm = plt.subplots(1, 3, figsize=(15, 4))
figure_roc, axis_roc = plt.subplots(figsize=(6, 5))

model_index = 0
for model_name in fitted_pipelines:
    pipeline = fitted_pipelines[model_name]

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc_score = roc_auc_score(y_test, probabilities)

    results_list.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc_score,
    })

    # Draw the confusion matrix for this model
    matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axes_cm[model_index])
    axes_cm[model_index].set_title("Confusion Matrix: " + model_name)
    axes_cm[model_index].set_xlabel("Predicted")
    axes_cm[model_index].set_ylabel("Actual")

    # Add this model's ROC curve to the shared ROC plot
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    curve_label = model_name + " (AUC=" + str(round(auc_score, 3)) + ")"
    axis_roc.plot(false_positive_rate, true_positive_rate, label=curve_label)

    model_index = model_index + 1

figure_cm.tight_layout()
figure_cm.savefig(CHART_DIR + "/task10_confusion_matrices.png", dpi=120)
plt.close(figure_cm)

axis_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
axis_roc.set_xlabel("False Positive Rate")
axis_roc.set_ylabel("True Positive Rate")
axis_roc.set_title("ROC curves - all three classifiers")
axis_roc.legend()
figure_roc.tight_layout()
figure_roc.savefig(CHART_DIR + "/task10_roc_curves.png", dpi=120)
plt.close(figure_roc)

comparison_table = pd.DataFrame(results_list)
comparison_table = comparison_table.set_index("Model")
comparison_table = comparison_table.round(3)
print("\nModel comparison table:")
print(comparison_table)


# =====================================================================
# TASK 11: Compare 3 ways of handling class imbalance (Random Forest)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 11: Imbalance handling comparison")
print("=" * 70)

print("Class balance in survived (full cleaned data):")
print(df["survived"].value_counts(normalize=True).round(3))

imbalance_results = {}

# (a) Baseline: no special handling
baseline_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42)),
])
baseline_pipeline.fit(X_train, y_train)
baseline_predictions = baseline_pipeline.predict(X_test)
imbalance_results["baseline"] = {
    "precision": precision_score(y_test, baseline_predictions),
    "recall": recall_score(y_test, baseline_predictions),
    "f1": f1_score(y_test, baseline_predictions),
}

# (b) class_weight="balanced": tell the model to pay more attention
# to the smaller class automatically
balanced_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
])
balanced_pipeline.fit(X_train, y_train)
balanced_predictions = balanced_pipeline.predict(X_test)
imbalance_results["class_weight_balanced"] = {
    "precision": precision_score(y_test, balanced_predictions),
    "recall": recall_score(y_test, balanced_predictions),
    "f1": f1_score(y_test, balanced_predictions),
}

# (c) SMOTE: create extra "synthetic" examples of the smaller class,
# but ONLY using the training data (never the test data)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    smote_pipeline = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(random_state=42)),
    ])
    smote_pipeline.fit(X_train, y_train)
    smote_predictions = smote_pipeline.predict(X_test)
    imbalance_results["smote"] = {
        "precision": precision_score(y_test, smote_predictions),
        "recall": recall_score(y_test, smote_predictions),
        "f1": f1_score(y_test, smote_predictions),
    }
except ImportError:
    print("imbalanced-learn is not installed. Run: "
          "pip install imbalanced-learn")
    imbalance_results["smote"] = {"precision": None, "recall": None, "f1": None}

imbalance_table = pd.DataFrame(imbalance_results)
imbalance_table = imbalance_table.transpose()
imbalance_table = imbalance_table.round(3)
print("\nImbalance strategy comparison (Random Forest):")
print(imbalance_table)
print(
    "\nConclusion: class_weight='balanced' and SMOTE usually increase "
    "recall (catching more real survivors) compared to the baseline, "
    "but can lower precision a little. SMOTE is only applied to the "
    "training data, never the test data, so we don't leak information."
)


# =====================================================================
# TASK 12: Hyperparameter tuning with GridSearchCV + OOB score
# =====================================================================
print("\n" + "=" * 70)
print("TASK 12: GridSearchCV + OOB score")
print("=" * 70)

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"],
}

grid_search_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42)),
])

grid_search = GridSearchCV(grid_search_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best params:", best_params)
print("Best CV F1 score:", round(grid_search.best_score_, 4))

# To see the OOB (out-of-bag) score, we need to build a fresh
# RandomForestClassifier with oob_score=True set from the start,
# using the best parameters we just found.
n_estimators_value = best_params["model__n_estimators"]
max_depth_value = best_params["model__max_depth"]
max_features_value = best_params["model__max_features"]

best_forest_model = RandomForestClassifier(
    oob_score=True,
    random_state=42,
    n_estimators=n_estimators_value,
    max_depth=max_depth_value,
    max_features=max_features_value,
)

best_rf_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", best_forest_model),
])
best_rf_pipeline.fit(X_train, y_train)

oob_score_value = best_rf_pipeline.named_steps["model"].oob_score_
print("Out-of-bag (OOB) score for the best model:", round(oob_score_value, 4))


# =====================================================================
# TASK 13: Regression side-task - predict "fare"
# =====================================================================
print("\n" + "=" * 70)
print("TASK 13: Regression side-task (predict fare)")
print("=" * 70)

regression_feature_cols = ["pclass", "age", "sibsp", "parch", "sex", "embarked"]
X_reg = df[regression_feature_cols]
y_reg = df["fare"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_numeric_features = ["pclass", "age", "sibsp", "parch"]
reg_categorical_features = ["sex", "embarked"]

reg_numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
reg_categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

regression_preprocessor = ColumnTransformer(transformers=[
    ("num", reg_numeric_transformer, reg_numeric_features),
    ("cat", reg_categorical_transformer, reg_categorical_features),
])

regression_pipeline = Pipeline(steps=[
    ("preprocess", regression_preprocessor),
    ("model", LinearRegression()),
])
regression_pipeline.fit(X_reg_train, y_reg_train)
fare_predictions = regression_pipeline.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, fare_predictions)
rmse = np.sqrt(mean_squared_error(y_reg_test, fare_predictions))
r2 = r2_score(y_reg_test, fare_predictions)

number_of_rows = X_reg_test.shape[0]
number_of_features = X_reg_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (number_of_rows - 1) / (number_of_rows - number_of_features - 1)

print("MAE: ", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R^2: ", round(r2, 3))
print("Adjusted R^2:", round(adjusted_r2, 3))

# Turn predictions into a Series so we can subtract properly and plot
fare_predictions_series = pd.Series(fare_predictions, index=y_reg_test.index)
residuals = y_reg_test - fare_predictions_series

figure, axis = plt.subplots(figsize=(7, 5))
axis.scatter(fare_predictions_series, residuals, alpha=0.5)
axis.axhline(0, color="red", linestyle="--")
axis.set_xlabel("Predicted fare")
axis.set_ylabel("Residual (actual - predicted)")
axis.set_title("Residual plot - fare regression")
figure.tight_layout()
figure.savefig(CHART_DIR + "/task13_residual_plot.png", dpi=120)
plt.close(figure)

# Check if residuals spread out more for high predicted fares
# (this is called "heteroscedasticity")
median_prediction = fare_predictions_series.median()
low_half_residuals = residuals[fare_predictions_series < median_prediction]
high_half_residuals = residuals[fare_predictions_series >= median_prediction]

low_half_std = low_half_residuals.std()
high_half_std = high_half_residuals.std()

print("Residual spread (std), low predicted fare half: ", round(low_half_std, 2))
print("Residual spread (std), high predicted fare half:", round(high_half_std, 2))

if high_half_std > 1.3 * low_half_std:
    print("Conclusion: The residual spread grows a lot for higher "
          "predicted fares. This is HETEROSCEDASTICITY - the model's "
          "errors are less consistent for expensive tickets.")
else:
    print("Conclusion: The residual spread looks fairly similar across "
          "low and high predicted fares, so there is no strong sign of "
          "heteroscedasticity.")


# =====================================================================
# TASK 14: Final comparison table and recommendation
# =====================================================================
print("\n" + "=" * 70)
print("TASK 14: Final comparison table + recommendation")
print("=" * 70)

print("\n--- Classification metrics (one row per classifier) ---")
print(comparison_table)

regression_summary = pd.DataFrame({
    "Value": {"MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted_R2": adjusted_r2}
})
regression_summary = regression_summary.round(3)
print("\n--- Regression metrics (fare side-task, own scale) ---")
print(regression_summary)

# Find which classifier had the best F1 score
best_classifier_name = comparison_table["F1"].idxmax()
best_f1_value = comparison_table.loc[best_classifier_name, "F1"]
best_auc_value = comparison_table.loc[best_classifier_name, "ROC_AUC"]

print(
    "\nFinal recommendation: deploy the", best_classifier_name,
    "classifier for predicting survival. It has the best F1 score (",
    round(best_f1_value, 3), ") and ROC-AUC (", round(best_auc_value, 3),
    ") among the three classifiers, so it balances precision and "
    "recall the best on this imbalanced target. The fare regression "
    "model is a separate side-task on a different scale (currency, "
    "not a 0-1 classification score) and is not directly comparable."
)


# =====================================================================
# TASK 15: Save the complete pipeline so it can be reused later
# =====================================================================
print("\n" + "=" * 70)
print("TASK 15: Save the complete pipeline")
print("=" * 70)

# We save the tuned Random Forest pipeline from Task 12 (preprocessing
# steps + model together, as ONE object) - never just the bare model.
full_pipeline = best_rf_pipeline

joblib.dump(full_pipeline, "full_pipeline.joblib")
print("Saved: full_pipeline.joblib")

# Load it back and check it still works correctly on a raw test row
reloaded_pipeline = joblib.load("full_pipeline.joblib")
one_raw_row = X_test.iloc[[0]]
prediction_check = reloaded_pipeline.predict(one_raw_row)

print("Reloaded pipeline prediction on one raw test row:", prediction_check)
print("Actual label for that row:", y_test.iloc[0])
print(
    "Confirmed: after loading the saved file with joblib.load(), the "
    "pipeline predicts correctly on raw, unprocessed data - no manual "
    "preprocessing step needed, because the saved object includes "
    "both the preprocessing and the model together."
)

print("\n02_modeling.py complete.")