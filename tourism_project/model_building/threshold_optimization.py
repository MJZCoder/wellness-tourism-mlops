
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"

# --------------------------------------------------
# 2. Load the previously trained model
# --------------------------------------------------

model = joblib.load(DEPLOYMENT_DIR / "best_model.pkl")

with open(DEPLOYMENT_DIR / "features.json") as f:
    feature_columns = json.load(f)

print("Trained XGBoost model loaded successfully!")

# --------------------------------------------------
# 3. Load training data only
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")

X_train = train_df[feature_columns]
y_train = train_df["ProdTaken"]

print("Training features:", X_train.shape)
print("Training target:", y_train.shape)

# --------------------------------------------------
# 4. Define cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nThreshold optimization setup completed!")


# --------------------------------------------------
# 5. Generate out-of-fold predicted probabilities
# --------------------------------------------------

print("\nGenerating out-of-fold predictions...")

oof_prob = cross_val_predict(
    clone(model),
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

print("Out-of-fold predictions generated successfully!")
print("Number of predictions:", len(oof_prob))
print("Minimum predicted probability:", round(oof_prob.min(), 4))
print("Maximum predicted probability:", round(oof_prob.max(), 4))


# --------------------------------------------------
# 6. Compare classification thresholds
# --------------------------------------------------

thresholds = np.round(
    np.arange(0.20, 0.501, 0.01), 2
)

results = []

for threshold in thresholds:

    y_pred = (oof_prob >= threshold).astype(int)

    results.append({
        "Threshold": float(threshold),
        "Precision": precision_score(
            y_train, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_train, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_train, y_pred, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

best_row = results_df.loc[
    results_df["F1"].idxmax()
]

best_threshold = float(best_row["Threshold"])

print("\nThreshold comparison:")
print(results_df.round(4).to_string(index=False))

print("\nBest threshold:", best_threshold)
print("Best out-of-fold F1:", round(float(best_row["F1"]), 4))
print("Recall at best threshold:", round(float(best_row["Recall"]), 4))
print("Precision at best threshold:", round(float(best_row["Precision"]), 4))


# --------------------------------------------------
# 7. Evaluate optimized threshold on test data
# --------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)

test_df = pd.read_csv(DATA_DIR / "test.csv")

X_test = test_df[feature_columns]
y_test = test_df["ProdTaken"]

# Obtain predicted probabilities from the trained model
test_prob = model.predict_proba(X_test)[:, 1]

# Apply original classification threshold
original_pred = (test_prob >= 0.50).astype(int)

# Apply optimized classification threshold
optimized_pred = (
    test_prob >= best_threshold
).astype(int)

print("\nOriginal model (threshold = 0.50):")

print(classification_report(
    y_test,
    original_pred,
    zero_division=0
))

print("\nOptimized model (threshold =", best_threshold, "):")

print(classification_report(
    y_test,
    optimized_pred,
    zero_division=0
))

print("\nOptimized test metrics:")

print("Accuracy:", round(
    accuracy_score(y_test, optimized_pred), 4
))

print("Precision:", round(
    precision_score(y_test, optimized_pred), 4
))

print("Recall:", round(
    recall_score(y_test, optimized_pred), 4
))

print("F1-score:", round(
    f1_score(y_test, optimized_pred), 4
))

print("ROC-AUC:", round(
    roc_auc_score(y_test, test_prob), 4
))


# --------------------------------------------------
# 8. Log threshold optimization experiment to MLflow
# --------------------------------------------------

import mlflow

MLFLOW_DIR = PROJECT_DIR / "mlruns"
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())
mlflow.set_experiment("Wellness_Tourism_Prediction")

with mlflow.start_run(run_name="XGBoost_Threshold_Optimization"):

    # Record experiment configuration
    mlflow.log_param("model_type", "XGBClassifier")
    mlflow.log_param("original_threshold", 0.50)
    mlflow.log_param(
        "optimized_threshold",
        best_threshold
    )
    mlflow.log_param("optimization_metric", "f1")
    mlflow.log_param("cv_folds", 5)

    # Record original test performance
    mlflow.log_metric(
        "baseline_accuracy",
        accuracy_score(y_test, original_pred)
    )
    mlflow.log_metric(
        "baseline_precision",
        precision_score(
            y_test, original_pred, zero_division=0
        )
    )
    mlflow.log_metric(
        "baseline_recall",
        recall_score(
            y_test, original_pred, zero_division=0
        )
    )
    mlflow.log_metric(
        "baseline_f1",
        f1_score(
            y_test, original_pred, zero_division=0
        )
    )

    # Record optimized test performance
    mlflow.log_metric(
        "optimized_accuracy",
        accuracy_score(y_test, optimized_pred)
    )
    mlflow.log_metric(
        "optimized_precision",
        precision_score(
            y_test, optimized_pred, zero_division=0
        )
    )
    mlflow.log_metric(
        "optimized_recall",
        recall_score(
            y_test, optimized_pred, zero_division=0
        )
    )
    mlflow.log_metric(
        "optimized_f1",
        f1_score(
            y_test, optimized_pred, zero_division=0
        )
    )
    mlflow.log_metric(
        "optimized_roc_auc",
        roc_auc_score(y_test, test_prob)
    )

    # Record cross-validation optimization results
    mlflow.log_metric(
        "threshold_cv_f1",
        float(best_row["F1"])
    )
    mlflow.log_metric(
        "threshold_cv_recall",
        float(best_row["Recall"])
    )
    mlflow.log_metric(
        "threshold_cv_precision",
        float(best_row["Precision"])
    )

print("\nMLflow threshold optimization experiment logged successfully!")


# --------------------------------------------------
# 9. Save optimized threshold automatically
# --------------------------------------------------

import json

threshold_path = DEPLOYMENT_DIR / "threshold.json"

threshold_data = {
    "classification_threshold": best_threshold,
    "optimization_metric": "f1",
    "selection_method": "5-fold stratified cross-validation"
}

with open(threshold_path, "w") as f:
    json.dump(threshold_data, f, indent=4)

print("\nOptimized threshold saved automatically!")
print("Threshold:", best_threshold)
print("File:", threshold_path)
