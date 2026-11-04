
from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"

DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

# Local MLflow tracking directory
MLFLOW_DIR = PROJECT_DIR / "mlruns"
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())
mlflow.set_experiment("Wellness_Tourism_Prediction")


# --------------------------------------------------
# 2. Load training and testing datasets
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)

TARGET = "ProdTaken"

# Exclude identifiers and features unavailable before
# customer contact to avoid prediction-time leakage.
EXCLUDED_COLUMNS = [
    "CustomerID",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch"
]

X_train = train_df.drop(
    columns=[TARGET] + EXCLUDED_COLUMNS,
    errors="ignore"
)
y_train = train_df[TARGET]

X_test = test_df.drop(
    columns=[TARGET] + EXCLUDED_COLUMNS,
    errors="ignore"
)
y_test = test_df[TARGET]

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and test feature columns differ.")

if y_train.nunique() != 2:
    raise ValueError("Training data must contain both target classes.")


# --------------------------------------------------
# 3. Identify numerical and categorical features
# --------------------------------------------------

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)


# --------------------------------------------------
# 4. Define preprocessing pipelines
# --------------------------------------------------

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# --------------------------------------------------
# 5. Define XGBoost classifier
# --------------------------------------------------

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])


# --------------------------------------------------
# 6. Hyperparameter grid
# --------------------------------------------------

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.1]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)


# --------------------------------------------------
# 7. Train model and track experiment with MLflow
# --------------------------------------------------

with mlflow.start_run(run_name="XGBoost_GridSearch"):

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print("\nBest cross-validation F1:")
    print(round(grid_search.best_score_, 4))

    # Log tuning parameters
    mlflow.log_params(grid_search.best_params_)

    mlflow.log_param("model_type", "XGBClassifier")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring", "f1")
    mlflow.log_param("random_state", 42)

    mlflow.log_metric(
        "cv_best_f1",
        grid_search.best_score_
    )

    # ----------------------------------------------
    # 8. Evaluate on held-out test data
    # ----------------------------------------------

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    print("\nTest evaluation metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
        mlflow.log_metric(name, value)

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

    # ----------------------------------------------
    # 9. Save deployment-ready pipeline
    # ----------------------------------------------

    model_path = DEPLOYMENT_DIR / "best_model.pkl"

    joblib.dump(best_model, model_path)

    print(f"\nModel saved to: {model_path}")

    # Save expected feature names for deployment
    feature_path = DEPLOYMENT_DIR / "features.json"

    with open(feature_path, "w") as f:
        json.dump(
            X_train.columns.tolist(),
            f,
            indent=4
        )

    # Log model artifact to MLflow
    mlflow.log_artifact(
        str(model_path),
        artifact_path="deployment"
    )

    mlflow.log_artifact(
        str(feature_path),
        artifact_path="deployment"
    )

    # Save metrics as a JSON file
    metrics_path = DEPLOYMENT_DIR / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.log_artifact(
        str(metrics_path),
        artifact_path="deployment"
    )

print("\nTraining and MLflow tracking completed!")
