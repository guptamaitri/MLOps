import pandas as pd
import os
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from preprocessing import (
    fill_missing_values,
    split_data,
    apply_smote,
    save_columns
)
from utils import (
    get_git_commit,
    create_confusion_matrix,
    save_classification_report
)

DATA_PATH = "dataset.csv"
EXPERIMENT_NAME = "CustomerChurn"
MODEL_NAME = "CustomerChurnModel"
MIN_ACCURACY = 0.80

# -----------------------------
# Load and preprocess data
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = fill_missing_values(df)
X_train, X_test, y_train, y_test = split_data(df)
X_train_s, y_train_s = apply_smote(
    X_train,
    y_train
)
save_columns(X_train)
# -----------------------------
# MLflow Experiment
# -----------------------------
mlflow.set_experiment(
    EXPERIMENT_NAME
)

# mlflow.set_tracking_uri(
#     os.environ["MLFLOW_TRACKING_URI"]
# )

commit_hash = get_git_commit()
# -----------------------------
# Start MLflow Run
# -----------------------------
with mlflow.start_run():
    # Model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )

    # Train
    model.fit(
        X_train_s,
        y_train_s
    )

    # Signature
    signature = infer_signature(
        X_train_s,
        model.predict(X_train_s)
    )

    # Predictions
    y_pred = model.predict(
        X_test
    )
    y_pred_prob = model.predict_proba(
        X_test
    )[:,1]
    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )
    precision = precision_score(
        y_test,
        y_pred
    )
    recall = recall_score(
        y_test,
        y_pred
    )
    f1 = f1_score(
        y_test,
        y_pred
    )
    roc_auc = roc_auc_score(
        y_test,
        y_pred_prob
    )

    # -----------------------------
    # Validation before deployment
    # -----------------------------
    if accuracy < MIN_ACCURACY:
        raise Exception(
            f"Accuracy {accuracy} is below threshold"
        )

    # -----------------------------
    # Log Parameters
    # -----------------------------
    mlflow.log_params(
        model.get_params()
    )

    # -----------------------------
    # Log Metrics
    # -----------------------------
    mlflow.log_metric(
        "accuracy",
        accuracy
    )
    mlflow.log_metric(
        "precision",
        precision
    )
    mlflow.log_metric(
        "recall",
        recall
    )
    mlflow.log_metric(
        "f1_score",
        f1
    )
    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    # -----------------------------
    # Tags
    # -----------------------------
    mlflow.set_tag(
        "owner",
        "Maitri"
    )
    mlflow.set_tag(
        "dataset_version",
        "v1"
    )
    mlflow.set_tag(
        "git_commit",
        commit_hash
    )
    # -----------------------------
    # Artifacts
    # -----------------------------
    mlflow.log_artifact(
        "models/columns.json"
    )
    cm_path = create_confusion_matrix(
        y_test,
        y_pred
    )
    mlflow.log_artifact(
        cm_path
    )
    report_path = save_classification_report(
        y_test,
        y_pred
    )
    mlflow.log_artifact(
        report_path
    )

    # -----------------------------
    # Log and Register Model
    # -----------------------------

    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train_s.iloc[:5],
        registered_model_name=
        MODEL_NAME
    )


    # -----------------------------
    # Print metrics
    # -----------------------------

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC AUC  :", roc_auc)
