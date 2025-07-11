import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/saiskrithik.k/kidney-tumor.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "saiskrithik.k"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "4eea8f63a1f4bb90c2b5e41a72a2f8c055b47286"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with mlflow.start_run():
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_accuracy", 0.85)

