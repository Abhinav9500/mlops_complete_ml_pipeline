import os
import numpy as np
import pandas as pd
import pickle
import json
import logging

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from dvclive import Live

# ============================================================
# LOGGING SETUP
# ============================================================

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "model_evaluation.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_model(file_path: str):
    """Load trained model."""
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", file_path)
        return model
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data."""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba)
        }

        logger.debug("Evaluation metrics calculated")
        return metrics

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save metrics to JSON."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.debug("Metrics saved to %s", file_path)
    except Exception as e:
        logger.error("Failed to save metrics: %s", e)
        raise

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    try:
        # Hard-coded model parameters (since no params.yaml)
        params = {
            "n_estimators": 25,
            "random_state": 2
        }

        # Load model and test data
        model = load_model("./models/model.pkl")
        test_data = load_data("./data/processed/test_tfidf.csv")

        # Split features and labels
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Log experiments using dvclive
        with Live(save_dvc_exp=True) as live:
            for key, value in metrics.items():
                live.log_metric(key, value)

            live.log_params(params)

        # Save metrics to reports
        save_metrics(metrics, "reports/metrics.json")

        logger.info("Model evaluation completed successfully")

    except Exception as e:
        logger.error("Model evaluation pipeline failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
