import os
import joblib
import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)
from sklearn.pipeline import Pipeline
from src.feature_engineering import TransactionTransformer

def train_logistic_regression(X_train, y_train, seed: int = 42) -> LogisticRegression:
    """Trains a Logistic Regression model with class weighting."""
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")
    return model

def train_random_forest(X_train, y_train, seed: int = 42) -> RandomForestClassifier:
    """Trains a Random Forest model with class weighting."""
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(class_weight="balanced", random_state=seed, n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return model

def train_xgboost(X_train, y_train, seed: int = 42) -> XGBClassifier:
    """Trains an XGBoost model with scale_pos_weight."""
    logger.info("Training XGBoost model...")
    
    # Calculate class weighting ratio
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")
    
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("XGBoost training complete.")
    return model

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluates the model on test data and returns a dictionary of metrics.
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Check if model has predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred
        
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob)
    }
    
    logger.info(f"Model Evaluation Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f" - {metric_name.upper():<10}: {value:.4f}")
        
    logger.debug("\n" + classification_report(y_test, y_pred, zero_division=0))
    return metrics

def build_and_save_pipeline(transformer: TransactionTransformer, classifier, filepath: str) -> Pipeline:
    """
    Combines the fitted transformer and trained classifier into a single scikit-learn Pipeline
    and serializes it to disk.
    """
    logger.info(f"Building complete Pipeline with transformer and classifier...")
    pipeline = Pipeline([
        ("transformer", transformer),
        ("classifier", classifier)
    ])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    logger.info(f"Saving serialized pipeline to {filepath}...")
    joblib.dump(pipeline, filepath)
    logger.info("Pipeline saved successfully.")
    return pipeline

def load_pipeline(filepath: str) -> Pipeline:
    """Loads a serialized pipeline from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No serialized model found at {filepath}")
    logger.info(f"Loading pipeline from {filepath}...")
    pipeline = joblib.load(filepath)
    logger.info("Pipeline loaded successfully.")
    return pipeline
