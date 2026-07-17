import os
import pandas as pd  # pyright: ignore [reportMissingImports]
import numpy as np  # pyright: ignore [reportMissingImports]
from loguru import logger  # pyright: ignore [reportMissingImports]

# Use non-interactive Agg backend for headless environments
import matplotlib  # pyright: ignore [reportMissingImports]
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pyright: ignore [reportMissingImports]
import seaborn as sns  # pyright: ignore [reportMissingImports]

from sklearn.metrics import (  # pyright: ignore [reportMissingImports]
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

from src.utils import setup_logging, set_seed, load_env_variables
from src.data_pipeline import load_raw_data, split_data
from src.feature_engineering import TransactionTransformer
from src.model_pipeline import (
    train_logistic_regression,
    tune_logistic_regression,
    train_random_forest,
    train_xgboost,
    evaluate_model,
    build_and_save_pipeline
)

def plot_confusion_matrices(models_dict: dict, X_test: pd.DataFrame, y_test: np.ndarray, output_dir: str):
    """Generates and saves a multi-panel confusion matrix plot for all models."""
    fig, axes = plt.subplots(1, len(models_dict), figsize=(6 * len(models_dict), 5))
    if len(models_dict) == 1:
        axes = [axes]
        
    for ax, (model_name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Purples",
            ax=ax,
            cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["Legit (0)", "Fraud (1)"],
            yticklabels=["Legit (0)", "Fraud (1)"]
        )
        ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
        ax.set_ylabel("True Label", fontsize=10, labelpad=8)
        
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrices plot to {filepath}")

def plot_roc_curves(models_dict: dict, X_test: pd.DataFrame, y_test: np.ndarray, output_dir: str):
    """Generates and saves a combined ROC curve plot for all models."""
    plt.figure(figsize=(8, 6))
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess (AUC = 0.50)")
    
    # Custom harmonious colors
    colors = {
        "Logistic Regression (Untuned)": "#1f77b4",
        "Logistic Regression (Tuned)": "#d62728",
        "Random Forest": "#2ca02c",
        "XGBoost": "#ff7f0e"
    }
    
    for model_name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = model.predict(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        color = colors.get(model_name, None)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", linewidth=2, color=color)
        
    plt.title("ROC Curves Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate", fontsize=11, labelpad=8)
    plt.ylabel("True Positive Rate", fontsize=11, labelpad=8)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    filepath = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curves plot to {filepath}")

def plot_precision_recall_curves(models_dict: dict, X_test: pd.DataFrame, y_test: np.ndarray, output_dir: str):
    """Generates and saves a combined Precision-Recall curve plot for all models."""
    plt.figure(figsize=(8, 6))
    
    # Plot baseline reference line
    baseline = sum(y_test) / len(y_test)
    plt.axhline(y=baseline, linestyle="--", color="gray", label=f"Baseline (AP = {baseline:.4f})")
    
    # Custom harmonious colors
    colors = {
        "Logistic Regression (Untuned)": "#1f77b4",
        "Logistic Regression (Tuned)": "#d62728",
        "Random Forest": "#2ca02c",
        "XGBoost": "#ff7f0e"
    }
    
    for model_name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = model.predict(X_test)
            
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        
        color = colors.get(model_name, None)
        plt.plot(recall, precision, label=f"{model_name} (AP = {ap:.4f})", linewidth=2, color=color)
        
    plt.title("Precision-Recall Curves Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Recall", fontsize=11, labelpad=8)
    plt.ylabel("Precision", fontsize=11, labelpad=8)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    filepath = os.path.join(output_dir, "precision_recall_curves.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Precision-Recall curves plot to {filepath}")

def plot_feature_importances(models_dict: dict, feature_names: list, output_dir: str):
    """Generates and saves a multi-panel feature importance/coefficient plot for all models."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # 1. Logistic Regression Coefficients (Untuned)
    lr_model = models_dict.get("Logistic Regression (Untuned)")
    if lr_model:
        coefs = lr_model.coef_[0]
        indices = np.argsort(np.abs(coefs))[-10:]
        axes[0].barh(range(10), coefs[indices], color="#1f77b4", alpha=0.85)
        axes[0].set_yticks(range(10))
        axes[0].set_yticklabels([feature_names[i] for i in indices], fontsize=10)
        axes[0].set_title("Logistic Regression (Untuned)\nTop 10 Coefficients", fontsize=11, fontweight="bold", pad=10)
        axes[0].set_xlabel("Coefficient Value", fontsize=10)
        axes[0].grid(True, axis="x", linestyle=":", alpha=0.6)
        
    # 2. Logistic Regression Coefficients (Tuned)
    lr_tuned_model = models_dict.get("Logistic Regression (Tuned)")
    if lr_tuned_model:
        coefs = lr_tuned_model.coef_[0]
        indices = np.argsort(np.abs(coefs))[-10:]
        axes[1].barh(range(10), coefs[indices], color="#d62728", alpha=0.85)
        axes[1].set_yticks(range(10))
        axes[1].set_yticklabels([feature_names[i] for i in indices], fontsize=10)
        axes[1].set_title("Logistic Regression (Tuned)\nTop 10 Coefficients", fontsize=11, fontweight="bold", pad=10)
        axes[1].set_xlabel("Coefficient Value", fontsize=10)
        axes[1].grid(True, axis="x", linestyle=":", alpha=0.6)
        
    # 3. Random Forest Feature Importances
    rf_model = models_dict.get("Random Forest")
    if rf_model:
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[-10:]
        axes[2].barh(range(10), importances[indices], color="#2ca02c", alpha=0.85)
        axes[2].set_yticks(range(10))
        axes[2].set_yticklabels([feature_names[i] for i in indices], fontsize=10)
        axes[2].set_title("Random Forest\nTop 10 Feature Importances", fontsize=11, fontweight="bold", pad=10)
        axes[2].set_xlabel("Relative Importance", fontsize=10)
        axes[2].grid(True, axis="x", linestyle=":", alpha=0.6)
        
    # 4. XGBoost Feature Importances
    xgb_model = models_dict.get("XGBoost")
    if xgb_model:
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[-10:]
        axes[3].barh(range(10), importances[indices], color="#ff7f0e", alpha=0.85)
        axes[3].set_yticks(range(10))
        axes[3].set_yticklabels([feature_names[i] for i in indices], fontsize=10)
        axes[3].set_title("XGBoost\nTop 10 Feature Importances", fontsize=11, fontweight="bold", pad=10)
        axes[3].set_xlabel("F-Score / Relative Importance", fontsize=10)
        axes[3].grid(True, axis="x", linestyle=":", alpha=0.6)
        
    plt.tight_layout()
    filepath = os.path.join(output_dir, "feature_importances.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importances plot to {filepath}")

def compare_models(lr_metrics: dict, lr_tuned_metrics: dict, rf_metrics: dict, xgb_metrics: dict):
    """Prints a comparison table between all models."""
    metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    
    print("\n" + "="*120)
    print(f"{'METRIC':<20} | {'LR (UNTUNED)':<22} | {'LR (TUNED)':<22} | {'RANDOM FOREST':<22} | {'XGBOOST':<22}")
    print("="*120)
    
    for metric in metrics_list:
        lr_val = lr_metrics.get(metric, 0.0)
        lr_tuned_val = lr_tuned_metrics.get(metric, 0.0)
        rf_val = rf_metrics.get(metric, 0.0)
        xgb_val = xgb_metrics.get(metric, 0.0)
        print(f"{metric.upper():<20} | {lr_val:<22.4f} | {lr_tuned_val:<22.4f} | {rf_val:<22.4f} | {xgb_val:<22.4f}")
        
    print("="*120)

def main():
    # 1. Setup configurations
    setup_logging("INFO")
    set_seed(42)
    load_env_variables()
    
    logger.info("Starting Phase 3: Model Training, Hyperparameter Tuning, and Evaluation...")
    
    # 2. Ingest raw data
    raw_df = load_raw_data()
    
    # 3. Perform Train-Test Split
    train_df, test_df = split_data(raw_df)
    
    # 4. Extract features (X) and target label (y)
    X_train_raw = train_df.drop(columns=["Class"])
    y_train = train_df["Class"].values
    X_test_raw = test_df.drop(columns=["Class"])
    y_test = test_df["Class"].values
    
    # 5. Preprocess using TransactionTransformer
    transformer = TransactionTransformer()
    X_train = transformer.fit_transform(X_train_raw)
    X_test = transformer.transform(X_test_raw)
    
    feature_names = list(X_train.columns)
    
    # 6. Train Logistic Regression (Untuned)
    logger.info("--- Training and Evaluating Logistic Regression (Untuned) ---")
    lr_model = train_logistic_regression(X_train, y_train, seed=42)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    
    # 7. Tune Logistic Regression using GridSearchCV (Tuned)
    logger.info("--- Tuning Logistic Regression with GridSearchCV ---")
    lr_tuned_model, best_params = tune_logistic_regression(X_train, y_train, seed=42)
    lr_tuned_metrics = evaluate_model(lr_tuned_model, X_test, y_test)
    
    # 8. Train Random Forest
    logger.info("--- Training and Evaluating Random Forest ---")
    rf_model = train_random_forest(X_train, y_train, seed=42)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    
    # 9. Train XGBoost
    logger.info("--- Training and Evaluating XGBoost ---")
    xgb_model = train_xgboost(X_train, y_train, seed=42)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
    
    # 10. Compare all models
    compare_models(lr_metrics, lr_tuned_metrics, rf_metrics, xgb_metrics)
    
    # 11. Generate and save all visualization plots
    output_figures_dir = "reports/figures"
    models_dict = {
        "Logistic Regression (Untuned)": lr_model,
        "Logistic Regression (Tuned)": lr_tuned_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }
    
    logger.info("Generating evaluation plots...")
    plot_confusion_matrices(models_dict, X_test, y_test, output_figures_dir)
    plot_roc_curves(models_dict, X_test, y_test, output_figures_dir)
    plot_precision_recall_curves(models_dict, X_test, y_test, output_figures_dir)
    plot_feature_importances(models_dict, feature_names, output_figures_dir)
    
    # 12. Programmatically identify the best model (highest PR-AUC) and serialize it
    metrics_dict = {
        "Logistic Regression (Untuned)": lr_metrics,
        "Logistic Regression (Tuned)": lr_tuned_metrics,
        "Random Forest": rf_metrics,
        "XGBoost": xgb_metrics
    }
    
    best_model_name = max(metrics_dict, key=lambda k: metrics_dict[k]["pr_auc"])
    best_model = models_dict[best_model_name]
    logger.info(f"Best model selected: {best_model_name} with PR-AUC of {metrics_dict[best_model_name]['pr_auc']:.4f}")
    
    pipeline_path = "models/fraud_detection_pipeline.pkl"
    build_and_save_pipeline(transformer, best_model, pipeline_path)
    logger.info(f"Successfully saved end-to-end best model pipeline to {pipeline_path}")

if __name__ == "__main__":
    main()
