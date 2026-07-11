import logging
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend for matplotlib to avoid issues when running in web server or headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Set up logger
logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics:
      - MAE (Mean Absolute Error)
      - MSE (Mean Squared Error)
      - RMSE (Root Mean Squared Error)
      - R² Score (Coefficient of Determination)
    """
    logger.info("Calculating model evaluation metrics...")
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }
    
    for metric_name, val in metrics.items():
        logger.info(f"Metric - {metric_name}: {val:.4f}")
        
    return metrics

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path):
    """
    Plots Actual vs Predicted prices with a 45-degree reference line.
    """
    logger.info(f"Generating Actual vs Predicted plot -> {save_path}")
    plt.figure(figsize=(7, 5))
    
    # Use dark or modern neutral color palette
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color="#4F46E5", edgecolor="w", linewidth=0.5)
    
    # Draw reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#EF4444", linestyle="--", linewidth=1.5, label="Perfect Fit")
    
    plt.title("Actual vs. Predicted House Prices", fontsize=12, pad=15, fontweight="semibold")
    plt.xlabel("Actual Prices ($)", labelpad=10)
    plt.ylabel("Predicted Prices ($)", labelpad=10)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path):
    """
    Plots residuals (Error = Actual - Predicted) against predicted values.
    Ideally, residuals should be randomly distributed around horizontal line at y=0.
    """
    logger.info(f"Generating Residuals plot -> {save_path}")
    residuals = y_true - y_pred
    
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color="#10B981", edgecolor="w", linewidth=0.5)
    plt.axhline(y=0, color="#EF4444", linestyle="--", linewidth=1.5)
    
    plt.title("Residuals vs. Predicted Prices", fontsize=12, pad=15, fontweight="semibold")
    plt.xlabel("Predicted Prices ($)", labelpad=10)
    plt.ylabel("Residuals / Errors ($)", labelpad=10)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_prediction_error(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path):
    """
    Plots the distribution of prediction errors.
    """
    logger.info(f"Generating Prediction Error distribution plot -> {save_path}")
    errors = y_true - y_pred
    
    plt.figure(figsize=(7, 5))
    # Plot histogram and KDE
    sns.histplot(errors, kde=True, color="#6366F1", bins=25, edgecolor="w")
    plt.axvline(x=0, color="#EF4444", linestyle="--", linewidth=1.5, label="Zero Error")
    
    plt.title("Distribution of Prediction Errors (Residuals)", fontsize=12, pad=15, fontweight="semibold")
    plt.xlabel("Prediction Error ($)", labelpad=10)
    plt.ylabel("Count / Density", labelpad=10)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def run_evaluation_suite(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> Dict[str, float]:
    """
    Runs the entire evaluation suite: calculates metrics and saves all three plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = calculate_metrics(y_true, y_pred)
    
    plot_actual_vs_predicted(y_true, y_pred, output_dir / "actual_vs_predicted.png")
    plot_residuals(y_true, y_pred, output_dir / "residuals_plot.png")
    plot_prediction_error(y_true, y_pred, output_dir / "prediction_error.png")
    
    return metrics
