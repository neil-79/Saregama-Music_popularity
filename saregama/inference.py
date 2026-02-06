"""Stage 4: Inference and testing.

This module loads trained models and evaluates them on test data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    classification_report
)

# Import the original DeepLearningModel
PROJECT_ROOT = Path(__file__).resolve().parents[1]
hitmusic_root = PROJECT_ROOT / "hitmusiclyricnet" / "hitmusiclyricnet"
sys.path.insert(0, str(hitmusic_root))
from models.dnn_models import DeepLearningModel  # type: ignore

from .config import model_config


def load_trained_model_and_predict(
    x_test_path: Path | None = None,
    y_test_path: Path | None = None,
    task: str = 'regression',
    name_file: str = 'spotgentrack_lyricscleaned_2025'
) -> dict:
    """Load trained model and evaluate on test set.
    
    Args:
        x_test_path: Path to test features CSV
        y_test_path: Path to test labels CSV
        task: 'regression' or 'classification'
        name_file: Name file used during training
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("Saregama – Stage 4: Inference and testing")
    
    # Default paths
    if x_test_path is None:
        x_test_path = hitmusic_root / "input" / f"x_test_{task}_{name_file}.csv"
    if y_test_path is None:
        y_test_path = hitmusic_root / "input" / f"y_test_{task}_{name_file}.csv"
    
    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"Test files not found:\n"
            f"  - {x_test_path}\n"
            f"  - {y_test_path}\n"
            "Make sure you have run the training stage first."
        )
    
    # Load test data
    x_test = pd.read_csv(x_test_path, index_col=0)
    y_test = pd.read_csv(y_test_path, index_col=0).iloc[:, 0]
    
    print(f"Test set shape: X={x_test.shape}, y={y_test.shape}")
    
    # Build model with same config
    model_args = model_config.to_model_args()
    
    dl_model = DeepLearningModel(
        model_name=model_args['model_name'],
        model_dir=model_args['model_dir'],
        model_subDir=model_args['model_subDir'],
        input_dim=x_test.shape[1],
        output_dim=model_args['output_dim'],
        optimizer=model_args['optimizer'],
        metrics=model_args['metrics'],
        loss=model_args['loss'],
        add_earlyStopping=False,
        weights_path=model_args['weights'],
        plot_loss=False,
        neuron_parameters=model_args['neurons'],
        layers=model_args['n_layers'],
        initialization=model_args['weights_init'],
        level_dropout=model_args['dropout'],
        problem=task,
        saved_weights=True,
        load_weights=True
    )
    
    # Build, compile, and predict
    print("Loading model...")
    dl_model.build_model()
    dl_model.compile_model()
    
    print("Running predictions...")
    y_pred = dl_model.validate_model(x_test.values)
    
    # Compute metrics
    metrics = {}
    
    if task == 'regression':
        y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        y_test_vals = y_test.values
        
        metrics['mae'] = mean_absolute_error(y_test_vals, y_pred_flat)
        metrics['mse'] = mean_squared_error(y_test_vals, y_pred_flat)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_test_vals, y_pred_flat)
        metrics['ev'] = explained_variance_score(y_test_vals, y_pred_flat)
        metrics['mape'] = np.mean(
            np.abs((y_test_vals - y_pred_flat) / np.maximum(np.abs(y_test_vals), 1e-10))
        ) * 100
        
        print("\n" + "=" * 50)
        print("Test Set Evaluation Metrics")
        print("=" * 50)
        print(f"Mean Absolute Error (MAE):     {metrics['mae']:.4f}")
        print(f"Mean Squared Error (MSE):      {metrics['mse']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        print(f"R² Score:                      {metrics['r2']:.4f}")
        print(f"Explained Variance:            {metrics['ev']:.4f}")
        print(f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
        print("=" * 50)
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
        target_names = ['Low', 'Medium', 'High']
        
        report = classification_report(
            y_test.values, y_pred_classes,
            target_names=target_names,
            output_dict=True
        )
        metrics['classification_report'] = report
        
        print("\n" + "=" * 50)
        print("Test Set Classification Report")
        print("=" * 50)
        print(classification_report(
            y_test.values, y_pred_classes,
            target_names=target_names
        ))
        print("=" * 50)
    
    return metrics


if __name__ == "__main__":
    load_trained_model_and_predict()
