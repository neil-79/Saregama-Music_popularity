"""Stage 3: Model training.

This module contains clean implementations of the training pipeline
extracted from the original messy code.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    classification_report
)

# Import the original DeepLearningModel (we keep it as-is since it's complex)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
hitmusic_root = PROJECT_ROOT / "hitmusiclyricnet" / "hitmusiclyricnet"
sys.path.insert(0, str(hitmusic_root))
from models.dnn_models import DeepLearningModel  # type: ignore

from .config import data_paths, model_config
from .data_processing import load_raw_data, preprocess_tracks_data


def compress_audio_features(X_sc: pd.DataFrame, compress: str = 'auto') -> pd.DataFrame:
    """Compress audio/metadata features using autoencoder.
    
    Extracted from hitmusiclyricnet/popularity_prediction.py apply_feature_compression
    """
    print('Applying Autoencoder to reduce the dimensionality ...')
    
    # Create a temporary DeepLearningModel to use its autoencoder
    temp_model = DeepLearningModel(
        model_name='temp_ae',
        model_dir='temp',
        model_subDir='temp',
        input_dim=X_sc.shape[1],
        output_dim=1,
        optimizer='adam',
        metrics=[],
        loss='mse',
        problem='regression'
    )
    
    encoder = temp_model.train_autoencoder(x_train=X_sc.values, epochs=50, compress=compress)
    x_compressed = encoder.predict(X_sc.values)
    
    # Scale compressed features
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_compressed = scaler.fit_transform(x_compressed)
    
    cols = [f'feat_compressed_{i + 1}' for i in range(x_compressed.shape[1])]
    return pd.DataFrame(x_compressed, columns=cols)


def prepare_training_data(
    task: str = 'regression',
    compress: str = 'auto',
    name_file: str = 'spotgentrack_lyricscleaned_2025'
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare all features for training: merge audio, lyrics, metadata.
    
    Extracted from hitmusiclyricnet/popularity_prediction.py train_model
    
    Returns:
        x_train, y_train, x_test, y_test, tracks_id
    """
    print("Preparing training data...")
    
    # Auto-run preprocessing if files don't exist
    if not data_paths.ae_compressed_audio_csv.exists():
        print("Compressed audio features not found.")
        print("Please run preprocessing first:")
        print("  python -m saregama.preprocessing")
        raise FileNotFoundError(
            f"Compressed audio features not found: {data_paths.ae_compressed_audio_csv}\n"
            "Run preprocessing first: python -m saregama.preprocessing"
        )
    
    total_df = pd.read_csv(data_paths.ae_compressed_audio_csv, index_col=0)
    target = 'popularity' if task == 'regression' else 'popularity_class'
    y_sc = total_df.loc[:, target]
    X_sc = total_df.drop([target], axis=1, inplace=False)
    
    # Load compressed lyrics embeddings
    if not data_paths.lyrics_compressed_csv.exists():
        raise FileNotFoundError(
            f"Compressed lyrics embeddings not found: {data_paths.lyrics_compressed_csv}\n"
            "Run embeddings compression first."
        )
    
    lyrics_df = pd.read_csv(data_paths.lyrics_compressed_csv)
    
    # Load high-level audio metadata
    if not data_paths.hl_audio_metadata_csv.exists():
        raise FileNotFoundError(
            f"High-level audio metadata not found: {data_paths.hl_audio_metadata_csv}\n"
            "Run data preprocessing first."
        )
    
    HLaudio_metadata_df = pd.read_csv(data_paths.hl_audio_metadata_csv, index_col=0)
    
    # Load track IDs
    if not data_paths.tracks_id_csv.exists():
        raise FileNotFoundError(
            f"Track IDs not found: {data_paths.tracks_id_csv}\n"
            "Run data preprocessing first."
        )
    
    tracks_id = pd.read_csv(data_paths.tracks_id_csv, index_col=0)
    
    # Merge all features
    x_total = pd.concat([X_sc, y_sc, tracks_id], axis=1)
    x_total = pd.merge(x_total, lyrics_df, on='track_id', how='inner')
    x_total = pd.merge(x_total, HLaudio_metadata_df, on='track_id', how='inner')
    
    print(f"Final feature matrix shape: {x_total.shape}")
    
    # Split train/test
    train, test = train_test_split(x_total, test_size=0.20, random_state=42)
    
    y_train = train.loc[:, target]
    x_train = train.drop([target, 'track_id'], axis=1, inplace=False)
    y_test = test.loc[:, target]
    x_test = test.drop([target, 'track_id'], axis=1, inplace=False)
    
    # Save test set
    output_dir = hitmusic_root / "input"
    output_dir.mkdir(exist_ok=True)
    x_test.to_csv(output_dir / f'x_test_{task}_{name_file}.csv')
    y_test.to_csv(output_dir / f'y_test_{task}_{name_file}.csv')
    
    return x_train, y_train, x_test, y_test, tracks_id


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    model_args: dict,
    task: str = 'regression'
) -> dict:
    """Train DNN model for one fold.
    
    Extracted from hitmusiclyricnet/popularity_prediction.py train_model
    """
    dl_model = DeepLearningModel(
        model_name=model_args['model_name'],
        model_dir=model_args['model_dir'],
        model_subDir=model_args['model_subDir'],
        input_dim=x_train.shape[1],
        output_dim=model_args['output_dim'],
        optimizer=model_args['optimizer'],
        metrics=model_args['metrics'],
        loss=model_args['loss'],
        add_earlyStopping=model_args['earlyStop'],
        weights_path=model_args['weights'],
        plot_loss=model_args['plot_loss'],
        neuron_parameters=model_args['neurons'],
        layers=model_args['n_layers'],
        initialization=model_args['weights_init'],
        level_dropout=model_args['dropout'],
        problem=task
    )
    
    dl_model.build_model()
    dl_model.compile_model()
    
    history = dl_model.train_model(
        x_train.values, y_train.values,
        x_val.values, y_val.values,
        epochs=model_args['epochs'],
        batch_size=model_args['batch_size'],
        monitor_early=model_args['early_mon'],
        mode=model_args['mode'],
        monitor_checkout=model_args['checkout_mon']
    )
    
    dl_model.save_model()
    
    return history.history


def run_training(
    n_splits: int = 5,
    task: str = 'regression',
    compress: str = 'auto',
    name_file: str = 'spotgentrack_lyricscleaned_2025'
) -> dict:
    """Main training pipeline with k-fold cross-validation.
    
    Extracted from hitmusiclyricnet/popularity_prediction.py train_model
    """
    print("Saregama – Stage 3: Model training")
    
    # Prepare data
    x_train, y_train, x_test, y_test, tracks_id = prepare_training_data(
        task=task, compress=compress, name_file=name_file
    )
    
    # Get model config
    model_args = model_config.to_model_args()
    model_args['model_subDir'] += '_' + compress
    
    # Setup cross-validation
    if task == 'classification':
        skf = StratifiedKFold(n_splits=n_splits)
        splits = skf.split(x_train, y_train)
    else:
        skf = KFold(n_splits=n_splits, shuffle=True)
        splits = skf.split(x_train.index)
    
    # Train on each fold
    model_history_cv = {}
    start_time = time.time()
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        print(f"\n=== Training fold {fold_idx + 1}/{n_splits} ===")
        
        x_train_fold = x_train.iloc[train_indices]
        x_val_fold = x_train.iloc[val_indices]
        y_train_fold = y_train.iloc[train_indices]
        y_val_fold = y_train.iloc[val_indices]
        
        # Round to avoid precision issues
        x_train_fold = x_train_fold.round(5)
        x_val_fold = x_val_fold.round(5)
        y_train_fold = y_train_fold.round(5)
        y_val_fold = y_val_fold.round(5)
        
        history = train_model(
            x_train_fold, y_train_fold,
            x_val_fold, y_val_fold,
            model_args, task=task
        )
        
        # Save metrics
        for metric in history.keys():
            if metric not in model_history_cv:
                model_history_cv[metric] = []
            model_history_cv[metric].append(history[metric][-1])
    
    elapsed_time = time.time() - start_time
    model_history_cv['cpu (ms)'] = elapsed_time
    
    # Evaluate on test set
    print("\n=== Evaluating on test set ===")
    
    # Load best model from last fold
    dl_model = DeepLearningModel(
        model_name=model_args['model_name'],
        model_dir=model_args['model_dir'],
        model_subDir=model_args['model_subDir'],
        input_dim=x_train.shape[1],
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
    
    dl_model.build_model()
    dl_model.compile_model()
    y_pred = dl_model.validate_model(x_test.values)
    
    # Compute metrics
    if task == 'regression':
        mae = mean_absolute_error(y_test.values, y_pred)
        mse = mean_squared_error(y_test.values, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test.values, y_pred)
        ev = explained_variance_score(y_test.values, y_pred)
        mape = np.mean(np.abs((y_test.values - y_pred.flatten()) / 
                              np.maximum(np.abs(y_test.values), 1e-10))) * 100
        
        model_history_cv['test_mean_absolute_error'] = mae
        model_history_cv['test_mean_squared_error'] = mse
        model_history_cv['test_root_mean_squared_error'] = rmse
        model_history_cv['test_r2_score'] = r2
        model_history_cv['test_explained_variance'] = ev
        model_history_cv['test_mean_absolute_percentage_error'] = mape
        
        print("\nTest Set Metrics:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  EV:   {ev:.4f}")
        print(f"  MAPE: {mape:.2f}%")
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
        target_names = ['Low', 'Medium', 'High']
        report = classification_report(
            y_test.values, y_pred_classes,
            target_names=target_names,
            output_dict=True
        )
        model_history_cv['test_classification_report'] = report
        print("\nTest Set Classification Report:")
        print(classification_report(y_test.values, y_pred_classes, target_names=target_names))
    
    print(f"\n✓ Training completed in {elapsed_time:.2f} seconds")
    
    return model_history_cv


if __name__ == "__main__":
    run_training()
