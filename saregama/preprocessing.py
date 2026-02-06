"""Data preprocessing: Create all intermediate files needed for training.

This module creates:
- Compressed audio/metadata features
- High-level audio metadata features
- Track IDs
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
hitmusic_root = PROJECT_ROOT / "hitmusiclyricnet" / "hitmusiclyricnet"
sys.path.insert(0, str(hitmusic_root))
from popularity_prediction import PopularityPrediction  # type: ignore

from .config import data_paths
from .data_processing import load_raw_data


def run_preprocessing(
    task: str = 'regression',
    compress: str = 'auto',
    name_file: str = 'spotgentrack_lyricscleaned_2025'
) -> None:
    """Run full preprocessing pipeline to create all intermediate files.
    
    Uses the original PopularityPrediction class to ensure compatibility.
    
    Creates:
    - tracks_id_{task}_{name_file}.csv
    - HL_audio_metadata_features_{name_file}_{task}.csv
    - AE_compressed_auto_LLaudio_{name_file}.csv
    """
    print("Saregama – Data Preprocessing")
    print("=" * 50)
    
    # Ensure output directory exists
    output_dir = hitmusic_root / "input"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to hitmusic directory for relative paths to work
    original_cwd = os.getcwd()
    os.chdir(str(hitmusic_root))
    
    try:
        # Load raw data
        print("\n1. Loading raw data...")
        df_tracks, df_artists, df_albums, df_audio_features, df_lyrics_features = load_raw_data()
        
        # Create dummy model_args (not used for preprocessing)
        model_args = {
            'model_name': 'temp',
            'model_dir': 'temp',
            'model_subDir': 'temp',
            'input_dim': 1,
            'output_dim': 1,
            'optimizer': 'adam',
            'metrics': [],
            'loss': 'mse'
        }
        
        # Use original PopularityPrediction class for preprocessing
        print("\n2. Running preprocessing using PopularityPrediction...")
        popularity_model = PopularityPrediction(
            df_tracks=df_tracks,
            df_artists=df_artists,
            df_albums=df_albums,
            df_audio_features=df_audio_features,
            df_text_features=df_lyrics_features,
            model_args=model_args,
            mode='Train',
            task=task,
            new_preprocessing=True,
            compress=compress
        )
        
        # Run preprocessing (this creates all the files)
        popularity_model.data_preprocessing(
            compressed=True,
            compressed_method='AE',
            save_data=True,
            name_file=name_file
        )
        
        print("\n" + "=" * 50)
        print("✓ Preprocessing completed successfully!")
        print(f"  Created AE_compressed_{compress}_LLaudio_{name_file}.csv")
        print(f"  Created tracks_id_{task}_{name_file}.csv")
        print(f"  Created HL_audio_metadata_features_{name_file}_{task}.csv")
        
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    run_preprocessing()
