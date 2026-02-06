"""Stage 1: Data cleaning and preprocessing.

This module contains clean implementations of data preprocessing logic
extracted from the original messy code.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from functools import reduce
from sklearn.preprocessing import MinMaxScaler

from .config import data_paths


def clean_lyrics(text: str) -> str:
    """Clean lyrics text by removing line numbers and escape sequences.
    
    Extracted from embeddings_extraction/openai_embeddings.py
    """
    if not isinstance(text, str):
        return ""
    
    # Remove escape sequences
    text = text.replace('\r\n', '\n')
    
    # Remove line numbers at the beginning of lines
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove leading numbers with spaces/tabs
        cleaned_line = line.strip()
        if cleaned_line and cleaned_line[0].isdigit():
            # Check if line starts with a number followed by whitespace
            parts = cleaned_line.split(maxsplit=1)
            if len(parts) > 1 and parts[0].isdigit():
                cleaned_line = parts[1]
        
        if cleaned_line:  # Only add non-empty lines
            cleaned_lines.append(cleaned_line)
    
    # Join with spaces instead of newlines
    return ' '.join(cleaned_lines)


def get_artist_data(row: pd.Series, df_artists: pd.DataFrame, col_name: str) -> tuple:
    """Extract artist statistics (count, max, min, median) for a track.
    
    Extracted from hitmusiclyricnet/popularity_prediction.py
    """
    try:
        artist_id_n = row['artists_id'][2:-2].replace("'", "").split()  # Remove brackets
        artist_id = [a_id.replace(",", "") for a_id in artist_id_n]
        total_artist_count = len(artist_id)
        out = [total_artist_count]
        
        for a_id in artist_id:
            res = df_artists[df_artists['id'] == a_id]
            if res.shape[0] > 0:
                art_data = res[col_name].values[0]
                out.append(art_data)
        
        max_out = max(out)
        min_out = min(out)
        median_out = np.median(out)
        return total_artist_count, max_out, min_out, median_out
    except Exception as e:
        print(f"Error in get_artist_data: {e}")
        return 0, 0, 0, 0


def get_total_markets(row: pd.Series) -> int:
    """Count number of available markets for a track.
    
    Extracted from hitmusiclyricnet/popularity_prediction.py
    """
    try:
        n_markets = len(row['available_markets'][2:-2].replace("'", "").split())
        return n_markets
    except Exception as e:
        print(f"Error in get_total_markets: {e}")
        return 0


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all raw data CSVs.
    
    Returns:
        df_tracks, df_artists, df_albums, df_audio_features, df_lyrics_features
    """
    print("Loading raw data files...")
    
    # Try cleaned tracks first, fallback to raw
    if data_paths.tracks_clean_en_csv.exists():
        df_tracks = pd.read_csv(data_paths.tracks_clean_en_csv, index_col=0)
        print(f"Loaded tracks (cleaned): {df_tracks.shape}")
        
        if data_paths.tracks_clean_other_csv.exists():
            df_tracks_2 = pd.read_csv(data_paths.tracks_clean_other_csv, index_col=0)
            df_tracks = pd.concat([df_tracks, df_tracks_2], axis=0)
            print(f"Loaded tracks (otherlang): {df_tracks_2.shape}")
            print(f"Total tracks after merge: {df_tracks.shape}")
    else:
        df_tracks = pd.read_csv(data_paths.tracks_raw_csv, index_col=0)
        print(f"Loaded tracks (raw): {df_tracks.shape}")
    
    df_artists = pd.read_csv(data_paths.artists_csv, index_col=0)
    df_albums = pd.read_csv(data_paths.albums_csv, index_col=0)
    df_audio_features = pd.read_csv(data_paths.audio_lowlevel_csv, index_col=0)
    df_lyrics_features = pd.read_csv(data_paths.lyrics_feature_csv, index_col=0)
    
    print(f"Artists: {df_artists.shape}")
    print(f"Albums: {df_albums.shape}")
    print(f"Audio features: {df_audio_features.shape}")
    print(f"Lyrics features: {df_lyrics_features.shape}")
    
    return df_tracks, df_artists, df_albums, df_audio_features, df_lyrics_features


def preprocess_tracks_data(
    df_tracks: pd.DataFrame,
    df_artists: pd.DataFrame,
    df_albums: pd.DataFrame,
    df_audio_features: pd.DataFrame,
    task: str = 'regression'
) -> pd.DataFrame:
    """Preprocess tracks data: merge, add features, clean columns.
    
    Extracted from hitmusiclyricnet/popularity_prediction.py data_preprocessing method.
    
    Args:
        df_tracks: Raw tracks DataFrame
        df_artists: Artists DataFrame
        df_albums: Albums DataFrame
        df_audio_features: Audio features DataFrame
        task: 'regression' or 'classification'
    
    Returns:
        Preprocessed DataFrame ready for feature engineering
    """
    print('Pre-processing data for {} problem ...'.format(task))
    
    # Rename id column to be consistent
    key_id = 'track_id'
    df_tracks = df_tracks.rename(columns={'id': key_id})
    
    # Add artist and market features
    if df_artists is not None:
        print('Applying Market transformation ...')
        df_tracks['n_markets'] = df_tracks.apply(get_total_markets, axis=1)
        
        print('Applying Artists followers transformation ...')
        followers_data = df_tracks.apply(
            lambda x: get_artist_data(x, df_artists, 'followers'), axis=1
        )
        df_tracks['artist_count'] = followers_data.apply(lambda x: x[0])
        df_tracks['artist_followers_max'] = followers_data.apply(lambda x: x[1])
        df_tracks['artist_followers_min'] = followers_data.apply(lambda x: x[2])
        df_tracks['artist_followers_median'] = followers_data.apply(lambda x: x[3])
        
        print('Applying Artists popularity transformation ...')
        popularity_data = df_tracks.apply(
            lambda x: get_artist_data(x, df_artists, 'artist_popularity'), axis=1
        )
        df_tracks['artist_popularity_max'] = popularity_data.apply(lambda x: x[1])
        df_tracks['artist_popularity_min'] = popularity_data.apply(lambda x: x[2])
        df_tracks['artist_popularity_median'] = popularity_data.apply(lambda x: x[3])
    
    # Merge DataFrames
    df_join = [df_tracks, df_audio_features]
    df_final = reduce(lambda left, right: pd.merge(left, right, on=key_id), df_join)
    
    print(f"Size of df_final after merge: {df_final.shape}")
    
    # Drop non-numeric/irrelevant columns
    drop_cols = [
        'album_id', 'analysis_url', 'artists_id', 'available_markets', 'country',
        'disc_number', 'lyrics', 'name', 'playlist', 'preview_url',
        'track_href', 'track_name_prev', 'track_number', 'uri', 'time_signature',
        'href', 'track_id', 'type'
    ]
    
    if task == 'classification':
        # Add classification target
        def get_popularity_class(row, scale=1):
            if row['popularity'] < 29 * scale:
                return 0
            elif row['popularity'] >= 29 * scale and row['popularity'] < 52 * scale:
                return 1
            else:
                return 2
        
        df_final['popularity_class'] = df_final.apply(get_popularity_class, scale=1, axis=1)
        drop_cols += ['popularity']
    
    check_cols = [col for col in drop_cols if col in list(df_final.columns)]
    df_final.drop(check_cols, axis=1, inplace=True)
    
    return df_final


def run_cleaning() -> None:
    """Main entry point for data cleaning stage."""
    print("Saregama – Stage 1: Data cleaning and preprocessing")
    
    # Check that all required files exist
    required_files = [
        ("artists", data_paths.artists_csv),
        ("tracks (raw)", data_paths.tracks_raw_csv),
        ("tracks (cleaned EN)", data_paths.tracks_clean_en_csv),
        ("tracks (cleaned other)", data_paths.tracks_clean_other_csv),
        ("albums", data_paths.albums_csv),
        ("audio features", data_paths.audio_lowlevel_csv),
        ("lyrics features", data_paths.lyrics_feature_csv),
    ]
    
    print("\nChecking required files:")
    for name, path in required_files:
        status = "✓" if path.exists() else "✗ MISSING"
        print(f"  {name:25} {status:15} {path}")
    
    # Load and show basic stats
    try:
        df_tracks, df_artists, df_albums, df_audio_features, df_lyrics_features = load_raw_data()
        print("\n✓ Data loading completed successfully")
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        raise


if __name__ == "__main__":
    run_cleaning()
