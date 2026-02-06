"""Central configuration for the Saregama pipeline.

All filesystem paths and high-level experiment choices are collected here
so that the rest of the code does not contain hard-coded machine-specific paths.
"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DataPaths:
    """All data paths, now pointing to the local ./data folder."""

    # Base data folder in this repo: ./data
    base_data: Path = PROJECT_ROOT / "data"

    # Core CSVs (matching your actual data structure)
    # Raw CSVs are under SpotGenTrack/Data Sources
    artists_csv: Path = base_data / "SpotGenTrack" / "Data Sources" / "spotify_artists.csv"
    tracks_raw_csv: Path = base_data / "SpotGenTrack" / "Data Sources" / "spotify_tracks.csv"
    # For now we only have a single spotify_tracks.csv; cleaned/en/other variants
    # are produced/used by the original code inside hitmusiclyricnet.
    tracks_clean_en_csv: Path = tracks_raw_csv  # placeholder alias
    tracks_clean_other_csv: Path = tracks_raw_csv  # placeholder alias
    albums_csv: Path = base_data / "SpotGenTrack" / "Data Sources" / "spotify_albums.csv"

    # Feature CSVs are at root level under Features Extracted
    features_root: Path = base_data / "Features Extracted"
    audio_lowlevel_csv: Path = features_root / "low_level_audio_features.csv"
    lyrics_feature_csv: Path = features_root / "lyrics_features.csv"

    # Embeddings & compressed representations (still using original naming)
    embeddings_root: Path = base_data / "SpotGenTrack" / "Embeddings"
    embeddings_2025: Path = base_data / "SpotGenTrack" / "Embeddings_2025"
    openai_embeddings_en_csv: Path = embeddings_root / "openai_embeddings_large.csv"
    openai_embeddings_other_csv: Path = embeddings_2025 / "openai_large_embeddings_otherlang_228877.csv"
    lyrics_compressed_csv: Path = embeddings_2025 / "LIAE_compressed_openai_large_2025.csv"

    # HitMusicLyricNet preprocessed / compressed feature files
    hitmusic_input_root: Path = PROJECT_ROOT / "hitmusiclyricnet" / "hitmusiclyricnet" / "input"
    # Note: The actual filename includes the compression method, so we use a pattern
    # The preprocessing will create: AE_compressed_auto_LLaudio_spotgentrack_lyricscleaned_2025.csv
    ae_compressed_audio_csv: Path = hitmusic_input_root / "AE_compressed_auto_LLaudio_spotgentrack_lyricscleaned_2025.csv"
    hl_audio_metadata_csv: Path = hitmusic_input_root / "HL_audio_metadata_features_spotgentrack_lyricscleaned_2025_regression.csv"
    tracks_id_csv: Path = hitmusic_input_root / "tracks_id_regression_spotgentrack_lyricscleaned_2025.csv"


@dataclass
class ModelConfig:
    # Mirrors the best-performing configuration from the original main.py
    model_name: str = "model_1_A"
    model_dir: str = "saved_models"
    model_subdir: str = "ARR_ACL_SpotGenTrack_lyricscleaned_openai_large_LIAE_2025"
    input_dim: int = 97  # This will be overwritten with the actual dimension at runtime
    output_dim: int = 1
    optimizer: str = "adam"
    metrics: tuple = ("mean_absolute_error",)
    loss: str = "mse"
    early_stop: bool = True
    weights: str = "weights.weights.h5"
    plot_loss: bool = True
    neurons: dict = None
    n_layers: int = 4
    weights_init: str = "glorot_uniform"
    dropout: float = 0.25
    epochs: int = 100
    batch_size: int = 256
    early_mon: str = "val_mean_absolute_error"
    mode: str = "min"
    checkout_mon: str = "val_loss"

    def to_model_args(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_dir": self.model_dir,
            "model_subDir": self.model_subdir,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "optimizer": self.optimizer,
            "metrics": list(self.metrics),
            "loss": self.loss,
            "earlyStop": self.early_stop,
            "weights": self.weights,
            "plot_loss": self.plot_loss,
            "neurons": self.neurons or {"alpha": 1, "beta": 0.5, "gamma": 0.25},
            "n_layers": self.n_layers,
            "weights_init": self.weights_init,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "early_mon": self.early_mon,
            "mode": self.mode,
            "checkout_mon": self.checkout_mon,
        }


# ============================================================================
# OpenAI API Key - PASTE YOUR KEY HERE
# ============================================================================
# Get your API key from: https://platform.openai.com/api-keys







# OPENAI_API_KEY = 


data_paths = DataPaths()
model_config = ModelConfig()

