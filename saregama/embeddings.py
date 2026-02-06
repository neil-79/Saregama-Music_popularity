"""Stage 2: Lyrics embeddings generation and compression.

This module contains clean implementations of embedding generation and
compression logic extracted from the original messy code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import data_paths, OPENAI_API_KEY
from .data_processing import clean_lyrics

# Try to import OpenAI library (optional - only needed for direct API)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# OpenAI Embeddings
# ============================================================================

def create_openai_batch_jsonl(
    csv_path: Path | None = None,
    output_path: Path | None = None
) -> Path:
    """Create JSONL file for OpenAI Batch embeddings API.
    
    Extracted from embeddings_extraction/openai_embeddings.py
    
    Args:
        csv_path: Path to tracks CSV with lyrics. If None, uses config default.
        output_path: Path to output JSONL file. If None, uses embeddings_extraction folder.
    
    Returns:
        Path to created JSONL file
    """
    if csv_path is None:
        # Try cleaned otherlang first, fallback to raw
        if data_paths.tracks_clean_other_csv.exists():
            csv_path = data_paths.tracks_clean_other_csv
        elif data_paths.tracks_raw_csv.exists():
            csv_path = data_paths.tracks_raw_csv
        else:
            raise FileNotFoundError("No tracks CSV found")
    
    if output_path is None:
        output_dir = Path(__file__).parent.parent / "embeddings_extraction"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "lyrics_embeddings_batch.jsonl"
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Find track ID column
    track_id_column = 'id' if 'id' in df.columns else None
    if track_id_column is None:
        print("Warning: No 'id' column found. Using row index as custom_id.")
    
    if 'lyrics' not in df.columns:
        raise ValueError(f"'lyrics' column not found. Available: {df.columns.tolist()}")
    
    print(f"Total records: {len(df)}")
    
    # Create JSONL entries
    with open(output_path, 'w', encoding='utf-8') as f:
        count = 0
        for idx, row in df.iterrows():
            lyrics = clean_lyrics(row['lyrics'])
            
            if lyrics:
                custom_id = str(row[track_id_column]) if track_id_column else f"row-{idx}"
                
                json_obj = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "text-embedding-3-large",
                        "input": lyrics,
                        "encoding_format": "float"
                    }
                }
                
                f.write(json.dumps(json_obj) + '\n')
                count += 1
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1} records")
    
    print(f"JSONL file created at {output_path} with {count} records")
    return output_path


def create_openai_embeddings_direct(
    csv_path: Path | None = None,
    output_path: Path | None = None,
    api_key: str | None = None,
    batch_size: int = 100
) -> Path:
    """Create embeddings using OpenAI API directly (requires API key).
    
    This is an alternative to the Batch API - it calls OpenAI API directly.
    
    Args:
        csv_path: Path to tracks CSV with lyrics. If None, uses config default.
        output_path: Path to output CSV. If None, uses config default.
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        batch_size: Number of lyrics to process in each API call.
    
    Returns:
        Path to created CSV file
    """
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "OpenAI library not installed. Install with: pip install openai\n"
            "Alternatively, use --create-jsonl for Batch API (manual upload)."
        )
    
    if api_key is None:
        api_key = OPENAI_API_KEY
    
    if not api_key or api_key == "paste-your-openai-api-key-here":
        raise ValueError(
            "OpenAI API key not set!\n"
            "Edit saregama/config.py and set OPENAI_API_KEY = 'your-actual-key-here'"
        )
    
    if csv_path is None:
        if data_paths.tracks_clean_other_csv.exists():
            csv_path = data_paths.tracks_clean_other_csv
        elif data_paths.tracks_raw_csv.exists():
            csv_path = data_paths.tracks_raw_csv
        else:
            raise FileNotFoundError("No tracks CSV found")
    
    if output_path is None:
        data_paths.embeddings_2025.mkdir(parents=True, exist_ok=True)
        output_path = data_paths.embeddings_2025 / "openai_large_embeddings_otherlang_228877.csv"
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if 'lyrics' not in df.columns:
        raise ValueError(f"'lyrics' column not found. Available: {df.columns.tolist()}")
    
    track_id_column = 'id' if 'id' in df.columns else None
    print(f"Total records: {len(df)}")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Process lyrics in batches
    embeddings_list = []
    track_ids = []
    lyrics_list = []
    
    lyrics_batch = []
    ids_batch = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing lyrics"):
        lyrics = clean_lyrics(row['lyrics'])
        
        if lyrics:
            track_id = str(row[track_id_column]) if track_id_column else f"row-{idx}"
            lyrics_batch.append(lyrics)
            ids_batch.append(track_id)
            
            # Process batch when it reaches batch_size
            if len(lyrics_batch) >= batch_size:
                try:
                    response = client.embeddings.create(
                        model="text-embedding-3-large",
                        input=lyrics_batch
                    )
                    
                    for i, embedding_obj in enumerate(response.data):
                        embeddings_list.append(embedding_obj.embedding)
                        track_ids.append(ids_batch[i])
                        lyrics_list.append(lyrics_batch[i])
                    
                    lyrics_batch = []
                    ids_batch = []
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    lyrics_batch = []
                    ids_batch = []
    
    # Process remaining items
    if lyrics_batch:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=lyrics_batch
            )
            
            for i, embedding_obj in enumerate(response.data):
                embeddings_list.append(embedding_obj.embedding)
                track_ids.append(ids_batch[i])
                lyrics_list.append(lyrics_batch[i])
        except Exception as e:
            print(f"Error processing final batch: {e}")
    
    # Create DataFrame
    embeddings_array = np.array(embeddings_list)
    embedding_dim = embeddings_array.shape[1]
    columns = [f'embedding_{i}' for i in range(embedding_dim)]
    
    embeddings_df = pd.DataFrame(embeddings_array, columns=columns)
    embeddings_df['id'] = track_ids
    embeddings_df['lyrics'] = lyrics_list
    
    # Reorder columns
    cols = ['id', 'lyrics'] + columns
    embeddings_df = embeddings_df[cols]
    
    embeddings_df.to_csv(output_path, index=False)
    print(f"Embeddings saved to {output_path} with shape: {embeddings_df.shape}")
    
    return output_path


def convert_openai_batch_to_csv(
    jsonl_path: Path,
    output_path: Path | None = None
) -> Path:
    """Convert OpenAI Batch API output JSONL to embeddings CSV.
    
    Extracted from embeddings_extraction/openai_embeddings_to_csv.py
    
    Args:
        jsonl_path: Path to OpenAI Batch output JSONL file
        output_path: Path to output CSV. If None, uses config default.
    
    Returns:
        Path to created CSV file
    """
    if output_path is None:
        data_paths.embeddings_2025.mkdir(parents=True, exist_ok=True)
        output_path = data_paths.embeddings_2025 / "openai_large_embeddings_otherlang_228877.csv"
    
    print(f"Reading JSONL file: {jsonl_path}")
    
    track_ids = []
    embeddings_list = []
    lyrics_list = []
    
    with open(jsonl_path, 'r') as file:
        for line in tqdm(file):
            data = json.loads(line)
            
            track_id = data['custom_id']
            embedding = data['response']['body']['data'][0]['embedding']
            lyrics = data['response']['body']['data'][0].get('text', "")
            
            track_ids.append(track_id)
            embeddings_list.append(embedding)
            lyrics_list.append(lyrics)
    
    embeddings_array = np.array(embeddings_list)
    embedding_dim = embeddings_array.shape[1]
    columns = [f'embedding_{i}' for i in range(embedding_dim)]
    
    embeddings_df = pd.DataFrame(embeddings_array, columns=columns)
    embeddings_df['id'] = track_ids
    embeddings_df['lyrics'] = lyrics_list
    
    # Reorder columns
    cols = ['id', 'lyrics'] + columns
    embeddings_df = embeddings_df[cols]
    
    embeddings_df.to_csv(output_path, index=False)
    print(f"Embeddings saved to {output_path} with shape: {embeddings_df.shape}")
    
    return output_path


# ============================================================================
# Autoencoder for Compression (from lyrics_Improved_AE_2025.py)
# ============================================================================

class ResBlock(nn.Module):
    """Residual block for autoencoder."""
    
    def __init__(self, dim: int, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        return x + self.net(x)


class ResidualAE(nn.Module):
    """Residual Autoencoder for compressing embeddings."""
    
    def __init__(self, in_dim: int, hid_fracs: tuple = (0.5, 0.25, 0.125), 
                 bott_frac: float = 1/12, drop: float = 0.05):
        super().__init__()
        self.hid_fracs = hid_fracs
        self.bott_frac = bott_frac
        h_dims = [int(in_dim * f) for f in hid_fracs]
        bott = int(in_dim * bott_frac)
        
        # Encoder
        enc = []
        prev = in_dim
        for h in h_dims:
            enc += [nn.Linear(prev, h), ResBlock(h, drop)]
            prev = h
        enc += [nn.LayerNorm(prev), nn.Linear(prev, bott)]
        self.encoder = nn.Sequential(*enc)
        
        # Decoder
        dec = []
        prev = bott
        for h in reversed(h_dims):
            dec += [nn.Linear(prev, h), ResBlock(h, drop)]
            prev = h
        dec += [nn.Linear(prev, in_dim)]
        self.decoder = nn.Sequential(*dec)
        
        self.sigma_noise = 0.03
    
    def forward(self, x):
        noisy = x + torch.randn_like(x) * self.sigma_noise
        z = self.encoder(noisy)
        recon = self.decoder(z)
        return z, recon


def cosine_recon_loss(x, recon):
    """Cosine reconstruction loss."""
    x = F.normalize(x, dim=-1)
    r = F.normalize(recon, dim=-1)
    return 1 - (x * r).sum(dim=-1).mean()


def info_nce_loss(z1, z2, tau: float = 0.07):
    """InfoNCE contrastive loss."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def train_lyrics_autoencoder(
    X_np: np.ndarray,
    in_dim: int,
    hid_fracs: tuple = (0.5, 0.25, 0.125),
    bott_frac: float = 1/12,
    beta: float = 0.2,
    gamma_mse: float = 1.0,
    tau: float = 0.05,
    epochs: int = 100,
    batch: int = 256,
    patience: int = 8
) -> np.ndarray:
    """Train residual autoencoder on embeddings.
    
    Extracted from miscellaneous/lyrics_Improved_AE_2025.py
    """
    print(f"Training autoencoder: input_dim={in_dim}, bottleneck={int(in_dim * bott_frac)}")
    print(f"Data variance: {np.var(X_np):.6f}, min={np.min(X_np):.6f}, max={np.max(X_np):.6f}")
    
    ds = TensorDataset(torch.from_numpy(X_np))
    val_len = int(0.25 * len(ds))
    train_ds, val_ds = random_split(
        ds, [len(ds) - val_len, val_len],
        generator=torch.Generator().manual_seed(76)
    )
    
    dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=True)
    v_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, drop_last=False,
                      num_workers=4, pin_memory=True)
    
    model = ResidualAE(in_dim, hid_fracs=hid_fracs, bott_frac=bott_frac).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, fused=True)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = GradScaler()
    
    best_val, wait = float('inf'), 0
    best_state = None
    
    for ep in range(1, epochs + 1):
        # Training
        model.train()
        tot, rec_tot, nce_tot, mse_tot = 0, 0, 0, 0
        for (x,) in dl:
            x = F.normalize(x.to(DEVICE, dtype=torch.float32), dim=-1)
            
            with autocast(device_type='cuda' if DEVICE == 'cuda' else 'cpu'):
                z1, recon = model(x)
                z2, _ = model(x)  # second noisy pass
                rec = cosine_recon_loss(x, recon)
                nce = info_nce_loss(z1, z2, tau=tau)
                mse = F.mse_loss(recon, x)
                loss = rec + gamma_mse * mse + beta * nce
            
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            tot += loss.item() * x.size(0)
            rec_tot += rec.item() * x.size(0)
            nce_tot += nce.item() * x.size(0)
            mse_tot += mse.item() * x.size(0)
        
        train_loss = tot / len(ds)
        
        # Validation
        model.eval()
        v_tot = 0
        with torch.no_grad():
            for (x,) in v_dl:
                x = F.normalize(x.to(DEVICE, dtype=torch.float32), dim=-1)
                z1, recon = model(x)
                z2, _ = model(x)
                rec = cosine_recon_loss(x, recon)
                nce = info_nce_loss(z1, z2, tau=tau)
                mse = F.mse_loss(recon, x)
                v_tot += (rec + gamma_mse * mse + beta * nce).item() * x.size(0)
        
        val_loss = v_tot / len(ds)
        scheduler.step(val_loss)
        
        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:03d}: train={train_loss:.6f}, val={val_loss:.6f}")
        
        if val_loss + 1e-4 < best_val:
            best_val, wait = val_loss, 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break
    
    model.load_state_dict(best_state)
    model.eval()
    
    # Extract compressed embeddings
    Z = []
    with torch.no_grad():
        for (x,) in DataLoader(ds, batch_size=1024, num_workers=4, pin_memory=True):
            z, _ = model(F.normalize(x.to(DEVICE, dtype=torch.float32), dim=-1))
            Z.append(z.cpu())
    
    return torch.cat(Z).numpy()


def compress_lyrics_embeddings(
    csv1_path: Path,
    csv2_path: Path | None = None,
    output_path: Path | None = None,
    **ae_kwargs
) -> Path:
    """Compress lyrics embeddings using residual autoencoder.
    
    Args:
        csv1_path: Path to first embeddings CSV
        csv2_path: Path to second embeddings CSV (optional, will use csv1 if None)
        output_path: Path to output compressed CSV
        **ae_kwargs: Additional arguments for train_lyrics_autoencoder
    
    Returns:
        Path to created compressed CSV
    """
    if output_path is None:
        data_paths.embeddings_2025.mkdir(parents=True, exist_ok=True)
        output_path = data_paths.lyrics_compressed_csv
    
    # Load embeddings
    print(f"Loading embeddings from {csv1_path}")
    df1 = pd.read_csv(csv1_path)
    
    if csv2_path and csv2_path.exists():
        print(f"Loading embeddings from {csv2_path}")
        df2 = pd.read_csv(csv2_path)
        merged_df = pd.concat([df1, df2], ignore_index=True)
    else:
        merged_df = df1
    
    # Shuffle
    merged_df = merged_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Extract track_ids
    track_ids = merged_df.pop("id") if "id" in merged_df.columns else None
    
    # Extract numeric embedding columns
    X = merged_df.select_dtypes(include=[np.number]).to_numpy(np.float32)
    print(f"Embedding matrix shape: {X.shape}")
    print(f"Null values: {np.isnan(X).sum()}")
    
    # Train autoencoder
    embedding_dim = X.shape[1]
    Z = train_lyrics_autoencoder(X, embedding_dim, **ae_kwargs)
    
    # Create output DataFrame
    comp_cols = [f"cmp_{i}" for i in range(Z.shape[1])]
    out_df = pd.DataFrame(Z, columns=comp_cols)
    if track_ids is not None:
        out_df["track_id"] = track_ids.values
    
    out_df.to_csv(output_path, index=False)
    print(f"Compressed embeddings saved to: {output_path}")
    
    return output_path


def run_embeddings_pipeline() -> None:
    """Main entry point for embeddings stage."""
    print("Saregama – Stage 2: Embeddings generation and compression")
    
    # Step 1: Create OpenAI batch JSONL
    print("\n=== Step 1: Create OpenAI Batch JSONL ===")
    jsonl_path = create_openai_batch_jsonl()
    print(f"\n✓ JSONL file created: {jsonl_path}")
    print("  → Upload this file to OpenAI Batch API")
    print("  → Wait for completion and download the output JSONL")
    print("  → Then run: python -m saregama.embeddings --convert-jsonl <path_to_output.jsonl>")
    
    # Note: Steps 2 and 3 require manual OpenAI API interaction
    print("\n=== Next steps (manual) ===")
    print("1. Upload JSONL to OpenAI Batch API")
    print("2. Download output JSONL")
    print("3. Run: convert_openai_batch_to_csv(<output_jsonl_path>)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-jsonl", action="store_true", help="Create OpenAI batch JSONL (for manual upload)")
    parser.add_argument("--direct-api", action="store_true", help="Use OpenAI API directly (API key must be in config.py)")
    parser.add_argument("--convert-jsonl", type=str, help="Convert OpenAI output JSONL to CSV")
    parser.add_argument("--compress", action="store_true", help="Compress embeddings using AE")
    parser.add_argument("--csv1", type=str, help="First embeddings CSV for compression")
    parser.add_argument("--csv2", type=str, help="Second embeddings CSV for compression (optional)")
    
    args = parser.parse_args()
    
    if args.create_jsonl:
        create_openai_batch_jsonl()
    elif args.direct_api:
        create_openai_embeddings_direct()
    elif args.convert_jsonl:
        convert_openai_batch_to_csv(Path(args.convert_jsonl))
    elif args.compress:
        # Use the exact file location from config where we save embeddings
        csv1 = data_paths.openai_embeddings_other_csv
        csv2 = Path(args.csv2) if args.csv2 else None
        compress_lyrics_embeddings(csv1, csv2)
    else:
        run_embeddings_pipeline()
