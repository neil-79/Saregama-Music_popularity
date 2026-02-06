# Saregama: Clean Pipeline for Music Popularity Prediction

This folder contains **clean, refactored implementations** extracted from the messy reference code in other folders. All the messy code remains untouched as reference.

## Structure

```
saregama/
├── __init__.py          # Package initialization
├── config.py            # Centralized paths and model configuration
├── data_processing.py    # Data cleaning and preprocessing functions
├── embeddings.py        # Lyrics embeddings generation and compression
├── train.py             # Model training pipeline
├── inference.py         # Model inference and evaluation
├── cleaning.py          # Entry point for Stage 1
└── README.md            # This file
```

## Usage

### Stage 1: Data Cleaning
```bash
cd /Users/indraneelghosh/Desktop/yash
python -m saregama.cleaning
```
Checks that all required data files exist and prints basic statistics.

### Stage 2: Embeddings
```bash
python -m saregama.embeddings --create-jsonl
```
Creates JSONL file for OpenAI Batch API.

After uploading to OpenAI and downloading the output:
```bash
python -m saregama.embeddings --convert-jsonl <path_to_output.jsonl>
```

Then compress embeddings:
```bash
python -m saregama.embeddings --compress --csv1 <path1> --csv2 <path2>
```

### Stage 3: Training
```bash
python -m saregama.train
```
Trains the DNN model with k-fold cross-validation and evaluates on test set.

### Stage 4: Inference
```bash
python -m saregama.inference
```
Loads trained model and evaluates on saved test set.

## Full Pipeline

```bash
# Complete pipeline from data check to training
python -m saregama.cleaning && \
python -m saregama.embeddings --create-jsonl && \
# ... (manual OpenAI API step) ... && \
python -m saregama.embeddings --convert-jsonl <output.jsonl> && \
python -m saregama.embeddings --compress && \
python -m saregama.train
```

## Configuration

All paths and model settings are centralized in `saregama/config.py`. Edit `DataPaths` and `ModelConfig` to match your setup.

## Note

The original messy code in `hitmusiclyricnet/`, `embeddings_extraction/`, and `miscellaneous/` folders remains **untouched** and serves as reference. This `saregama/` folder contains clean, modular implementations extracted from that code.
