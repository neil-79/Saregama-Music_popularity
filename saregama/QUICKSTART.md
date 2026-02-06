# Quick Start Guide

## Prerequisites

Make sure you're in the project root directory:
```bash
cd /Users/indraneelghosh/Desktop/yash
```

## Step-by-Step Execution

### Step 1: Verify Data Files
```bash
python -m saregama.cleaning
```
**What it does:** Checks that all required CSV files exist and prints data statistics.

**Expected output:** All files should show ✓ (checkmark).

---

### Step 2: Generate Lyrics Embeddings

**Choose ONE of these methods:**

#### Option A: Direct API (Recommended - Requires API Key)

**2a. Set your OpenAI API Key**

Edit the file: `saregama/config.py`

Find this line:
```python
OPENAI_API_KEY = "paste-your-openai-api-key-here"
```

Replace it with your actual key:
```python
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxx"
```

**2b. Generate embeddings directly**
```bash
python -m saregama.embeddings --direct-api
```
**What it does:** Calls OpenAI API directly to generate embeddings.

**Output:** `data/SpotGenTrack/Embeddings_2025/openai_large_embeddings_otherlang_228877.csv`

#### Option B: Batch API (Manual Upload - No API Key Needed)

**2a. Create OpenAI Batch JSONL File**
```bash
python -m saregama.embeddings --create-jsonl
```
**What it does:** Creates a JSONL file (`embeddings_extraction/lyrics_embeddings_batch.jsonl`) ready for OpenAI Batch API.

**Output:** `embeddings_extraction/lyrics_embeddings_batch.jsonl`

**2b. Upload to OpenAI (Manual Step)**
1. Go to https://platform.openai.com/batch
2. Upload the JSONL file created in step 2a
3. Wait for the batch job to complete
4. Download the output JSONL file

**2c. Convert OpenAI Output to CSV**
```bash
python -m saregama.embeddings --convert-jsonl <path_to_downloaded_output.jsonl>
```
**Example:**
```bash
python -m saregama.embeddings --convert-jsonl embeddings_extraction/batch_xxxxx_output.jsonl
```
**What it does:** Converts the OpenAI Batch API output into a CSV file with embeddings.

**Output:** `data/SpotGenTrack/Embeddings_2025/openai_large_embeddings_otherlang_228877.csv`

#### 2d. Compress Embeddings (Optional but Recommended)
```bash
python -m saregama.embeddings --compress
```
**What it does:** Trains a residual autoencoder to compress the high-dimensional embeddings into a smaller representation.

**Output:** `data/SpotGenTrack/Embeddings_2025/LIAE_compressed_openai_large_2025.csv`

**Note:** If you have two embedding CSVs (English + other languages), specify both:
```bash
python -m saregama.embeddings --compress --csv1 <path1> --csv2 <path2>
```

---

### Step 3: Preprocess Audio Features (REQUIRED)

**⚠️ CRITICAL:** Before training, you MUST run preprocessing to create compressed audio features.

```bash
python -m saregama.preprocessing
```

**What it does:** 
- Merges audio features with track metadata
- Applies feature engineering (artist followers, market counts)
- Compresses low-level audio features using an autoencoder
- Creates all intermediate files needed for training

**Output files:**
- `hitmusiclyricnet/hitmusiclyricnet/input/AE_compressed_auto_LLaudio_spotgentrack_lyricscleaned_2025.csv`
- `hitmusiclyricnet/hitmusiclyricnet/input/tracks_id_regression_spotgentrack_lyricscleaned_2025.csv`
- `hitmusiclyricnet/hitmusiclyricnet/input/HL_audio_metadata_features_spotgentrack_lyricscleaned_2025_regression.csv`

**⚠️ Troubleshooting:** If you get exit code 139 (segmentation fault), this is likely due to TensorFlow/Keras environment issues. Try:
- Using a different Python environment (conda/virtualenv)
- Updating TensorFlow/Keras versions
- Running with reduced memory/features
- Or manually running the original `hitmusiclyricnet` preprocessing code

---

### Step 4: Train the Model

```bash
python -m saregama.train
```
**What it does:**
- Loads and preprocesses all data
- Compresses audio/metadata features using autoencoder
- Merges compressed audio + compressed lyrics + metadata features
- Trains DNN model with 5-fold cross-validation
- Evaluates on test set and prints metrics

**Output:**
- Trained models saved in `hitmusiclyricnet/hitmusiclyricnet/saved_models/`
- Test set saved in `hitmusiclyricnet/hitmusiclyricnet/input/`
- Metrics printed to console

---

### Step 4: Inference/Testing

```bash
python -m saregama.inference
```
**What it does:** Loads the trained model and evaluates it on the saved test set.

**Output:** Prints evaluation metrics (MAE, RMSE, R², etc.)

---

## Complete Pipeline (One Command)

For a full run from data check to training:

```bash
# Stage 1: Check data
python -m saregama.cleaning && \

# Stage 2: Create embeddings JSONL
python -m saregama.embeddings --create-jsonl && \

# (Manual: Upload to OpenAI, wait, download output) && \

# Stage 2 continued: Convert and compress
python -m saregama.embeddings --convert-jsonl <output.jsonl> && \
python -m saregama.embeddings --compress && \

# Stage 3: Train
python -m saregama.train
```

---

## Troubleshooting

### "File not found" errors
- Check that all CSV files exist in `data/` folder
- Run `python -m saregama.cleaning` to verify paths

### "Module not found" errors
- Make sure you're in the project root: `/Users/indraneelghosh/Desktop/yash`
- Install required packages: `pip install pandas numpy scikit-learn torch keras`

### Training fails
- Make sure embeddings compression completed successfully
- Check that `data/SpotGenTrack/Embeddings_2025/LIAE_compressed_openai_large_2025.csv` exists

---

## What Each Stage Does

1. **cleaning** → Verifies data files exist and are readable
2. **embeddings** → Converts lyrics to vector embeddings and compresses them
3. **train** → Trains neural network to predict song popularity
4. **inference** → Tests the trained model on held-out data

---

## Configuration

All paths and settings are in `saregama/config.py`. You can modify:
- Data file paths
- Model hyperparameters (epochs, batch size, etc.)
- Output directories
