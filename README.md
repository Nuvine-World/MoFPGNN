<h1 align="center">MoFPGNN: Molecular Fingerprint-Enhanced Graph Neural Networks for LNP Transfection Efficiency Prediction</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/></a>
</p>

<p align="justify">
<strong>MoFPGNN</strong> extends the <a href="https://github.com/AsalMehradfar/LANTERN">LANTERN</a> framework with a hybrid pipeline that fuses <strong>KPGT</strong> (Knowledge-Guided Pre-Training) pretrained molecular embeddings with <strong>2048-bit count-based Morgan fingerprints</strong> via late fusion for predicting the mRNA transfection potency (mTP) of ionizable lipids used in lipid nanoparticle (LNP) delivery systems.
</p>

<p align="justify">
The core idea is that pretrained KPGT graph-level embeddings and fixed-radius substructure counts (captured by Morgan fingerprints) encode complementary chemical information. By concatenating these representations and training an MLP regression head, the hybrid model is expected to outperform either representation alone.
</p>

### Key Features

- **Pretrained KPGT embeddings** (2304-dim) extracted from the official pretrained KPGT model
- **2048-bit count-based Morgan fingerprints** (radius 2) extracted with RDKit
- **Late-fusion hybrid architecture**: `MLP([kpgt_emb || f_morgan]) -> mTP`
- **Ablation baselines**: pure KPGT pretrained regressor, Morgan-only MLP, plus the original LANTERN models
- **Extended evaluation**: R2, RMSE, MAE, Pearson r, top-k% ranking recovery
- **Random and Murcko scaffold splits** with configurable train/val/test ratios

## Table of Contents

  * [Architecture](#architecture)
  * [Project Structure](#project-structure)
  * [Environment Setup](#environment-setup)
  * [Quick Start](#quick-start)
  * [Detailed Usage](#detailed-usage)
  * [Configuration Reference](#configuration-reference)
  * [Original LANTERN Pipeline](#original-lantern-pipeline)
  * [Citation](#citation)

## Architecture

### Step 1 — Embedding Extraction (one-time, offline)

```
  Molecular SMILES
        |
        +-----> [Official Pretrained KPGT Model] -----> kpgt_emb  (2304-dim, stored to disk)
        |
        +-----> [RDKit Morgan FP, radius=2, 2048-bit] -> morgan_fp (2048-dim, stored to disk)
```

No KPGT encoder is trained from scratch. Embeddings are extracted once using the official
pretrained weights and reused across all experiments.

---

### Step 2A — KPGT Pretrained Regressor (Option A)

```
    kpgt_emb (2304)
          |
          v
    +-------------+
    |  MLP Head   |
    | 512 -> 256  |
    |    -> 1     |
    +-------------+
          |
          v
    Predicted mTP
```

---

### Step 2B — MoFPGNN: Pretrained KPGT + Morgan Hybrid (Option B, core model)

```
    kpgt_emb (2304)      morgan_fp (2048)
          \                    /
           \                  /
            +--Concatenation--+
                    |
                    v
           [kpgt_emb || morgan_fp]
               (4352-dim)
                    |
                    v
            +-------------+
            |  MLP Head   |
            | 512 -> 256  |
            |    -> 1     |
            +-------------+
                    |
                    v
            Predicted mTP
```

## Project Structure

```
.
├── config/
│   ├── kpgt_pretrained_regressor_config.yaml      # KPGT pretrained embeddings + regressor
│   └── kpgt_morgan_pretrained_hybrid_config.yaml  # KPGT pretrained + Morgan (MoFPGNN core)
│
├── models/
│   ├── kpgt_encoder.py         PretrainedEmbeddingRegressor
│   └── hybrid_model.py         PretrainedKPGTMorganHybrid + MorganOnlyMLP
│
├── pipeline/
│   ├── result_saver.py        # Result saver
│   ├── kpgt_trainer.py        # Training loop
│   └── kpgt_pipeline.py       # Pipeline orchestrator
│
├── scripts/
│   ├── extract_kpgt_embeddings.py    # Extract embeddings from pretrained KPGT
│   ├── extract_morgan_fingerprint.py # 2048-bit Morgan FP extraction
│   └── run_kpgt_pipeline.py          # Pipeline entry point
│
├── utils/
│   ├── io_tools.py            # Pickle/YAML I/O
│   ├── visual_utils.py        # Plotting utilities
│   ├── utils.py               # Seed setting
│   ├── metrics.py             # Extended evaluation metrics
│   └── confusion_matrix_utils.py  # Confusion matrix helpers
│
├── data/
│   ├── AGILE.csv              # Dataset (1100 molecules)
│   ├── fingerprints/AGILE/    # Pre-computed fingerprints & embeddings
│   └── splits/AGILE/          # Pre-computed data splits
│
├── KPGT/                      # Cloned KPGT repo (for embedding extraction)
├── checkpoints/               # Saved model weights
├── results/                   # Experiment outputs
├── mofpgnn.yml                # Conda environment file
└── README.md
```

## Environment Setup

The project uses [Conda](https://docs.conda.io/en/latest/) for dependency management. All required packages (PyTorch, PyTorch Geometric, RDKit, DeepChem, etc.) are specified in `mofpgnn.yml`.

```bash
# Create the environment
conda env create -f mofpgnn.yml

# Activate
conda activate mofpgnn
```

To update after changes:

```bash
conda env update -f mofpgnn.yml --prune
```
## Pre-requisite

### Create KPGT embeddings from pre-trained kpgt model

```bash
# 1. Clone KPGT (from your mofpgnn project root)

git clone git@github.com:lihan97/KPGT.git

# 2. Download pretrained weights from:
#    https://figshare.com/s/d488f30c23946cf6898f
# Unzip and place at:
#    kpgt/models/pretrained/base/base.pth

# 3. Set up the KPGT environment
cd kpgt
conda env create -f environment.yml
conda activate KPGT
pip install dgllife descriptastorus
cd ..

# 4. Extract embeddings (from mofpgnn root)
python scripts/extract_kpgt_embeddings.py \
    --kpgt_dir ./kpgt \
    --checkpoint ./kpgt/models/pretrained/base/base.pth \
    --data_name AGILE
```

## Quick Start

### Option A - KPGT with pretrained embeddings

Uses the official pretrained KPGT model to extract fixed 2304-dim embeddings, then trains a lightweight regressor. This matches the LANTERN paper methodology.

```bash
# 1. Extract KPGT embeddings (one-time, requires KPGT env — see Pre-requisite)
conda activate KPGT
python scripts/extract_kpgt_embeddings.py \
    --kpgt_dir ./KPGT \
    --checkpoint ./KPGT/models/pretrained/base/base.pth \
    --data_name AGILE
conda activate mofpgnn

# 2. Train the pretrained-embedding regressor
python scripts/run_kpgt_pipeline.py --config config/kpgt_pretrained_regressor_config.yaml
```

### Option B - KPGT + Morgan pretrained hybrid (MoFPGNN core model)

Fuses pretrained KPGT embeddings (2304-dim) with Morgan fingerprints (2048-dim) via late fusion for a total of 4352-dim input to the MLP head.

```bash
# 1. Extract KPGT embeddings (if not already done — see Option A step 1)

# 2. Extract Morgan fingerprints (one-time)
python scripts/extract_morgan_fingerprint.py

# 3. Train the pretrained hybrid model
python scripts/run_kpgt_pipeline.py --config config/kpgt_morgan_pretrained_hybrid_config.yaml
```

## Detailed Usage

### 1. Extract KPGT Pretrained Embeddings

See [Pre-requisite](#pre-requisite) for full setup. This produces `data/fingerprints/AGILE/kpgt_embeddings.pkl`.

### 2. Extract Morgan Fingerprints

Generates 2048-bit count-based Morgan fingerprints (radius 2) for all molecules:

```bash
python scripts/extract_morgan_fingerprint.py \
    --data_name AGILE \
    --save_path data/fingerprints/AGILE \
    --radius 2 \
    --n_bits 2048
```

Output: `data/fingerprints/AGILE/morgan.pkl`

### 3. Run Individual Experiments

Each experiment is controlled by a YAML config file:

```bash
# KPGT pretrained embeddings + regressor
python scripts/run_kpgt_pipeline.py --config config/kpgt_pretrained_regressor_config.yaml

# KPGT pretrained + Morgan hybrid (MoFPGNN core model)
python scripts/run_kpgt_pipeline.py --config config/kpgt_morgan_pretrained_hybrid_config.yaml
```

Override split type from the command line:

```bash
# Use Murcko scaffold split instead of random
python scripts/run_kpgt_pipeline.py \
    --config config/kpgt_morgan_pretrained_hybrid_config.yaml \
    --split Murcko_scaffold
```

### 4. Outputs

Each experiment saves to `results/<pipeline_name>/`:

| File | Contents |
|------|----------|
| `metrics.txt` | R2, RMSE, MAE, Pearson r, top-k% recovery, relative error stats |
| `train_results.csv` | True vs predicted values (train split) |
| `val_results.csv` | True vs predicted values (validation split) |
| `test_results.csv` | True vs predicted values (test split) |
| `loss_curve.png` | Training and validation loss over epochs |
| `prediction_vs_true_test.png` | Scatter plot of predictions vs ground truth |

Model checkpoints are saved to `checkpoints/<pipeline_name>.pth`.

## Configuration Reference

### Model Types

| `model_type` | Description | Required data |
|---|---|---|
| `kpgt_pretrained_regressor` | Pretrained KPGT embeddings + MLP regressor | `kpgt_embeddings.pkl` |
| `kpgt_morgan_pretrained_hybrid` | Pretrained KPGT + Morgan late fusion (MoFPGNN) | `kpgt_embeddings.pkl`, `morgan.pkl` |

### KPGT Pretrained Regressor Config

```yaml
model_type: kpgt_pretrained_regressor
dataset: AGILE
split: random                    # random | Murcko_scaffold
device: cuda                     # cuda | cpu

# Paths
csv_path: data/AGILE.csv
split_path: data/splits/AGILE/random.npy
embeddings_path: data/fingerprints/AGILE/kpgt_embeddings.pkl

# Training (per LANTERN paper: 20 epochs, hidden_dim 512)
epochs: 20
lr: 0.001
batch_size: 64
weight_decay: 0.00001
patience: 20
warmup_epochs: 3
grad_clip: 1.0
loss_fn: mse

# Regressor
embedding_dim: 2304
regressor:
  hidden_dim: 512
  dropout: 0.3
```

### KPGT + Morgan Pretrained Hybrid Config (MoFPGNN core)

```yaml
model_type: kpgt_morgan_pretrained_hybrid
dataset: AGILE
split: random                    # random | Murcko_scaffold
device: cuda                     # cuda | cpu

# Paths
csv_path: data/AGILE.csv
split_path: data/splits/AGILE/random.npy
embeddings_path: data/fingerprints/AGILE/kpgt_embeddings.pkl
morgan_path: data/fingerprints/AGILE/morgan.pkl

# Training
epochs: 100
lr: 0.001
batch_size: 64
weight_decay: 0.0001
patience: 30
warmup_epochs: 5
grad_clip: 1.0
loss_fn: huber

# MLP Head (input = 2304 + 2048 = 4352)
mlp_head:
  hidden_layers: [512, 256]
  dropout: 0.3

