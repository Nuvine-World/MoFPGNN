<h1 align="center">MoFPGNN: Molecular Fingerprint-Enhanced Graph Neural Networks for LNP Transfection Efficiency Prediction</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/></a>
</p>

<p align="justify">
<strong>MoFPGNN</strong> extends the <a href="https://github.com/AsalMehradfar/LANTERN">LANTERN</a> framework with a hybrid pipeline that fuses <strong>KPGT</strong> (Knowledge-Guided Pre-Training) pretrained molecular embeddings with <strong>2048-bit Morgan fingerprints</strong> via late fusion for predicting the mRNA transfection potency (mTP) of ionizable lipids used in lipid nanoparticle (LNP) delivery systems.
</p>

<p align="justify">
The core idea is that pretrained KPGT graph-level embeddings and fixed-radius substructure counts (captured by Morgan fingerprints) encode complementary chemical information. By concatenating these representations and training an MLP regression head, the hybrid model is expected to outperform either representation alone. Two fingerprint variants are experimented: <strong>circular Morgan fingerprints (CMF)</strong>, which encode substructure counts, and <strong>binary Morgan fingerprints (BMF)</strong>, which encode substructure presence only.
</p>

### Key Features

- **Pretrained KPGT embeddings** (2304-dim) extracted from the official pretrained KPGT model
- **Two Morgan fingerprint variants**: circular count-based (CMF) and binary bit-vector (BMF / ECFP4), both radius 2, 2048-bit
- **Late-fusion hybrid architecture**: `MLP([kpgt_emb || f_morgan]) -> mTP`
- **Ablation baselines**: pure KPGT pretrained regressor, CMF-only MLP, BMF-only MLP
- **Evaluation**: R2, RMSE, MAE, Pearson r, top-k% ranking recovery
- **Random and Murcko scaffold splits** with configurable train/val/test ratios

## Table of Contents

  * [Architecture](#architecture)
  * [Project Structure](#project-structure)
  * [Environment Setup](#environment-setup)
  * [Quick Start](#quick-start)
  * [Detailed Usage](#detailed-usage)
  * [Configuration Reference](#configuration-reference)
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

### Step 2B — Morgan-Only MLP (Options B1 / B2, ablation baselines)

```
    morgan_fp (2048)              morgan_fp (2048)
    [CMF — count-based]           [BMF — binary / ECFP4]
          |                               |
          v                               v
    +-------------+               +-------------+
    |  MLP Head   |               |  MLP Head   |
    | 512 -> 256  |               | 512 -> 256  |
    |    -> 1     |               |    -> 1     |
    +-------------+               +-------------+
          |                               |
          v                               v
    Predicted mTP                  Predicted mTP
```

Two ablation baselines using Morgan fingerprints only (no graph information):
- **B1 — CMF-only**: count-based fingerprints; captures substructure frequency
- **B2 — BMF-only**: binary bit-vector (ECFP4); encodes presence only; assesses whether counts add value over binary

Together these isolate how much KPGT graph embeddings contribute on top of fingerprints alone.

---

### Step 2C — MoFPGNN: Pretrained KPGT + Morgan Hybrid (Options C1 / C2, core models)

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

Two hybrid variants, one per fingerprint type:
- **C1 — KPGT + CMF**: count-based fusion (4352-dim input)
- **C2 — KPGT + BMF**: binary fusion (4352-dim input); assesses value of binary vs count-based representations

## Project Structure

```
.
├── config/
│   ├── kpgt_pretrained_regressor_config.yaml      # Option A: KPGT-only regressor
│   ├── morgan_only_config.yaml                    # Option B1: CMF-only ablation
│   ├── morgan_binary_only_config.yaml             # Option B2: BMF-only ablation
│   ├── kpgt_morgan_pretrained_hybrid_config.yaml  # Option C1: KPGT + CMF hybrid (MoFPGNN core)
│   └── kpgt_morgan_binary_hybrid_config.yaml      # Option C2: KPGT + BMF hybrid
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

Uses the official pretrained KPGT model to extract fixed 2304-dim embeddings, then trains a lightweight regressor. 

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

### Option B1 - CMF-only (count-based Morgan fingerprints)

MLP trained on 2048-bit count-based Morgan fingerprints. No graph information.

```bash
# 1. Extract circular (count-based) Morgan fingerprints
python scripts/extract_morgan_fingerprint.py --fp_type circular

# 2. Train
python scripts/run_kpgt_pipeline.py --config config/morgan_only_config.yaml
```

### Option B2 - BMF-only (binary Morgan fingerprints / ECFP4)

MLP trained on 2048-bit binary Morgan fingerprints. Assesses whether binary presence encoding differs from count-based (CMF).

```bash
# 1. Extract binary Morgan fingerprints
python scripts/extract_morgan_fingerprint.py --fp_type binary

# 2. Train
python scripts/run_kpgt_pipeline.py --config config/morgan_binary_only_config.yaml
```

### Option C1 - KPGT + CMF hybrid (MoFPGNN core model)

Fuses pretrained KPGT embeddings (2304-dim) with count-based Morgan fingerprints (2048-dim).

```bash
# 1. Extract KPGT embeddings (if not already done — see Option A step 1)

# 2. Extract circular Morgan fingerprints (if not already done)
python scripts/extract_morgan_fingerprint.py --fp_type circular

# 3. Train
python scripts/run_kpgt_pipeline.py --config config/kpgt_morgan_pretrained_hybrid_config.yaml
```

### Option C2 - KPGT + BMF hybrid

Fuses pretrained KPGT embeddings with binary Morgan fingerprints. Explores whether binary presence encoding changes performance relative to CMF.

```bash
# 1. Extract KPGT embeddings (if not already done — see Option A step 1)

# 2. Extract binary Morgan fingerprints (if not already done)
python scripts/extract_morgan_fingerprint.py --fp_type binary

# 3. Train
python scripts/run_kpgt_pipeline.py --config config/kpgt_morgan_binary_hybrid_config.yaml
```

## Detailed Usage

### 1. Extract KPGT Pretrained Embeddings

See [Pre-requisite](#pre-requisite) for full setup. This produces `data/fingerprints/AGILE/kpgt_embeddings.pkl`.

### 2. Extract Morgan Fingerprints

Two fingerprint types are supported via `--fp_type`:

| Type | Flag | Description | Output file | 
|------|------|-------------|-------------|
| Circular (CMF) | `--fp_type circular` | Count-based; encodes substructure frequency | `morgan_circular.pkl` | 
| Binary (BMF) | `--fp_type binary` | Bit-vector (ECFP4); encodes substructure presence only | `morgan_binary.pkl` | 

```bash
# Circular (count-based) — CMF
python scripts/extract_morgan_fingerprint.py \
    --fp_type circular \
    --data_name AGILE \
    --save_path data/fingerprints/AGILE \
    --radius 2 \
    --n_bits 2048

# Binary (ECFP4-style) — BMF
python scripts/extract_morgan_fingerprint.py \
    --fp_type binary \
    --data_name AGILE \
    --save_path data/fingerprints/AGILE \
    --radius 2 \
    --n_bits 2048
```

### 3. Run Individual Experiments

Each experiment is controlled by a YAML config file:

```bash
# Option A — KPGT-only regressor
python scripts/run_kpgt_pipeline.py --config config/kpgt_pretrained_regressor_config.yaml

# Option B1 — CMF-only ablation
python scripts/run_kpgt_pipeline.py --config config/morgan_only_config.yaml

# Option B2 — BMF-only ablation
python scripts/run_kpgt_pipeline.py --config config/morgan_binary_only_config.yaml

# Option C1 — KPGT + CMF hybrid (MoFPGNN core)
python scripts/run_kpgt_pipeline.py --config config/kpgt_morgan_pretrained_hybrid_config.yaml

# Option C2 — KPGT + BMF hybrid
python scripts/run_kpgt_pipeline.py --config config/kpgt_morgan_binary_hybrid_config.yaml
```

Override the split from the command line (applies to all configs):

```bash
python scripts/run_kpgt_pipeline.py \
    --config config/<any_config>.yaml \
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

| Option | `model_type` | Fingerprint | Description |  Required data |
|--------|---|---|---|---|
| A | `kpgt_pretrained_regressor` | — | KPGT embeddings + MLP | `kpgt_embeddings.pkl` |
| B1 | `morgan_only` | CMF (circular) | Count-based FP-only MLP | `morgan_circular.pkl` |
| B2 | `morgan_only` | BMF (binary) | ECFP4-style FP-only MLP | `morgan_binary.pkl` |
| C1 | `kpgt_morgan_pretrained_hybrid` | CMF (circular) | KPGT + CMF late fusion (MoFPGNN core) | `kpgt_embeddings.pkl`, `morgan_circular.pkl` |
| C2 | `kpgt_morgan_pretrained_hybrid` | BMF (binary) | KPGT + BMF late fusion |`kpgt_embeddings.pkl`, `morgan_binary.pkl` |

### CMF-Only Config (Option B1)

```yaml
model_type: morgan_only
dataset: AGILE
split: random                    # random | Murcko_scaffold
device: cuda                     # cuda | cpu

# Paths
csv_path: data/AGILE.csv
split_path: data/splits/AGILE/random.npy
morgan_path: data/fingerprints/AGILE/morgan_circular.pkl

# Training
epochs: 100
lr: 0.001
batch_size: 64
weight_decay: 0.0001
patience: 30
warmup_epochs: 5
grad_clip: 1.0
loss_fn: huber

# MLP Head (input = 2048)
morgan_dim: 2048
mlp_head:
  hidden_layers: [512, 256]
  dropout: 0.3
```

### BMF-Only Config (Option B2)

```yaml
model_type: morgan_only
dataset: AGILE
split: random                    # random | Murcko_scaffold
device: cuda                     # cuda | cpu

# Paths
csv_path: data/AGILE.csv
split_path: data/splits/AGILE/random.npy
morgan_path: data/fingerprints/AGILE/morgan_binary.pkl

# Training
epochs: 100
lr: 0.001
batch_size: 64
weight_decay: 0.0001
patience: 30
warmup_epochs: 5
grad_clip: 1.0
loss_fn: huber

# MLP Head (input = 2048)
morgan_dim: 2048
mlp_head:
  hidden_layers: [512, 256]
  dropout: 0.3
```

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

# Training (20 epochs, hidden_dim 512)
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

### KPGT + CMF Hybrid Config (Option C1 — MoFPGNN core)

```yaml
model_type: kpgt_morgan_pretrained_hybrid
dataset: AGILE
split: random                    # random | Murcko_scaffold
device: cuda                     # cuda | cpu

# Paths
csv_path: data/AGILE.csv
split_path: data/splits/AGILE/random.npy
embeddings_path: data/fingerprints/AGILE/kpgt_embeddings.pkl
morgan_path: data/fingerprints/AGILE/morgan_circular.pkl

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
```

### KPGT + BMF Hybrid Config (Option C2)

```yaml
model_type: kpgt_morgan_pretrained_hybrid
dataset: AGILE
split: random                    # random | Murcko_scaffold
device: cuda                     # cuda | cpu

# Paths
csv_path: data/AGILE.csv
split_path: data/splits/AGILE/random.npy
embeddings_path: data/fingerprints/AGILE/kpgt_embeddings.pkl
morgan_path: data/fingerprints/AGILE/morgan_binary.pkl

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
```