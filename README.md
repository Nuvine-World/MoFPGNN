<h1 align="center">MoFPGNN: Molecular Fingerprint-Enhanced Graph Neural Networks for LNP Transfection Efficiency Prediction</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/></a>
</p>

<p align="justify">
<strong>MoFPGNN</strong> predicts the mRNA transfection potency (mTP) of ionizable lipids used in lipid nanoparticle (LNP) delivery. The core model <strong>fine-tunes the pretrained KPGT (LiGhT) graph transformer end-to-end</strong> on the AGILE dataset, and optionally <strong>fuses a 2048-bit Morgan fingerprint</strong> (count-based CMF or binary BMF) with the transformer's graph readout before the regression head.
</p>

<p align="justify">
The hypothesis is that the pretrained KPGT graph representation and fixed-radius substructure counts (Morgan fingerprints) encode complementary chemical information. Two fingerprint variants are compared: <strong>circular Morgan fingerprints (CMF)</strong>, which encode substructure counts, and <strong>binary Morgan fingerprints (BMF)</strong>, which encode presence only. Morgan-only MLP baselines isolate how much the graph transformer adds over fingerprints alone.
</p>

### Key Features

- **End-to-end KPGT fine-tune** - the full LiGhT graph transformer (~91M params) is fine-tuned from the official `base.pth` pretrained checkpoint; a fresh 3-layer GELU head is trained on the 2304-dim graph readout (`768 × 3`).
- **KPGT + Morgan hybrids** - the (z-scored) Morgan fingerprint is combined with the 2304-dim readout through a **gated fusion** block before the head: the fingerprint is encoded, a sigmoid gate (conditioned on the readout) scales it, and the result is concatenated - `head([ readout(2304) ‖ gate·enc(morgan) ]) → mTP`. Available with CMF (count) or BMF (binary). The gate lets the model down-weight the fingerprint when the graph already explains the molecule.
- **Morgan-only MLP baselines** - CMF-only and BMF-only, no graph information, to quantify the transformer's contribution.
- **Two Morgan variants (both 2048-bit):** CMF = DeepChem `CircularFingerprint(radius=1024, size=2048, is_counts_based=True, chiral=True)` (~311 active bits); BMF = RDKit ECFP4 (radius=2).
- **Evaluation:** R², RMSE, MAE, Pearson r, screening metrics (EF/NDCG/HitRate @5%/10% with 95% bootstrap CIs), and percentile-bin ranking confusion matrices.
- **Random and Murcko scaffold splits** (880 / 110 / 110).

## Table of Contents

  * [Models](#models)
  * [Project Structure](#project-structure)
  * [Environment Setup](#environment-setup)
  * [Prerequisites](#prerequisites)
  * [Running the models](#running-the-models)
  * [Outputs](#outputs)
  * [Ranking confusion matrices](#ranking-confusion-matrices)
  * [Configuration reference](#configuration-reference)

## Models

| Tag | Model | Script | Fingerprint | What trains |
|-----|-------|--------|-------------|-------------|
| **A** | KPGT (fine-tune) | `finetune_kpgt.py --morgan none` | - | whole LiGhT transformer + head |
| **C1** | KPGT + CMF | `finetune_kpgt.py --morgan count` | CMF (count) | LiGhT transformer + head; CMF gated-fused with the readout |
| **C2** | KPGT + BMF | `finetune_kpgt.py --morgan binary` | BMF (binary) | LiGhT transformer + head; BMF gated-fused with the readout |
| **B1** | CMF-only | `run_kpgt_pipeline.py --config config/morgan_only_config.yaml` | CMF (count) | MLP on fingerprint only |
| **B2** | BMF-only | `run_kpgt_pipeline.py --config config/morgan_binary_only_config.yaml` | BMF (binary) | MLP on fingerprint only |

```
                 KPGT fine-tune (A / C1 / C2)
   molecular graph + ECFP + RDKit descriptors
                      |
            [ pretrained LiGhT transformer ]      (fine-tuned end-to-end)
                      |
              2304-dim graph readout  ──(C1/C2)──>  gated fusion + Morgan FP (2048)
                      |                                    |
                      └──────────────> [ 3-layer GELU head ] ──> mTP

                 Morgan-only baselines (B1 / B2)
        Morgan FP (2048) ──> [ 7-layer MLP head ] ──> mTP
```

## Project Structure

```
.
├── config/
│   ├── morgan_only_config.yaml             # B1: CMF-only baseline
│   └── morgan_binary_only_config.yaml      # B2: BMF-only baseline
│
├── models/
│   └── hybrid_model.py        # MorganOnlyMLP (Morgan-only baselines)
│
├── pipeline/
│   ├── kpgt_trainer.py        # trainer for the Morgan-only baselines
│   └── kpgt_pipeline.py       # baseline pipeline orchestrator
│
├── scripts/
│   ├── extract_morgan_fingerprint.py   # 2048-bit Morgan FP extraction (CMF / BMF)
│   ├── finetune_kpgt.py                # KPGT fine-tune (A) + gated KPGT+Morgan hybrids (C1/C2)
│   ├── run_kpgt_pipeline.py            # Morgan-only baselines (B1 / B2)
│   ├── plot_confusion_grid.py          # percentile-bin ranking confusion matrices
│   └── seed_sweep.py                   # multi-seed runs -> mean +/- std per model
│
├── utils/
│   ├── io_tools.py            # pickle / YAML I/O
│   ├── metrics.py             # regression + screening metrics
│   ├── visual_utils.py        # plots
│   └── confusion_matrix_utils.py
│
├── data/
│   ├── AGILE.csv              # dataset (1100 molecules: SMILES, Target)
│   ├── fingerprints/AGILE/    # Morgan fingerprint pickles
│   ├── splits/AGILE/          # random.npy, Murcko_scaffold.npy (880/110/110)
│   └── kpgt_finetune/         # auto-built KPGT-format caches (created on first fine-tune)
│
├── KPGT/                      # cloned KPGT repo (lihan97/kpgt) - provides the LiGhT model
├── checkpoints/              # saved model weights
├── results/                  # experiment outputs
└── mofpgnn.yml               # single combined conda environment
```

## Environment Setup

```bash
conda env create -f mofpgnn.yml
conda activate mofpgnn
```

**Reliable fallback (recommended if the full solve is slow/fails):** since the fine-tune already
runs in your existing `KPGT` env, just add the few packages it lacks and use that as the one env:

```bash
conda activate KPGT
pip install torch_geometric seaborn matplotlib scikit-learn deepchem
```

## Prerequisites

### 1. Clone KPGT and download the pretrained checkpoint

```bash
# from the repo root
git clone https://github.com/lihan97/kpgt.git ./KPGT

# download base.pth from https://figshare.com/s/d488f30c23946cf6898f
# and place it at:
#   ./KPGT/models/pretrained/base/base.pth
```

The fine-tune imports the LiGhT model from this clone (via `--kpgt_dir ./KPGT`) and never
modifies it. The first fine-tune run also builds the KPGT-format graph/fingerprint/descriptor
caches under `data/kpgt_finetune/` automatically (subsequent runs reuse them).

### 2. Generate the Morgan fingerprints (needed for B1/B2 and the C1/C2 hybrids)

```bash
conda activate mofpgnn
python scripts/extract_morgan_fingerprint.py --fp_type count    # -> data/fingerprints/AGILE/morgan_count.pkl  (CMF)
python scripts/extract_morgan_fingerprint.py --fp_type binary   # -> data/fingerprints/AGILE/morgan_binary.pkl (BMF)
```

## Running the models

All commands run in the `mofpgnn` env from the repo root. Swap `--split random` for
`--split Murcko_scaffold` to get the scaffold-split results.

```bash
conda activate mofpgnn

# A: KPGT fine-tune (KPGT-only)
python scripts/finetune_kpgt.py \
    --kpgt_dir ./KPGT --checkpoint ./KPGT/models/pretrained/base/base.pth \
    --split random

# C1: KPGT + CMF (count Morgan, gated fusion)
python scripts/finetune_kpgt.py \
    --kpgt_dir ./KPGT --checkpoint ./KPGT/models/pretrained/base/base.pth \
    --split random --morgan count

# C2: KPGT + BMF (binary Morgan, gated fusion)
python scripts/finetune_kpgt.py \
    --kpgt_dir ./KPGT --checkpoint ./KPGT/models/pretrained/base/base.pth \
    --split random --morgan binary

# B1: CMF-only MLP baseline
python scripts/run_kpgt_pipeline.py --config config/morgan_only_config.yaml --split random

# B2: BMF-only MLP baseline
python scripts/run_kpgt_pipeline.py --config config/morgan_binary_only_config.yaml --split random
```

Run each of the five again with `--split Murcko_scaffold` for the scaffold split.

**Fine-tune recipe** (defaults baked into `finetune_kpgt.py`): Adam (lr 4e-5, weight_decay 0),
polynomial-decay LR with ~10% warmup, gradient clipping 5.0, MSE on train-z-scored labels,
3-layer GELU head `[512, 512]` (dropout 0.2), batch 32, **50 epochs** (the LR schedule spans the
full run, so a longer schedule keeps the LR useful instead of annealing to ~0 by epoch 20),
best-validation-RMSE selection (early-stop patience 20), seed 23. For the hybrids the Morgan
fingerprint is z-scored on the train split (clipped to `[-10, 10]`) and combined with the
2304-dim readout via **gated fusion** (encoded to `--morgan_emb_dim` = 256, then a sigmoid gate
scales it before the head). Override any default via flags (`--lr`, `--n_epochs`, `--dropout`,
`--seed`, `--morgan_emb_dim`, ...). Each run also writes a `loss_curve.png` (train vs. validation
loss per epoch, normalised-MSE units) so you can see whether training has converged.

**Morgan-only baseline recipe** (`morgan_only_config.yaml` / `morgan_binary_only_config.yaml`):
`MinMaxScaler(-1, 1)` on `[features | labels]`, plain Adam (lr 2e-4), MSE, batch 100, no
shuffle, 100 epochs, keep-best-val; 7-layer head `[200,300,500,500,300,200]`. The CMF baseline
reaches **R² ≈ 0.79** on the random split. Seed is set in the config (`seed: 42`); override with
`--seed`.

## Outputs

Every run writes to `results/<run_name>/`, where `<run_name>` is:

| Model | `<run_name>` prefix |
|-------|---------------------|
| A - KPGT fine-tune | `kpgt_finetune-<split>-<timestamp>` |
| C1 - KPGT + CMF | `kpgt_finetune-cmf-<split>-<timestamp>` |
| C2 - KPGT + BMF | `kpgt_finetune-bmf-<split>-<timestamp>` |
| B1 - CMF-only | `morgan_only-<split>-<timestamp>` |
| B2 - BMF-only | `morgan_only-binary-<split>-<timestamp>` |

| File | Contents |
|------|----------|
| `metrics.txt` | R², RMSE, MAE, Pearson r, EF/NDCG/HitRate @5%/10% (95% bootstrap CIs), relative-error stats |
| `{train,val,test}_results.csv` | true vs predicted values per split |
| `loss_curve.png` | training vs. validation loss per epoch (normalised-MSE; marks best epoch) |
| `prediction_vs_true_*.jpg` | predicted vs true scatter |
| `confusion_test.jpg` | percentile-bin ranking confusion matrix (see below) |

Model checkpoints are saved to `checkpoints/<run_name>.pth`.

## Ranking confusion matrices

Both the true and predicted mTP are binned into 6 percentile bins; the matrix is row-normalised
by the true bin (the diagonal is the fraction of each true bin that lands in the correct
predicted bin). The title reports **`Accuracy: XX.X%`** = the mean of the row-normalised
diagonal, i.e. the average rate at which a compound's predicted percentile bin matches its true
bin (matches Fig. 6 of the AGILE benchmark).

To build the per-model figures and a combined grid across all five models from their saved
`test_results.csv` files:

```bash
python scripts/plot_confusion_grid.py --split random
python scripts/plot_confusion_grid.py --split Murcko_scaffold
```

This writes `confusion_<tag>_<split>.png` (one per model) and `confusion_grid_<split>.png` to
`results/confusion_matrices/<split>/`. The script auto-discovers the most recent run directory
for each model, so no paths need editing.

## Configuration reference

The Morgan-only baselines are configured by YAML; the fine-tune is configured by CLI flags.

### Morgan-only baseline configs

| Flag | CMF (`morgan_only_config.yaml`) | BMF (`morgan_binary_only_config.yaml`) | Effect |
|------|---------------------------------|----------------------------------------|--------|
| `feature_scaling` | `minmax_joint` | `minmax_joint` | `MinMaxScaler(-1,1)` on `[features\|labels]`, fit on all data |
| `lr` | `0.0002` | `0.0002` | plain Adam learning rate |
| `batch_size` | `100` | `100` | - |
| `weight_decay` | `0.0` | `0.0` | AdamW(wd=0) ≡ plain Adam |
| `scheduler` | `none` | `none` | constant lr (no warmup/cosine) |
| `shuffle` | `false` | `false` | DataLoader not shuffled |
| `grad_clip` | `0` | `0` | `<=0` disables clipping |
| `loss_fn` | `mse` | `mse` | matches the R²/RMSE objective |
| `morgan_path` | `morgan_count.pkl` | `morgan_binary.pkl` | which fingerprint pickle to load |

### Why the fingerprint choice matters

| Item | ECFP4-style FP (BMF) | CMF |
|------|----------------------|-----|
| Recipe | RDKit ECFP4, radius=2, ~54 active bits | DeepChem `CircularFingerprint(radius=1024, size=2048, is_counts_based=True, chiral=True)`, ~311 active bits |
| Coverage | only ECFP4-scale environments | effectively unbounded radius + counts + chirality |

The long-chain ionizable lipids in AGILE need an effectively unbounded radius to be described, so
a radius-2 bit-vector feeds a sparser, information-poorer input. Switching to the CMF lifts the
Morgan-only R² from ~0.60 to ~0.79.

## Reproducibility and seed reporting

With only 110 test molecules, a single run is noisy - especially the KPGT fine-tune (~91M
parameters on 880 training points) and *especially* on the Murcko scaffold split, where R² can
swing by ~0.15 between seeds.  `scripts/seed_sweep.py` runs a model
over several seeds (fresh subprocess each) and prints/saves the per-seed table plus mean/std for
every test metric:

```bash
# KPGT + CMF (gated), Murcko scaffold, 5 seeds -> mean +/- std
python scripts/seed_sweep.py \
    --kpgt_dir ./KPGT --checkpoint ./KPGT/models/pretrained/base/base.pth \
    --split Murcko_scaffold --morgan count --seeds 0 1 2 3 4
```

It accepts the same `--split`, `--morgan`, `--n_epochs`, and `--lr` flags as `finetune_kpgt.py`,
and writes a summary to `results/seed_sweep-...txt`. Use the **same seed set for every model** so
the comparison is fair. Seeds are `np.random`, `torch`, `random`, CUDA, and `dgl` - but note that
identical seeds do **not** reproduce results bit-for-bit across different machines/library
versions (GPU nondeterminism), so cross-machine differences of a few points are expected.

### Split-dependent behaviour

The fingerprint helps in-distribution but not out-of-distribution: on the **random** split
KPGT + CMF is the strongest model, whereas on the **Murcko scaffold** split KPGT-only generalises
best and adding a raw fingerprint *hurts* (its substructures do not transfer to unseen scaffolds).
Gated fusion mitigates this by learning to down-weight the fingerprint on novel scaffolds, but the
graph model alone remains the best scaffold generaliser.