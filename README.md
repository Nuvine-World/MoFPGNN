<h1 align="center">MoFPGNN: Molecular Fingerprint-Enhanced Graph Neural Networks for LNP Transfection Efficiency Prediction</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/></a>
</p>

Code for the KPGT half of the manuscript: predicting mRNA transfection potency (mTP)
of ionizable lipids from the AGILE library of 1100 molecules, by fusing a Morgan
fingerprint with a frozen pretrained graph-transformer embedding.

Five models, one random split (880 / 110 / 110) fixed in
`data/splits/AGILE/random.npy`, ten seeds each.

```
   SMILES
     ├──> [ frozen KPGT / LiGhT ] ──> readout (2304)  ─┐
     │                                                 ├──> concat (4352) ──> [ MLP head ] ──> mTP
     └──> [ RDKit Morgan, no params ] ──> CMF|BMF (2048)┘
```

Only the MLP head trains. The KPGT embedding is a cached lookup and the fingerprint
is deterministic.

The Chemprop half of the manuscript is the collaborator's work and lives elsewhere.

## Models

Names follow the manuscript: the fingerprint baselines are named for the MLP applied
to them, counting its output layer.

| Manuscript name | Inputs | Config |
|---|---|---|
| MLP<sub>7</sub> + CMF | count fingerprint | `config/morgan_only_config.yaml` |
| MLP<sub>7</sub> + BMF | binary fingerprint | `config/morgan_binary_only_config.yaml` |
| KPGT | embedding | `config/kpgt_pretrained_regressor_config.yaml` |
| KPGT + CMF | embedding + count fingerprint | `config/kpgt_morgan_pretrained_hybrid_config.yaml` |
| KPGT + BMF | embedding + binary fingerprint | `config/kpgt_morgan_binary_hybrid_config.yaml` |

## Setup

```bash
conda env create -f mofpgnn.yml
conda activate mofpgnn
```

Verified with python 3.10.20, torch 2.3.1, numpy 1.26.4, scikit-learn 1.5.2,
rdkit 2024.09.1, PyG 2.6.1.

## Cached features

Every model reads pickles from `data/fingerprints/AGILE/`. They are not in the repo
and must be generated once before any run.

```bash
python scripts/extract_morgan_fingerprint.py --fp_type count  --radius 2 --n_bits 2048
python scripts/extract_morgan_fingerprint.py --fp_type binary --radius 2 --n_bits 2048
```

Radius 2 and 2048 bits are the reported settings and are also the script defaults.

The KPGT embeddings additionally need a KPGT clone, its checkpoint and DGL, and the
extraction **overwrites** the existing cache without a backup:

```bash
git clone https://github.com/lihan97/kpgt.git ./KPGT
# download base.pth (447 MB) from https://figshare.com/s/d488f30c23946cf6898f
# place at ./KPGT/models/pretrained/base/base.pth

# DGL 2.3.0 needs torchdata < 0.10, or `import dgl` fails
pip install --no-deps "torchdata==0.9.0"

python scripts/extract_kpgt_embeddings.py --kpgt_dir ./KPGT
```

DGL is needed only for this step. Once `kpgt_embeddings.pkl` exists, everything else
runs without it.

## Run one model

```bash
python scripts/run_kpgt_pipeline.py \
    --config config/morgan_only_config.yaml --split random --seed 0
```

Writes `results/<run_name>/` with `metrics.txt` (R2, RMSE, MAE, Pearson r, MRE, and
EF / NDCG / HitRate at 5% and 10%) and `{train,val,test}_results.csv` holding true and
predicted values on the original mTP scale.

## Reproduce the reported numbers

```bash
python scripts/seed_sweep_all.py
```

Runs all five configs over the ten seeds `0 1 2 3 4 5 6 7 23 42` and writes
`results/seed_sweep_all-<timestamp>/`, with one directory per run plus `per_run.csv`,
one row per model and seed. That file is the source of every KPGT number in the
manuscript. A few minutes on CPU.

Ranking metrics are reported as the mean over 1000 bootstrap resamples of the test
set, with the resample draws shared across models so comparisons stay paired
(`utils/metrics.py`).

## Block attribution

How much of each hybrid's prediction comes from the embedding versus the fingerprint,
by permutation, occlusion and integrated gradients. The models are retrained per seed,
so the attribution describes exactly the models in the results table.

```bash
python scripts/explain_blocks.py \
    --config config/kpgt_morgan_pretrained_hybrid_config.yaml --split random --seeds 0 1 2 3 4
python scripts/explain_blocks.py \
    --config config/kpgt_morgan_binary_hybrid_config.yaml --split random --seeds 0 1 2 3 4
```

Writes `block_attribution.csv` to `results/xai-<tag>-random/`.

## Manuscript figure

```bash
python scripts/make_fusion_figure.py
```

Parses the two hypothesis-test tables out of the manuscript source (`main.tex`,
kept with the paper rather than in this repo) and writes
`figures/fig_fusion_effects.{png,pdf}`, so the figure cannot drift from the tables.
The rendered figure is included here.

## Repository layout

```
config/     the five model configs
models/     the embedding regressor and the concatenation hybrid
pipeline/   data loading, scaling, training loop
scripts/    entry points (below)
utils/      io, metrics, plotting
data/       AGILE.csv and the random split index
figures/    the manuscript figure
```

| Script | Purpose |
|---|---|
| `run_kpgt_pipeline.py` | run one config at one seed |
| `seed_sweep_all.py` | all five configs over ten seeds, writes `per_run.csv` |
| `explain_blocks.py` | block attribution for the hybrids |
| `extract_morgan_fingerprint.py` | CMF and BMF caches |
| `extract_kpgt_embeddings.py` | KPGT embedding cache |
| `make_fusion_figure.py` | the manuscript figure |
