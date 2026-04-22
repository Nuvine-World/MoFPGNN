import os
import sys
import pathlib
import warnings

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from utils.io_tools import save_pickle

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def _compute_rdkit_fp(mol, min_path=1, max_path=7, fp_size=512):
    """RDKit topological fingerprint matching KPGT's preprocessing."""
    fp = AllChem.RDKFingerprint(
        mol, minPath=min_path, maxPath=max_path, fpSize=fp_size
    )
    return np.array(fp, dtype=np.float32)


def _compute_molecular_descriptors(mol):
    """
    Compute the 200-dim RDKit2DNormalized descriptors that KPGT uses.
    """
    try:
        from descriptastorus.descriptors import rdNormalizedDescriptors
    except ImportError:
        raise ImportError(
            "descriptastorus is required but not installed.\n"
            "The KPGT checkpoint expects exactly 200 RDKit2DNormalized descriptors.\n"
            "Fix: pip install descriptastorus\n"
            "  or: pip install git+https://github.com/bp-kelley/descriptastorus"
        )
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    smiles = Chem.MolToSmiles(mol)
    results = generator.process(smiles)
    if results is None:
        return None
    return np.array(results[1:], dtype=np.float32)


def extract_kpgt_official(smiles_list, kpgt_dir, checkpoint_path, batch_size=128):
    """
    Extract 2304-dim embeddings using the official KPGT pretrained model.
    """
    # Add KPGT source to Python path
    sys.path.insert(0, kpgt_dir)

    try:
        import dgl
        from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES, smiles_to_graph_tune
        from src.data.collator import Collator_tune
        from src.model.light import LiGhTPredictor as LiGhT
        from src.model_config import config_dict
    except ImportError as e:
        print(f"ERROR: Cannot import KPGT modules from '{kpgt_dir}'.")
        print(f"  Import error: {e}")
        print("\n  Make sure you have:")
        print("    1. Cloned the repo:  git clone https://github.com/lihan97/kpgt.git")
        print("    2. Activated the KPGT env:  conda activate KPGT")
        print("    3. Installed deps:  pip install dgl dgllife descriptastorus")
        return None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Load model
    config = config_dict["base"]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)

    # Pre-compute dimensions
    test_mol = Chem.MolFromSmiles("C")
    test_fp = _compute_rdkit_fp(test_mol)
    test_md = _compute_molecular_descriptors(test_mol)
    d_fp_feats = len(test_fp)
    d_md_feats = len(test_md)

    print(f"  FP dim: {d_fp_feats}, MD dim: {d_md_feats}")

    model = LiGhT(
        d_node_feats=config["d_node_feats"],
        d_edge_feats=config["d_edge_feats"],
        d_g_feats=config["d_g_feats"],
        d_fp_feats=d_fp_feats,
        d_md_feats=d_md_feats,
        d_hpath_ratio=config["d_hpath_ratio"],
        n_mol_layers=config["n_mol_layers"],
        path_length=config["path_length"],
        n_heads=config["n_heads"],
        n_ffn_dense_layers=config["n_ffn_dense_layers"],
        input_drop=0.0,
        attn_drop=0.0,
        feat_drop=0.0,
        n_node_types=vocab.vocab_size,
    )

    # Load checkpoint (strip DDP 'module.' prefix)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"  Loaded KPGT checkpoint from: {checkpoint_path}")
    print(f"  Model on: {device}")

    # ── Featurise molecules (fps/mds as tensors)
    print(f"  Featurising {len(smiles_list)} molecules...")
    graphs = []
    fps = []
    mds = []
    valid_smiles = []
    failed = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed.append(smi)
            continue

        try:
            g = smiles_to_graph_tune(smi)
            fp_np = _compute_rdkit_fp(mol)
            md_np = _compute_molecular_descriptors(mol)
            if md_np is None:
                failed.append(smi)
                continue

            fp = torch.from_numpy(fp_np).float()
            md = torch.from_numpy(md_np).float()

            graphs.append(g)
            fps.append(fp)
            mds.append(md)
            valid_smiles.append(smi)
        except Exception:
            failed.append(smi)
            continue

    if failed:
        print(f"  Warning: {len(failed)} molecules failed featurisation")

    # ── Extract embeddings in batches (labels as tensors)
    print(f"  Extracting embeddings for {len(valid_smiles)} molecules...")
    collator = Collator_tune(True)
    results = {}

    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            end = min(start + batch_size, len(graphs))
            batch_graphs = graphs[start:end]
            batch_fps = fps[start:end]
            batch_mds = mds[start:end]
            batch_smi = valid_smiles[start:end]

            # Dummy labels MUST be tensors (Collator_tune does torch.stack on them)
            dummy_labels = [torch.tensor([0.0], dtype=torch.float32) for _ in range(len(batch_graphs))]

            batch_data = list(zip(
                range(len(batch_graphs)),   # becomes "smiles_list" inside collator (ignored)
                batch_graphs,
                batch_fps,
                batch_mds,
                dummy_labels,
            ))
            _, batched_g, batched_fp, batched_md, _ = collator(batch_data)

            batched_g = batched_g.to(device)
            batched_fp = batched_fp.to(device)
            batched_md = batched_md.to(device)

            # Forward pass → 2304-dim embeddings
            emb = model.generate_fps(batched_g, batched_fp, batched_md)
            emb = emb.detach().cpu().numpy()

            for i, smi in enumerate(batch_smi):
                results[smi] = emb[i]

            if (start // batch_size) % 5 == 0:
                print(f"    Processed {end}/{len(graphs)} molecules...")

    print(f"  Extracted {len(results)} embeddings of dim {emb.shape[1]}")
    return results


def get_args():
    parser = ArgumentParser(description="Extract KPGT pretrained embeddings")
    parser.add_argument("--data_name", type=str, default="AGILE",
                        help="Dataset name (CSV at data/<name>.csv)")
    parser.add_argument("--save_path", type=str, default="data/fingerprints/AGILE",
                        help="Directory to save the embeddings pickle")
    parser.add_argument("--kpgt_dir", type=str, required=True,
                        help="Path to the cloned KPGT repository")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to base.pth (default: <kpgt_dir>/models/pretrained/base/base.pth)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Inference batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Resolve checkpoint path
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(args.kpgt_dir, "models", "pretrained", "base", "base.pth")

    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint not found at: {checkpoint}")
        print("\nDownload it from: https://figshare.com/s/d488f30c23946cf6898f")
        print(f"Then place it at:  {checkpoint}")
        sys.exit(1)

    if not os.path.isdir(args.kpgt_dir):
        print(f"ERROR: KPGT directory not found: {args.kpgt_dir}")
        print("\nClone it with:  git clone https://github.com/lihan97/kpgt.git")
        sys.exit(1)

    # Load dataset
    csv_path = f"data/{args.data_name}.csv"
    data = pd.read_csv(csv_path)
    smiles_list = data["SMILES"].tolist()
    print(f"Processing {len(smiles_list)} molecules from {args.data_name}...")

    # Extract embeddings
    embeddings = extract_kpgt_official(
        smiles_list, args.kpgt_dir, checkpoint, batch_size=args.batch_size
    )

    if embeddings is None or len(embeddings) == 0:
        print("ERROR: Embedding extraction failed. See errors above.")
        sys.exit(1)

    sample = next(iter(embeddings.values()))
    print(f"\nEmbedding dimension: {sample.shape[0]}")
    print(f"Successfully extracted for {len(embeddings)}/{len(smiles_list)} molecules")

    # Save
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, "kpgt_embeddings.pkl")
    save_pickle(embeddings, save_path)
    print(f"Saved to: {save_path}")