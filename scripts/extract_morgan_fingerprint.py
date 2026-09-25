import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import pandas as pd
from argparse import ArgumentParser

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from utils.io_tools import save_pickle

RDLogger.DisableLog('rdApp.*')


def compute_count_fingerprints(smiles_list, n_bits=2048, radius=2):
    from deepchem.feat import CircularFingerprint

    featurizer = CircularFingerprint(radius=radius, size=n_bits,
                                     is_counts_based=True, chiral=True)
    fps = featurizer.featurize(smiles_list)
    results = {}
    for smiles, fp in zip(smiles_list, fps):
        results[smiles] = np.asarray(fp, dtype=np.float32)
    return results


def compute_binary_fingerprints(smiles_list, radius=2, n_bits=2048):
    results = {}
    failed = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed.append(smiles)
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        results[smiles] = arr

    if failed:
        print(f"Warning: failed to parse {len(failed)} SMILES: {failed[:5]}...")

    return results


def get_args():
    parser = ArgumentParser(
        description="Extract Morgan fingerprints (count-based CMF or binary BMF)"
    )
    parser.add_argument("--data_name", type=str, default="AGILE")
    parser.add_argument("--save_path", type=str, default="data/fingerprints/AGILE")
    parser.add_argument("--radius", type=int, default=2,
                        help="Circular-fingerprint radius (default 2, the setting "
                             "behind the reported results)")
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument(
        "--fp_type",
        type=str,
        default="count",
        choices=["count", "count_rdkit", "binary"],
        help=(
            "count: count-based circular fingerprint via DeepChem, chiral "
            "(CMF) -> morgan_count.pkl; "
            "count_rdkit: RDKit count Morgan at --radius -> morgan_count_rdkit.pkl; "
            "binary: bit-vector Morgan fingerprint (BMF) -> morgan_binary.pkl"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    data = pd.read_csv(f"data/{args.data_name}.csv")
    smiles_list = data["SMILES"].tolist()

    print(
        f"Computing {args.n_bits}-bit {args.fp_type} Morgan fingerprints "
        f"for {len(smiles_list)} molecules..."
    )

    if args.fp_type == "binary":
        fingerprints = compute_binary_fingerprints(
            smiles_list, radius=args.radius, n_bits=args.n_bits
        )
    elif args.fp_type == "count_rdkit":
        fingerprints = compute_count_fingerprints_rdkit(
            smiles_list, radius=args.radius, n_bits=args.n_bits
        )
    else:  # "count" -> DeepChem CMF (recommended)
        fingerprints = compute_count_fingerprints(
            smiles_list, n_bits=args.n_bits, radius=args.radius
        )

    print(f"Successfully computed fingerprints for {len(fingerprints)} molecules.")
    sample_fp = next(iter(fingerprints.values()))
    print(f"Fingerprint shape: {sample_fp.shape}")
    print(f"Non-zero features (sample): {int(np.count_nonzero(sample_fp))}/{args.n_bits}")

    os.makedirs(args.save_path, exist_ok=True)
    filename = f"morgan_{args.fp_type}.pkl"
    save_dest = os.path.join(args.save_path, filename)
    save_pickle(fingerprints, save_dest)
    print(f"Saved to {save_dest}")
