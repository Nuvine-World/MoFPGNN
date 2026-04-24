import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import pandas as pd
from argparse import ArgumentParser

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from utils.io_tools import save_pickle

RDLogger.DisableLog('rdApp.*')


def compute_circular_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """
    Compute count-based (circular) Morgan fingerprints.
    Each feature value is the substructure count at that hash bucket.
    Equivalent to ECFP with integer counts rather than binary presence.
    """
    results = {}
    failed = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed.append(smiles)
            continue

        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        for idx, count in fp.GetNonzeroElements().items():
            arr[idx % n_bits] = count
        results[smiles] = arr

    if failed:
        print(f"Warning: failed to parse {len(failed)} SMILES: {failed[:5]}...")

    return results


def compute_binary_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """
    Compute binary Morgan fingerprints (ECFP-style bit vectors).
    Each feature is 1 if the substructure is present, 0 otherwise.
    Equivalent to ECFP4 (radius=2, 2048 bits), matching the LANTERN baseline.
    """
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
        description="Extract Morgan fingerprints (circular count-based or binary bit-vector)"
    )
    parser.add_argument("--data_name", type=str, default="AGILE")
    parser.add_argument("--save_path", type=str, default="data/fingerprints/AGILE")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument(
        "--fp_type",
        type=str,
        default="circular",
        choices=["circular", "binary"],
        help=(
            "circular: count-based Morgan FP (CMF); "
            "binary: bit-vector Morgan FP (BMF / ECFP4-style, matches LANTERN)"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    data = pd.read_csv(f"data/{args.data_name}.csv")
    smiles_list = data["SMILES"].tolist()

    print(
        f"Computing {args.n_bits}-bit {args.fp_type} Morgan fingerprints "
        f"(radius={args.radius}) for {len(smiles_list)} molecules..."
    )

    if args.fp_type == "binary":
        fingerprints = compute_binary_morgan_fingerprints(
            smiles_list, radius=args.radius, n_bits=args.n_bits
        )
    else:
        fingerprints = compute_circular_morgan_fingerprints(
            smiles_list, radius=args.radius, n_bits=args.n_bits
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
