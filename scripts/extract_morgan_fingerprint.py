import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import pandas as pd
from argparse import ArgumentParser

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from utils.io_tools import save_pickle

RDLogger.DisableLog('rdApp.*')


def compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """
    Compute count-based Morgan fingerprints for a list of SMILES strings.

    Args:
        smiles_list: list of SMILES strings
        radius: Morgan fingerprint radius (default 2)
        n_bits: number of bits / fingerprint length (default 2048)

    Returns:
        dict mapping SMILES -> numpy array of shape (n_bits,)
    """
    results = {}
    failed = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed.append(smiles)
            continue

        # Generate count-based Morgan fingerprint
        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        for idx, count in fp.GetNonzeroElements().items():
            arr[idx % n_bits] = count
        results[smiles] = arr

    if failed:
        print(f"Warning: failed to parse {len(failed)} SMILES: {failed[:5]}...")

    return results


def get_args():
    parser = ArgumentParser(description="Extract count-based Morgan fingerprints using RDKit")
    parser.add_argument("--data_name", type=str, default="AGILE")
    parser.add_argument("--save_path", type=str, default="data/fingerprints/AGILE")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n_bits", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    data = pd.read_csv(f"data/{args.data_name}.csv")
    smiles_list = data["SMILES"].tolist()

    print(f"Computing {args.n_bits}-bit Morgan fingerprints (radius={args.radius}) for {len(smiles_list)} molecules...")
    fingerprints = compute_morgan_fingerprints(smiles_list, radius=args.radius, n_bits=args.n_bits)
    print(f"Successfully computed fingerprints for {len(fingerprints)} molecules.")

    sample_fp = next(iter(fingerprints.values()))
    print(f"Fingerprint shape: {sample_fp.shape}")

    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, "morgan.pkl")
    save_pickle(fingerprints, save_path)
    print(f"Saved to {save_path}")
