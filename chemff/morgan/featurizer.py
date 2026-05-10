# chemff/morgan/featurizer.py
#
# BLUEPRINT — Morgan Fingerprint Featurizer
# ==========================================
# Purpose:
#   Generate Morgan (circular) fingerprint vectors from SMILES strings.
#   These vectors are one of the two branches fed into the chemff combined
#   model (the other branch being the D-MPNN chemprop embedding).
#
# Paper reference (LANTERN_EXT):
#   LANTERN uses count-based circular fingerprints (radius=2, nBits=2048)
#   generated via RDKit's Morgan algorithm with chirality encoding enabled.
#   This matches DeepChem's CircularFingerprint(size=2048, radius=2,
#   is_counts_based=True, chiral=True) used in the base LANTERN model.
#
# Fingerprint settings (defaults from paper):
#   radius    = 2      (Morgan radius; captures 2-hop neighborhoods)
#   n_bits    = 2048   (bit-vector length)
#   use_counts = True  (count-based, not binary)
#   use_chirality = True
#
# Output shape: (n_bits,) = (2048,) float32 array per molecule
#
# Dependencies:
#   conda install -c conda-forge rdkit
# ==========================================

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class MorganFeaturizer:
    """
    Convert SMILES strings to Morgan (circular) fingerprint vectors.

    Parameters
    ----------
    radius : int
        Morgan radius (number of hops).  Default 2.
    n_bits : int
        Length of the bit/count vector.  Default 2048.
    use_counts : bool
        If True, return count-based fingerprint (each bit = count of
        substructure occurrences).  If False, binary (0/1).  Default True.
    use_chirality : bool
        Include chiral information in the fingerprint.  Default True.

    Usage
    -----
    featurizer = MorganFeaturizer()

    # Single molecule
    fp = featurizer("CC(=O)Oc1ccccc1C(=O)O")   # aspirin → np.ndarray (2048,)

    # Batch
    fps = featurizer.batch(smiles_list)           # np.ndarray (N, 2048)

    # Persist (save/load dict keyed by SMILES — same format as LANTERN)
    featurizer.save(smiles_list, "data/fingerprints/AGILE/morgan.pkl")
    fp_dict = MorganFeaturizer.load("data/fingerprints/AGILE/morgan.pkl")
    """

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_counts: bool = True,
        use_chirality: bool = True,
    ):
        self.radius = radius
        self.n_bits = n_bits
        self.use_counts = use_counts
        self.use_chirality = use_chirality

    # ------------------------------------------------------------------
    # Core featurisation
    # ------------------------------------------------------------------

    def __call__(self, smiles: str) -> np.ndarray:
        """
        Featurize a single SMILES string.

        Returns
        -------
        np.ndarray, shape (n_bits,), dtype float32
            Morgan fingerprint vector.

        Raises
        ------
        ValueError
            If RDKit cannot parse the SMILES.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

        if self.use_counts:
            # Count-based: returns a UIntSparseIntVect, convert to dense
            fp = AllChem.GetMorganFingerprint(
                mol,
                radius=self.radius,
                useChirality=self.use_chirality,
            )
            # Convert sparse count dict → dense array
            arr = np.zeros(self.n_bits, dtype=np.float32)
            for idx, count in fp.GetNonzeroElements().items():
                hashed = idx % self.n_bits          # fold into n_bits
                arr[hashed] += count
        else:
            # Binary bit-vector
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.radius,
                nBits=self.n_bits,
                useChirality=self.use_chirality,
            )
            arr = np.array(fp, dtype=np.float32)

        return arr

    def batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Featurize a list of SMILES strings.

        Returns
        -------
        np.ndarray, shape (N, n_bits), dtype float32
        """
        fps = [self(s) for s in smiles_list]
        return np.stack(fps, axis=0)

    # ------------------------------------------------------------------
    # Persistence helpers (mirrors LANTERN's pickle-based fingerprint store)
    # ------------------------------------------------------------------

    def save(self, smiles_list: List[str], path: str) -> Dict[str, np.ndarray]:
        """
        Featurize all SMILES and save as a {smiles: fingerprint} pickle.

        This is the same format used by LANTERN's extract_fingerprint.py so
        the resulting file can be consumed directly by data_utils.load_features().

        Parameters
        ----------
        smiles_list : list of str
        path        : str — destination .pkl path (directories created if needed)

        Returns
        -------
        dict  {smiles_str: np.ndarray(n_bits,)}
        """
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        fp_dict: Dict[str, np.ndarray] = {}
        for smiles in smiles_list:
            fp_dict[smiles] = self(smiles)

        with open(path, "wb") as f:
            pickle.dump(fp_dict, f)

        print(f"Saved {len(fp_dict)} Morgan fingerprints → {path}")
        return fp_dict

    @staticmethod
    def load(path: str) -> Dict[str, np.ndarray]:
        """
        Load a previously saved {smiles: fingerprint} pickle.

        Parameters
        ----------
        path : str — path to .pkl file

        Returns
        -------
        dict  {smiles_str: np.ndarray(n_bits,)}
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        """Dimension of the fingerprint vector (= n_bits)."""
        return self.n_bits

    def __repr__(self) -> str:
        return (
            f"MorganFeaturizer(radius={self.radius}, n_bits={self.n_bits}, "
            f"use_counts={self.use_counts}, use_chirality={self.use_chirality})"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
# Run as:
#   python -m chemff.morgan.featurizer --data_name AGILE --save_path data/fingerprints/AGILE
#
if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Generate Morgan fingerprints")
    parser.add_argument("--data_name",  type=str, default="AGILE")
    parser.add_argument("--save_path",  type=str, default="data/fingerprints/AGILE")
    parser.add_argument("--radius",     type=int, default=2)
    parser.add_argument("--n_bits",     type=int, default=2048)
    parser.add_argument("--no_counts",  action="store_true",
                        help="Use binary fingerprint instead of count-based")
    parser.add_argument("--no_chirality", action="store_true",
                        help="Disable chirality encoding")
    args = parser.parse_args()

    data = pd.read_csv(f"data/{args.data_name}.csv")
    smiles_list = data["SMILES"].tolist()

    featurizer = MorganFeaturizer(
        radius=args.radius,
        n_bits=args.n_bits,
        use_counts=not args.no_counts,
        use_chirality=not args.no_chirality,
    )

    out_path = os.path.join(args.save_path, "morgan.pkl")
    featurizer.save(smiles_list, out_path)
    print(f"Fingerprint shape per molecule: ({featurizer.n_bits},)")
