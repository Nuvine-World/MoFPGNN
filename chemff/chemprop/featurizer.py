from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Atom feature dimensions (must match ChempropEncoder.atom_fdim)
# ---------------------------------------------------------------------------
ATOM_FEATURES = {
    # (feature_name): list of possible values for one-hot encoding
    "atomic_num": list(range(1, 119)) + ["other"],  # 119 dims
    "degree": [0, 1, 2, 3, 4, 5],  # 6 dims
    "formal_charge": [-1, -2, 1, 2, 0],  # 5 dims
    "chiral_tag": [0, 1, 2, 3],  # 4 dims
    "num_Hs": [0, 1, 2, 3, 4],  # 5 dims
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],  # 5 dims
    # "is_aromatic" and "mass" are scalar, appended after one-hots
}
# Total atom feature dim = 119+6+5+4+5+5 + 2 (is_aromatic + mass_scaled) = 146
# Chemprop default: atom_fdim = 133  (uses slightly different bucketing)
ATOM_FDIM = 133  # match chemprop library default

# Bond feature dimensions
BOND_FDIM = 14  # match chemprop library default


def one_hot_encoding(value, choices: list) -> List[int]:
    """One-hot encode `value` against `choices`; unknown → last slot."""
    encoding = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    encoding[idx] = 1
    return encoding


def atom_features(atom) -> List[float]:
    """
    Build the feature vector for a single RDKit atom.

    Returns a list of floats of length ATOM_FDIM (133).

    Features (matching chemprop defaults):
      - atomic number   : one-hot, 100 elements + "other"  →  101 dims
      - degree          : one-hot [0-5]                     →    6 dims
      - formal charge   : scalar                            →    1 dim
      - num Hs (total)  : one-hot [0-4]                     →    5 dims
      - num radical e   : scalar                            →    1 dim
      - hybridization   : one-hot (SP, SP2, SP3, other)     →    4 dims
      - is_aromatic     : scalar bool                       →    1 dim
      - mass (scaled)   : scalar (/ 100)                    →    1 dim
    Total: 101+6+1+5+1+4+1+1 = 120   (pad to 133 for library compat.)
    """
    features = []
    # Atomic number: one-hot over [1..100] + unknown = 101 dims
    features += one_hot_encoding(atom.GetAtomicNum(), list(range(1, 101)))
    # Degree: one-hot over [0..5] + unknown = 7 dims
    features += one_hot_encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
    # Formal charge: scalar = 1 dim
    features += [float(atom.GetFormalCharge())]
    # Number of Hs: one-hot over [0..4] + unknown = 6 dims
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    # Number of radical electrons: scalar = 1 dim
    features += [float(atom.GetNumRadicalElectrons())]
    # Hybridization: one-hot over 5 types + unknown = 6 dims
    features += one_hot_encoding(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ],
    )
    # Is aromatic: scalar = 1 dim
    features += [float(atom.GetIsAromatic())]
    # Scaled mass: scalar = 1 dim
    features += [atom.GetMass() / 100.0]
    # Pad to ATOM_FDIM (133) for chemprop library compatibility
    features += [0.0] * (ATOM_FDIM - len(features))
    return features


def bond_features(bond) -> List[float]:
    """
    Build the feature vector for a single RDKit bond.

    Returns a list of floats of length BOND_FDIM (14).

    Features:
      - bond type       : one-hot (single, double, triple, aromatic)  → 4 dims
      - is_conjugated   : scalar bool                                  → 1 dim
      - is_in_ring      : scalar bool                                  → 1 dim
      - stereo          : one-hot (STEREONONE, STEREOANY, STEREOZ,
                                   STEREOE, STEREOCIS, STEREOTRANS)   → 6 dims
    Total: 4+1+1+6 = 12  (pad to 14 for library compat.)
    """
    features = []
    # Bond type: one-hot over 4 types + unknown = 5 dims
    features += one_hot_encoding(
        bond.GetBondType(),
        [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ],
    )
    # Is conjugated: scalar = 1 dim
    features += [float(bond.GetIsConjugated())]
    # Is in ring: scalar = 1 dim
    features += [float(bond.IsInRing())]
    # Stereo: one-hot over 6 types + unknown = 7 dims
    features += one_hot_encoding(
        bond.GetStereo(),
        [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
        ],
    )
    return features


@dataclass
class MolGraph:
    """
    Container for a single molecule's graph representation.

    Attributes
    ----------
    n_atoms : int
        Number of atoms in the molecule.
    n_bonds : int
        Number of directed bonds (each undirected bond → 2 directed edges).
    f_atoms : np.ndarray, shape (n_atoms, ATOM_FDIM)
        Atom feature matrix.
    f_bonds : np.ndarray, shape (n_bonds, ATOM_FDIM + BOND_FDIM)
        Bond feature matrix. Each directed bond concatenates the source
        atom features with the bond features (chemprop convention).
    a2b : List[List[int]]
        a2b[a] = list of directed-bond indices incoming to atom a.
    b2a : List[int]
        b2a[b] = index of the source atom for directed bond b.
    b2revb : List[int]
        b2revb[b] = index of the reverse directed bond of b.
    """

    n_atoms: int = 0
    n_bonds: int = 0
    f_atoms: np.ndarray = field(default_factory=lambda: np.zeros((0, ATOM_FDIM)))
    f_bonds: np.ndarray = field(
        default_factory=lambda: np.zeros((0, ATOM_FDIM + BOND_FDIM))
    )
    a2b: List[List[int]] = field(default_factory=list)
    b2a: List[int] = field(default_factory=list)
    b2revb: List[int] = field(default_factory=list)


class MolecularGraphFeaturizer:
    """
    Convert SMILES strings into MolGraph objects for the D-MPNN.

    Usage
    -----
    featurizer = MolecularGraphFeaturizer()
    mol_graph  = featurizer(smiles)          # single molecule
    mol_graphs = featurizer.batch(smiles_list)  # list of molecules

    Notes
    -----
    This featurizer mirrors what `chemprop.features.MolGraph` does internally.
    If the chemprop library is installed you can simply use:
        from chemprop.features import MolGraph
    and skip this class entirely.
    """

    def __call__(self, smiles: str) -> MolGraph:
        """Featurize a single SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

        graph = MolGraph()
        graph.n_atoms = mol.GetNumAtoms()

        # ---- Atom features ----
        f_atoms_list = [atom_features(a) for a in mol.GetAtoms()]
        graph.f_atoms = np.array(f_atoms_list, dtype=np.float32)

        # ---- Bond features & adjacency ----
        graph.a2b = [[] for _ in range(graph.n_atoms)]
        f_bonds_list = []

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            bf = bond_features(bond)
            # Each undirected bond becomes two directed bonds
            # Chemprop convention: f_bonds[b] = [f_atoms[source] || bond_feats]
            b1_features = f_atoms_list[a1] + bf  # directed bond a1 → a2
            b2_features = f_atoms_list[a2] + bf  # directed bond a2 → a1

            b1 = graph.n_bonds  # a1 → a2
            b2 = graph.n_bonds + 1  # a2 → a1
            graph.a2b[a2].append(b1)
            graph.a2b[a1].append(b2)
            graph.b2a.extend([a1, a2])
            graph.b2revb.extend([b2, b1])
            f_bonds_list.extend([b1_features, b2_features])
            graph.n_bonds += 2

        if f_bonds_list:
            graph.f_bonds = np.array(f_bonds_list, dtype=np.float32)
        else:
            graph.f_bonds = np.zeros((0, ATOM_FDIM + BOND_FDIM), dtype=np.float32)

        return graph

    def batch(self, smiles_list: List[str]) -> List[MolGraph]:
        """Featurize a list of SMILES strings."""
        return [self(s) for s in smiles_list]
