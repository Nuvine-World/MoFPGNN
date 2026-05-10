# chemff/chemprop/__init__.py
# Chemprop D-MPNN module for LANTERN chemff pipeline.
#
# This subpackage provides:
#   - MolecularGraphFeaturizer : converts SMILES → atom/bond feature tensors
#   - ChempropEncoder          : D-MPNN message-passing encoder (produces fixed-dim embedding)
#
# Usage:
#   from chemff.chemprop import MolecularGraphFeaturizer, ChempropEncoder

from .featurizer import MolecularGraphFeaturizer
from .model import ChempropEncoder

__all__ = ["MolecularGraphFeaturizer", "ChempropEncoder"]
