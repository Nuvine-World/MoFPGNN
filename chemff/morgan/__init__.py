# chemff/morgan/__init__.py
# Morgan fingerprint module for LANTERN chemff pipeline.
#
# This subpackage provides:
#   - MorganFeaturizer : converts SMILES → Morgan (circular) fingerprint vectors
#
# Usage:
#   from chemff.morgan import MorganFeaturizer

from .featurizer import MorganFeaturizer

__all__ = ["MorganFeaturizer"]
