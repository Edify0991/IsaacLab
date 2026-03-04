"""Factorized Manifold Motion Prior (FMMP) research prototype package.

The modules in this package are intentionally lightweight so they can be integrated
incrementally into Isaac Lab AMP workflows without breaking baseline training.
"""

from .manifold_encoders import FMMPEncoderBank, PartManifoldEncoder
from .priors import FMMPPrior
from .transition_generator import TransitionGenerator

__all__ = ["FMMPEncoderBank", "FMMPPrior", "PartManifoldEncoder", "TransitionGenerator"]
