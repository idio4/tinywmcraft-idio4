"""Optional inspection tools; never modify saved mechanics outcomes.

The shared confusion helper retains its previous public import:
    from mechanics_eval.audit import confusion
"""
from .reader import confusion

__all__ = ["confusion"]
