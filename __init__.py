"""
OPOCH CasADi - Precision Refinement for Safety-Critical Optimization

This package provides KKT verification and refinement for CasADi/IPOPT solutions.

Key Scripts:
    python precision_comparison.py  - See precision improvement table
    python run_all.py               - Run all 27 benchmark problems
    python show_ipopt_failures.py   - Demo: IPOPT lies caught and fixed

Core Module:
    kkt_verifier.py - KKT verification with correct IPOPT sign convention
"""

from .kkt_verifier import (
    compute_kkt_residuals,
    verify_casadi_solution,
    certify_or_repair,
    split_multipliers,
    KKTResiduals,
)

from .nlp_contract import (
    CasADiNLP,
    NLPBounds,
    create_nlp_from_casadi,
)

__version__ = "0.1.0"
__all__ = [
    "compute_kkt_residuals",
    "verify_casadi_solution",
    "certify_or_repair",
    "split_multipliers",
    "KKTResiduals",
    "CasADiNLP",
    "NLPBounds",
    "create_nlp_from_casadi",
]
