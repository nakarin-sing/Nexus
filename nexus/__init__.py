"""
NEXUS: Neural EXpert Unified System
Version 4.0.1

A memory-aware online learning algorithm for streaming data.
"""

__version__ = "4.0.1"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Direct imports from core module
from nexus.core import (
    NEXUS_River,
    CONFIG,
    Snapshot,
    safe_div,
    safe_exp,
    safe_std,
    safe_norm,
)

__all__ = [
    "NEXUS_River",
    "CONFIG",
    "Snapshot",
    "safe_div",
    "safe_exp",
    "safe_std",
    "safe_norm",
    "__version__",
]
