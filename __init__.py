# ==================== nexus/__init__.py ====================
"""
NEXUS: Neural EXpert Unified System
Version 4.0.1

A memory-aware online learning algorithm for streaming data.
"""

from nexus.core import NEXUS_River, CONFIG, Snapshot
from nexus.core import (
    safe_div,
    safe_exp,
    safe_std,
    safe_norm,
)

__version__ = "4.0.1"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "NEXUS_River",
    "CONFIG",
    "Snapshot",
    "safe_div",
    "safe_exp",
    "safe_std",
    "safe_norm",
]
