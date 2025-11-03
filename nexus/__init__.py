"""
NEXUS: Neural EXpert Unified System
Version 4.0.1

A memory-aware online learning algorithm for streaming data.
"""

__version__ = "4.0.1"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import core components
try:
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

except ImportError as e:
    import sys
    import warnings
    
    warnings.warn(
        f"Failed to import NEXUS components: {e}\n"
        "Make sure 'nexus.core' module exists and dependencies are installed.",
        ImportWarning
    )
    
    # Define minimal exports for partial imports
    __all__ = ["__version__"]
