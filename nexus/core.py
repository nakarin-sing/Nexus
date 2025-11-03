#!/usr/bin/env python3
"""
Verify nexus/core.py structure
Run this locally to check if core.py is complete
"""

import ast
import sys
from pathlib import Path

def check_core_py():
    """Check if core.py has all required components"""
    
    core_path = Path("nexus/core.py")
    
    if not core_path.exists():
        print("❌ nexus/core.py does not exist!")
        return False
    
    print(f"✓ File exists: {core_path}")
    print(f"  Size: {core_path.stat().st_size:,} bytes")
    print(f"  Lines: {len(core_path.read_text().splitlines()):,}")
    
    content = core_path.read_text()
    
    # Check for syntax errors
    try:
        ast.parse(content)
        print("✓ No syntax errors")
    except SyntaxError as e:
        print(f"❌ Syntax error at line {e.lineno}: {e.msg}")
        return False
    
    # Check required components
    required = {
        "class NEXUS_River": "Main class definition",
        "class Snapshot": "Snapshot dataclass",
        "def predict_one": "predict_one method",
        "def predict_proba_one": "predict_proba_one method",
        "def learn_one": "learn_one method",
        "def _predict_ncra": "NCRA prediction method",
        "CONFIG": "Config object",
        "safe_div": "safe_div function",
        "safe_exp": "safe_exp function",
        "safe_std": "safe_std function",
        "safe_norm": "safe_norm function",
    }
    
    print("\nChecking required components:")
    missing = []
    
    for component, description in required.items():
        if component in content:
            print(f"  ✓ {component:30s} — {description}")
        else:
            print(f"  ❌ {component:30s} — {description} (MISSING!)")
            missing.append(component)
    
    # Check imports
    print("\nChecking imports:")
    required_imports = [
        "from river.base import Classifier",
        "import numpy as np",
        "from typing import Dict",
        "from dataclasses import dataclass",
    ]
    
    for imp in required_imports:
        if imp in content:
            print(f"  ✓ {imp}")
        else:
            print(f"  ❌ {imp} (MISSING!)")
            missing.append(imp)
    
    if missing:
        print(f"\n❌ Missing {len(missing)} components!")
        print("\n💡 Solution: core.py is incomplete. Copy the full file from artifacts.")
        return False
    else:
        print("\n✅ All components present!")
        
        # Try importing
        print("\nTrying to import...")
        try:
            sys.path.insert(0, ".")
            from nexus.core import NEXUS_River, CONFIG, Snapshot
            print("✓ Import successful!")
            print(f"  NEXUS_River: {NEXUS_River}")
            print(f"  CONFIG: {CONFIG}")
            return True
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False

if __name__ == "__main__":
    print("="*70)
    print("NEXUS core.py Verification")
    print("="*70 + "\n")
    
    if not Path("nexus").exists():
        print("❌ Not in repository root! Run from Nexus/ directory")
        sys.exit(1)
    
    success = check_core_py()
    print("\n" + "="*70)
    
    if success:
        print("✅ core.py is complete and valid!")
    else:
        print("❌ core.py has issues — needs to be fixed")
    
    print("="*70)
    sys.exit(0 if success else 1)
