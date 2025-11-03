#!/usr/bin/env python3
"""
Diagnostic script to check NEXUS repository structure
Run this locally to identify missing files
"""

import os
from pathlib import Path
from typing import List, Tuple

def check_file_structure() -> List[Tuple[str, bool, str]]:
    """Check if all required files exist"""
    
    required_files = [
        # Core package
        ("nexus/__init__.py", True, "Package initialization"),
        ("nexus/core.py", True, "Main NEXUS implementation"),
        
        # Tests
        ("tests/__init__.py", True, "Test package init"),
        ("tests/test_nexus.py", True, "Unit tests"),
        ("tests/test_integration.py", True, "Integration tests"),
        
        # Examples
        ("examples/__init__.py", False, "Examples package init"),
        ("examples/quickstart.py", False, "Usage examples"),
        
        # Config files
        ("setup.py", True, "Package setup"),
        ("pyproject.toml", True, "Modern Python config"),
        ("requirements.txt", True, "Dependencies"),
        ("README.md", True, "Documentation"),
        ("LICENSE", False, "License file"),
        
        # GitHub
        (".github/workflows/test.yml", True, "CI/CD pipeline"),
        (".gitignore", True, "Git ignore patterns"),
    ]
    
    results = []
    for filepath, required, description in required_files:
        exists = Path(filepath).exists()
        status = "✓" if exists else ("✗" if required else "○")
        results.append((filepath, exists, status, description, required))
    
    return results

def print_results(results: List[Tuple[str, bool, str, str, bool]]) -> None:
    """Print check results"""
    
    print("\n" + "="*70)
    print("NEXUS Repository Structure Check")
    print("="*70 + "\n")
    
    missing_required = []
    missing_optional = []
    
    for filepath, exists, status, description, required in results:
        print(f"{status} {filepath:40s} {description}")
        
        if not exists:
            if required:
                missing_required.append(filepath)
            else:
                missing_optional.append(filepath)
    
    print("\n" + "="*70)
    
    if missing_required:
        print("\n❌ MISSING REQUIRED FILES:")
        for f in missing_required:
            print(f"   - {f}")
    
    if missing_optional:
        print("\n⚠️  MISSING OPTIONAL FILES:")
        for f in missing_optional:
            print(f"   - {f}")
    
    if not missing_required:
        print("\n✅ All required files present!")
    else:
        print("\n💡 Fix: Create missing files or check repository structure")
    
    print("="*70 + "\n")

def check_imports() -> None:
    """Try importing the package"""
    
    print("="*70)
    print("Import Check")
    print("="*70 + "\n")
    
    try:
        import nexus
        print(f"✓ nexus imported successfully")
        print(f"  Version: {nexus.__version__}")
        print(f"  Available: {', '.join(nexus.__all__)}")
    except ImportError as e:
        print(f"✗ Failed to import nexus: {e}")
        return
    
    try:
        from nexus import NEXUS_River
        print(f"✓ NEXUS_River imported successfully")
        model = NEXUS_River()
        print(f"  Model created: {model}")
    except ImportError as e:
        print(f"✗ Failed to import NEXUS_River: {e}")
    
    print()

def main():
    """Run all checks"""
    
    # Check if we're in the right directory
    if not Path("nexus").exists():
        print("\n❌ ERROR: Not in NEXUS repository root!")
        print("   Run this script from the repository root directory\n")
        return
    
    results = check_file_structure()
    print_results(results)
    
    # Try importing
    try:
        check_imports()
    except Exception as e:
        print(f"\n⚠️  Import check skipped: {e}\n")

if __name__ == "__main__":
    main()
