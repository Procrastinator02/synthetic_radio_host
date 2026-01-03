#!/usr/bin/env python3
"""
Test runner for Synthetic Radio Host Generator
Run with: python run_tests.py
"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run pytest with coverage and verbose output"""
    # Get the directory containing this script
    test_dir = Path(__file__).parent
    src_dir = test_dir.parent / "src"
    
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir / "test_radio_host.py"),
        "-v",
        "--tb=short",
        f"--cov={src_dir}",
        "--cov=radio_host_functions",
        "--cov-report=term-missing",
        "--cov-report=html"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)

