"""
Test configuration for pytest.
Ensures project root is in sys.path for imports.
"""
import pytest
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
