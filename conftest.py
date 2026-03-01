"""Pytest configuration — add python source to sys.path."""
import os
import sys

# Add the python source directory so tests can import gauss.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
