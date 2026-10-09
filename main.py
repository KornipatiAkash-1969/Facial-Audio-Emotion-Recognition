#!/usr/bin/env python3
"""
Main Entry Point for the Multimodal Emotion Recognition System.
Launches the full Desktop Graphical User Interface (GUI).

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add project root to sys.path so modules resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gui import launch_app

if __name__ == "__main__":
    launch_app()
