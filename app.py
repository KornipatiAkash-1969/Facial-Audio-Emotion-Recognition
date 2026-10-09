#!/usr/bin/env python3
"""
Main Entry Point for the Multimodal Emotion Recognition System.
Run this script to launch the graphical user interface:
    python app.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gui import launch_app

if __name__ == "__main__":
    launch_app()
