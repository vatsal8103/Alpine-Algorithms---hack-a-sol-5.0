"""TRACE-X Ingestion & Preparation Pipeline (try/download.py mirror)."""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from scripts.download_data import main

if __name__ == "__main__":
    main()
