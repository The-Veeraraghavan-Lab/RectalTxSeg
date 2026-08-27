"""Compatibility entry point for folder inference.

Prefer ``run_folder_inference.py`` for new commands. This file is kept so older
single-case invocations still reach the folder-capable helper.
"""

import sys
from pathlib import Path

# Allow `python analysis/preprocessing/run_single_case_pipeline.py` from the repo
# root, where this file's directory is not already on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_folder_inference import main  # noqa: E402


if __name__ == "__main__":
    main()
