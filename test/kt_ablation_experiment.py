"""Launcher for the DPGA meta-KT ablation experiment (v4.0)."""

import os
import sys


PAPER_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PAPER_CODE_DIR not in sys.path:
    sys.path.insert(0, PAPER_CODE_DIR)

from kt_ablation_experiment import main


if __name__ == "__main__":
    main()
