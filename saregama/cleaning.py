"""Stage 1: Data cleaning and preprocessing.

This module is a simple entry point that calls data_processing functions.
"""

from __future__ import annotations

from .data_processing import run_cleaning


if __name__ == "__main__":
    run_cleaning()
