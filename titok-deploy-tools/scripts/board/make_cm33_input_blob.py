#!/usr/bin/env python3
"""Compatibility wrapper for `run_board.py make-cm33-input`."""

from __future__ import annotations

import sys

from run_board import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "make-cm33-input", *sys.argv[1:]]
    main()
