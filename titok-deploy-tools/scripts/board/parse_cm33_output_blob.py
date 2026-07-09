#!/usr/bin/env python3
"""Compatibility wrapper for `run_board.py parse-cm33-output`."""

from __future__ import annotations

import sys

from run_board import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "parse-cm33-output", *sys.argv[1:]]
    main()
