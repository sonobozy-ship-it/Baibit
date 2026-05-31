#!/usr/bin/env python3
"""Entry point: python run_stat_arb.py"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from stat_arb.main import main

if __name__ == "__main__":
    main()
