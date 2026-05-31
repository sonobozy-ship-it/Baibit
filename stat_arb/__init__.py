"""
stat_arb — cross-exchange mean-reversion arbitrage bot.

Usage:
    python -m stat_arb.main
    # or
    python run_stat_arb.py
"""
from .config import ArbConfig, load_config
from .main import run

__all__ = ["ArbConfig", "load_config", "run"]
