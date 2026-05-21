"""
Centralized ENV-based configuration for Baibit trading bot.
All tunable parameters in one place — override via environment variables.
"""
import os


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


def _bool(key: str, default: bool) -> bool:
    v = os.getenv(key, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on")


# ── Position Monitor ─────────────────────────────────────────────────────────
MONITOR_INTERVAL_SECONDS: int   = _int("MONITOR_INTERVAL_SECONDS", 5)

# Close at X% of planned TP profit (USDT)
TP_CLOSE_PERCENT: float         = _float("TP_CLOSE_PERCENT", 80.0)

# Arm trailing profit protection after reaching this PnL (USDT)
MIN_PROFIT_ARM_USDT: float      = _float("MIN_PROFIT_ARM_USDT", 0.75)

# Close if PnL drops X% from its max after being armed
TRAILING_DROP_PERCENT: float    = _float("TRAILING_DROP_PERCENT", 45.0)

# Close if price has travelled X% of the way to TP
CLOSE_NEAR_TP: float            = _float("CLOSE_NEAR_TP", 90.0)

# Close timed-out positions if PnL < this (USDT)
MAX_POSITION_MINUTES: int       = _int("MAX_POSITION_MINUTES", 360)
MIN_EXPECTED_PROFIT: float      = _float("MIN_EXPECTED_PROFIT", 0.10)

# Anti profit-return: if PnL was > this threshold, then drops X% → close
ANTI_RETURN_ARM_USDT: float     = _float("ANTI_RETURN_ARM_USDT", 1.00)
ANTI_RETURN_DROP_PCT: float     = _float("ANTI_RETURN_DROP_PCT", 50.0)

# ── Risk Management ──────────────────────────────────────────────────────────
MAX_RISK_PER_TRADE_USDT: float  = _float("MAX_RISK_PER_TRADE_USDT", 5.0)
MAX_LEVERAGE: int               = _int("MAX_LEVERAGE", 5)
MAX_CONSECUTIVE_LOSSES: int     = _int("MAX_CONSECUTIVE_LOSSES", 3)
COOLDOWN_MINUTES: int           = _int("COOLDOWN_MINUTES", 60)

# ── Signal Quality ───────────────────────────────────────────────────────────
MIN_SIGNAL_CONFIDENCE: float    = _float("MIN_SIGNAL_CONFIDENCE", 0.65)
