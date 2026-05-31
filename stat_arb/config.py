"""
Configuration for the cross-exchange statistical arbitrage bot.
All API keys are loaded from environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ExchangeCreds:
    api_key:      str   = ""
    api_secret:   str   = ""
    passphrase:   str   = ""    # OKX only
    taker_fee_pct: float = 0.10  # % taker commission


@dataclass
class ArbConfig:
    # ── Capital & sizing ────────────────────────────────────────────────────────
    capital_usdt:          float = 500.0
    risk_per_trade_pct:    float = 1.0    # % of capital per leg
    max_deployed_pct:      float = 3.0    # % of capital max simultaneous
    default_leverage:      int   = 3
    max_leverage:          int   = 5

    # ── Spread thresholds ────────────────────────────────────────────────────────
    default_min_spread_pct:  float = 0.8
    large_cap_min_spread_pct: float = 0.4
    large_cap_symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT"
    ])

    # ── Entry filters ────────────────────────────────────────────────────────────
    spread_hold_seconds:    float = 5.0
    max_volatility_60s_pct: float = 2.0
    min_liquidity_ratio:    float = 5.0   # orderbook depth >= N × notional
    max_execution_gap_ms:   float = 300.0

    # ── Exit conditions ──────────────────────────────────────────────────────────
    take_profit_ratio:       float = 0.50  # close when spread shrinks to 50% of entry
    aggressive_profit_ratio: float = 0.70  # close when 70% of expected PnL reached
    stop_loss_ratio:         float = 1.50  # close if spread grows to 150% of entry

    # ── Trend filter ─────────────────────────────────────────────────────────────
    trend_filter_enabled:  bool  = True
    ema_short:             int   = 50
    ema_long:              int   = 200
    trend_deviation_pct:   float = 0.2    # |EMA50 - EMA200| / EMA200 > this → trend

    # ── News filter ──────────────────────────────────────────────────────────────
    news_filter_enabled:  bool = True
    news_buffer_minutes:  int  = 15
    news_events_file:     str  = "events.json"

    # ── Symbols ──────────────────────────────────────────────────────────────────
    allowed_symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT",
        "ADAUSDT", "LINKUSDT", "AVAXUSDT", "SUIUSDT", "TRXUSDT",
    ])

    # ── Mode & infra ─────────────────────────────────────────────────────────────
    paper_mode:        bool  = True
    scan_interval_sec: float = 1.0
    telegram_token:    str   = ""
    telegram_chat_id:  str   = ""
    db_path:           str   = "logs/stat_arb.db"

    # ── Exchange credentials (filled by load_config) ──────────────────────────────
    creds: Dict[str, ExchangeCreds] = field(default_factory=dict)

    # ccxt-specific options per exchange
    ccxt_options: Dict[str, dict] = field(default_factory=lambda: {
        "binance": {"options": {"defaultType": "future"}},
        "bybit":   {"options": {"defaultType": "linear"}},
        "okx":     {"options": {"defaultType": "swap"}},
        "mexc":    {"options": {"defaultType": "linear"}},
    })

    def min_spread_for(self, symbol: str) -> float:
        return (self.large_cap_min_spread_pct
                if symbol in self.large_cap_symbols
                else self.default_min_spread_pct)

    def position_notional(self) -> float:
        """Notional per leg in USDT."""
        return self.capital_usdt * self.risk_per_trade_pct / 100 * self.default_leverage

    def max_positions(self) -> int:
        per_pos = self.capital_usdt * self.risk_per_trade_pct / 100
        max_cap = self.capital_usdt * self.max_deployed_pct / 100
        return max(1, int(max_cap / per_pos))


def load_config() -> ArbConfig:
    from dotenv import load_dotenv
    load_dotenv()

    cfg = ArbConfig(
        capital_usdt       = float(os.getenv("ARB_CAPITAL_USDT", "500")),
        paper_mode         = os.getenv("ARB_PAPER_MODE", "true").lower() == "true",
        telegram_token     = os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id   = os.getenv("TELEGRAM_CHAT_ID", ""),
        db_path            = os.getenv("ARB_DB_PATH", "logs/stat_arb.db"),
        default_leverage   = int(os.getenv("ARB_LEVERAGE", "3")),
    )

    for name in ("binance", "bybit", "okx", "mexc"):
        p = name.upper()
        cfg.creds[name] = ExchangeCreds(
            api_key     = os.getenv(f"{p}_API_KEY", ""),
            api_secret  = os.getenv(f"{p}_API_SECRET", ""),
            passphrase  = os.getenv(f"{p}_PASSPHRASE", ""),
            taker_fee_pct = float(os.getenv(f"{p}_TAKER_FEE", "0.10")),
        )

    return cfg
