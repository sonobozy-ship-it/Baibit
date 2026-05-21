"""
Pre-Launch Audit Script for Baibit Trading Bot
Runs from /home/user/Baibit/bybit-bot-v5/backend/
"""
import sys
import os
import subprocess
import traceback
import importlib
import stat
from pathlib import Path

# Use paths relative to this script's directory
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
os.chdir(str(BASE_DIR))

import pandas as pd
import numpy as np

results = []

def ok(check, detail=""):
    results.append(("✅", check, detail))
    print(f"✅ {check}" + (f" — {detail}" if detail else ""))

def err(check, detail=""):
    results.append(("❌", check, detail))
    print(f"❌ {check} — {detail}")

def make_ohlcv(n=250, price=45000, interval_min=15, seed=42):
    np.random.seed(seed)
    t = pd.date_range("2024-01-01", periods=n, freq=f"{interval_min}min")
    close = price + np.cumsum(np.random.randn(n) * 50)
    high  = close + abs(np.random.randn(n) * 30)
    low   = close - abs(np.random.randn(n) * 30)
    open_ = close - np.random.randn(n) * 20
    vol   = np.random.uniform(1000, 5000, n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": vol}, index=t)
    # Ensure high >= close >= low
    df["high"] = df[["open","high","close"]].max(axis=1) + 5
    df["low"]  = df[["open","low","close"]].min(axis=1) - 5
    return df

# ============================================================
# CHECK 1: Syntax check all .py files
# ============================================================
print("\n" + "="*60)
print("CHECK 1: Syntax check all .py files")
print("="*60)

import glob
py_files = glob.glob(str(BASE_DIR / '**' / '*.py'), recursive=True)
syntax_errors = []
for f in sorted(py_files):
    try:
        with open(f) as fh:
            source = fh.read()
        compile(source, f, 'exec')
    except SyntaxError as e:
        syntax_errors.append(f"{f}: {e}")

if not syntax_errors:
    ok("Syntax check", f"All {len(py_files)} .py files pass")
else:
    for e in syntax_errors:
        err("Syntax check", e)

# ============================================================
# CHECK 2: Import check
# ============================================================
print("\n" + "="*60)
print("CHECK 2: Import check")
print("="*60)

modules_to_test = [
    ("pandas_ta",               "pandas_ta"),
    ("strategies.base",         "from strategies.base import BaseStrategy, TradingSignal"),
    ("strategies.all_strategies","from strategies.all_strategies import ALL_STRATEGIES"),
    ("strategies.fusion",       "from strategies.fusion import StrategyFusion, SignalBuffer, FusedSignal"),
    ("strategies.scalper_pro",  "from strategies.scalper_pro import ScalperProStrategy"),
    ("strategies.trend_fib",    "from strategies.trend_fib import TrendMomentumStrategy, TrendFibonacciStrategy"),
    ("boost_mode",              "from boost_mode import BoostManager, BoostCalculator"),
    ("risk_manager",            "from risk_manager import RiskManager"),
    ("paper_trader",            "from paper_trader import PaperTrader"),
    ("claude_orchestrator",     "from claude_orchestrator import ClaudeOrchestrator"),
    ("ai_analyzer",             "from ai_analyzer import AIAnalyzer"),
    ("position_calc",           "from position_calc import calculate_pnl, fixed_risk_position_size"),
    ("correlation_filter",      "from correlation_filter import CorrelationFilter"),
    ("trade_journal",           "from trade_journal import TradeJournal"),
    ("db_pool",                 "from db_pool import DBPool"),
    ("ml",                      "from ml import FeatureExtractor, MLDataStore, MLTrainer, MLPredictor, RegimeClassifier, AutoOptimizer, ML_AVAILABLE, DriftMonitor, AnomalyDetector, EnsembleTrainer, ContinuousTrainer"),
    ("news",                    "from news import NewsManager, background_news_loop"),
]

for name, stmt in modules_to_test:
    try:
        exec(stmt)
        ok(f"Import: {name}")
    except Exception as e:
        err(f"Import: {name}", str(e))

# ============================================================
# CHECK 3: Test all 11 strategies S1-S11
# ============================================================
print("\n" + "="*60)
print("CHECK 3: Test all 11 strategies")
print("="*60)

from strategies.all_strategies import ALL_STRATEGIES

df15 = make_ohlcv(250, 45000, 15)   # 15-min candles
df60 = make_ohlcv(250, 45000, 60)   # H1 candles for S11

for sid, cls in ALL_STRATEGIES.items():
    try:
        strat = cls(symbol="BTCUSDT")
        # S11 uses H1 timeframe
        df_use = df60 if sid == "S11" else df15
        result = strat.analyze(df_use.copy())
        # result can be None (no signal) or a TradingSignal — both are valid
        sig_info = f"signal={result.action}" if result else "no signal (OK)"
        ok(f"Strategy {sid} ({cls.__name__})", sig_info)
    except Exception as e:
        err(f"Strategy {sid} ({cls.__name__})", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-300:]}")

# ============================================================
# CHECK 4: StrategyFusion — all 11 strategies in INDICATOR_GROUPS
# ============================================================
print("\n" + "="*60)
print("CHECK 4: StrategyFusion INDICATOR_GROUPS")
print("="*60)

from strategies.fusion import StrategyFusion

expected_sids = set(ALL_STRATEGIES.keys())
groups_sids = set(StrategyFusion.INDICATOR_GROUPS.keys())
missing = expected_sids - groups_sids
extra   = groups_sids - expected_sids

if not missing:
    ok("StrategyFusion INDICATOR_GROUPS coverage", f"All {len(expected_sids)} strategies present: {sorted(groups_sids)}")
else:
    err("StrategyFusion INDICATOR_GROUPS missing", f"Missing: {missing}")
if extra:
    err("StrategyFusion INDICATOR_GROUPS extra", f"Unexpected extra keys: {extra}")

# ============================================================
# CHECK 5: Test all 11 pandas_ta indicators
# ============================================================
print("\n" + "="*60)
print("CHECK 5: pandas_ta indicators")
print("="*60)

import pandas_ta as ta

df = make_ohlcv(250, 45000, 15)

indicators = {
    "ema":         lambda: ta.ema(df["close"], length=9),
    "rsi":         lambda: ta.rsi(df["close"], length=14),
    "atr":         lambda: ta.atr(df["high"], df["low"], df["close"], length=14),
    "bbands":      lambda: ta.bbands(df["close"], length=20, std=2),
    "macd":        lambda: ta.macd(df["close"], fast=12, slow=26, signal=9),
    "stoch":       lambda: ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3),
    "adx":         lambda: ta.adx(df["high"], df["low"], df["close"], length=14),
    "obv":         lambda: ta.obv(df["close"], df["volume"]),
    "supertrend":  lambda: ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3),
    "ichimoku":    lambda: ta.ichimoku(df["high"], df["low"], df["close"], tenkan=9, kijun=26, senkou=52),
    "psar":        lambda: ta.psar(df["high"], df["low"], df["close"], af0=0.02, af_step=0.02, max_af=0.2),
}

for name, func in indicators.items():
    try:
        result = func()
        if result is None:
            err(f"pandas_ta: {name}", "returned None")
        elif isinstance(result, (pd.DataFrame, pd.Series)):
            if isinstance(result, pd.DataFrame):
                ok(f"pandas_ta: {name}", f"DataFrame cols={list(result.columns)[:4]}")
            else:
                ok(f"pandas_ta: {name}", f"Series len={len(result)}, last={result.dropna().iloc[-1]:.4f}")
        else:
            ok(f"pandas_ta: {name}", f"type={type(result)}")
    except Exception as e:
        err(f"pandas_ta: {name}", f"{type(e).__name__}: {e}")

# ============================================================
# CHECK 6: BoostManager — start/stop all 4 modes, S11 in phases
# ============================================================
print("\n" + "="*60)
print("CHECK 6: BoostManager")
print("="*60)

from boost_mode import BoostManager

modes = ["safe", "moderate", "aggressive", "scalp"]

from boost_mode import PHASES_AGGRESSIVE, PHASES_MODERATE, PHASES_SAFE, PHASES_SCALP
phases_map = {
    "aggressive": PHASES_AGGRESSIVE,
    "moderate": PHASES_MODERATE,
    "safe": PHASES_SAFE,
    "scalp": PHASES_SCALP,
}

for mode in modes:
    try:
        bm = BoostManager()
        result = bm.start(initial_balance=100.0, target_balance=500.0, deadline_days=30, mode=mode)
        if result.get("success"):
            phase = bm.session.phase
            bm.stop()
            phases = phases_map.get(mode, [])
            s11_phases = [p.name for p in phases if "S11" in p.allowed_strategies]
            ok(f"BoostManager: {mode}",
               f"started/stopped OK, phase={phase.name}, S11 in phases: {s11_phases}")
        else:
            err(f"BoostManager: {mode}", f"start returned: {result}")
    except Exception as e:
        err(f"BoostManager: {mode}", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-300:]}")

# ============================================================
# CHECK 7: RiskManager
# ============================================================
print("\n" + "="*60)
print("CHECK 7: RiskManager")
print("="*60)

from risk_manager import RiskManager

try:
    rm = RiskManager(
        daily_max_loss_pct=5.0,
        max_open_positions=4,
        risk_per_trade_pct=1.0,
        cooldown_after_loss_min=15,
    )
    ok("RiskManager: init")
except Exception as e:
    err("RiskManager: init", str(e))

try:
    can, reason = rm.can_open_trade(symbol="BTCUSDT", balance=1000.0)
    ok("RiskManager: can_open_trade", f"can={can}, reason='{reason}'")
except Exception as e:
    err("RiskManager: can_open_trade", str(e))

try:
    rm.register_trade_result(
        symbol="BTCUSDT",
        pnl_pct=2.5,
        is_win=True,
        strategy_id="S1",
    )
    ok("RiskManager: register_trade_result", "win trade registered")
except Exception as e:
    err("RiskManager: register_trade_result", str(e))

try:
    rm.register_trade_result(
        symbol="BTCUSDT",
        pnl_pct=-1.2,
        is_win=False,
        strategy_id="S2",
    )
    ok("RiskManager: register_trade_result", "loss trade registered")
except Exception as e:
    err("RiskManager: register_trade_result", str(e))

try:
    status = rm.get_status()
    ok("RiskManager: get_status", f"keys={list(status.keys())[:6]}")
except Exception as e:
    err("RiskManager: get_status", str(e))

# ============================================================
# CHECK 8: ClaudeOrchestrator
# ============================================================
print("\n" + "="*60)
print("CHECK 8: ClaudeOrchestrator")
print("="*60)

try:
    from claude_orchestrator import ClaudeOrchestrator
    orch = ClaudeOrchestrator(enabled=False)
    ok("ClaudeOrchestrator: init(enabled=False)")
except Exception as e:
    err("ClaudeOrchestrator: init(enabled=False)", str(e))

# Test 3 provider imports
providers = [
    ("anthropic",  "import anthropic"),
    ("openai",     "import openai"),
    ("httpx",      "import httpx"),  # used for Ollama
]
for name, stmt in providers:
    try:
        exec(stmt)
        ok(f"Provider import: {name}")
    except ImportError as e:
        err(f"Provider import: {name}", str(e))
    except Exception as e:
        err(f"Provider import: {name}", str(e))

# ============================================================
# CHECK 9: PaperTrader
# ============================================================
print("\n" + "="*60)
print("CHECK 9: PaperTrader")
print("="*60)

from paper_trader import PaperTrader

try:
    pt = PaperTrader()
    ok("PaperTrader: init", f"balance={pt.balance}, trades={len(pt.positions)}")
except Exception as e:
    err("PaperTrader: init", str(e))

try:
    result = pt.open_position(
        symbol="BTCUSDT",
        side="BUY",
        entry_price=45000.0,
        size=0.01,
        stop_loss=44100.0,
        take_profit=47000.0,
        strategy_id="S1",
    )
    ok("PaperTrader: open_position", f"result={result}")
except Exception as e:
    err("PaperTrader: open_position", str(e))

try:
    status = pt.get_status()
    ok("PaperTrader: get_status", f"keys={list(status.keys())[:6]}")
except Exception as e:
    err("PaperTrader: get_status", str(e))

# ============================================================
# CHECK 10: main.py references
# ============================================================
print("\n" + "="*60)
print("CHECK 10: main.py references")
print("="*60)

main_py = open(str(BASE_DIR / 'main.py')).read()

checks_main = {
    "ALL_STRATEGIES":    "ALL_STRATEGIES" in main_py,
    "BoostManager":      "BoostManager" in main_py,
    "RiskManager":       "RiskManager" in main_py,
    "ClaudeOrchestrator":"ClaudeOrchestrator" in main_py,
    "OPENAI_API_KEY":    "OPENAI_API_KEY" in main_py,
    "OLLAMA_BASE_URL":   "OLLAMA_BASE_URL" in main_py,
}

for key, found in checks_main.items():
    if found:
        ok(f"main.py references: {key}")
    else:
        err(f"main.py references: {key}", "NOT FOUND in main.py")

# ============================================================
# CHECK 11: deploy.sh exists and is executable
# ============================================================
print("\n" + "="*60)
print("CHECK 11: deploy.sh")
print("="*60)

deploy_paths = [
    str(BASE_DIR.parent / "deploy.sh"),
    str(BASE_DIR.parent.parent / "deploy.sh"),
]

for dpath in deploy_paths:
    if os.path.exists(dpath):
        fstat = os.stat(dpath)
        is_exec = bool(fstat.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        if is_exec:
            ok(f"deploy.sh: {dpath}", "exists and is executable")
        else:
            err(f"deploy.sh: {dpath}", "exists but NOT executable")
    else:
        err(f"deploy.sh: {dpath}", "does not exist")

# ============================================================
# CHECK 12: .gitignore coverage
# ============================================================
print("\n" + "="*60)
print("CHECK 12: .gitignore coverage")
print("="*60)

gitignore_path = str(BASE_DIR.parent.parent / ".gitignore")
try:
    gitignore = open(gitignore_path).read()
    runtime_checks = {
        ".env":               ".env" in gitignore,
        "*.log / logs/":      "*.log" in gitignore or "logs/" in gitignore,
        "__pycache__":        "__pycache__" in gitignore,
        "*.pkl (ML models)":  "*.pkl" in gitignore,
        "*.parquet":          "*.parquet" in gitignore,
        "data/ml_data.db":    "ml_data.db" in gitignore,
        "boost_session.json": "boost_session.json" in gitignore,
    }
    for item, covered in runtime_checks.items():
        if covered:
            ok(f".gitignore covers: {item}")
        else:
            err(f".gitignore missing: {item}")
except Exception as e:
    err(".gitignore", str(e))

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY TABLE")
print("="*60)

passes = sum(1 for r in results if r[0] == "✅")
failures = sum(1 for r in results if r[0] == "❌")

print(f"\nTotal checks: {len(results)}")
print(f"  ✅ PASS: {passes}")
print(f"  ❌ FAIL: {failures}")

if failures > 0:
    print("\nFailed checks:")
    for r in results:
        if r[0] == "❌":
            print(f"  ❌ {r[1]}: {r[2]}")

print("\n" + "="*60)
if failures == 0:
    print("OVERALL: ✅ PASS — All checks passed, bot is ready for launch!")
else:
    print(f"OVERALL: ❌ FAIL — {failures} check(s) failed, fix before launch.")
print("="*60)
