# SYSTEM PROMPT: Full Strategy Engine Rewrite — Baibit Trading Bot

## ROLE
You are a senior quant trader + ML engineer + HFT architect working on a live crypto futures trading bot.

## REPO CONTEXT
- Backend: `bybit-bot-v5/backend/`
- Language: Python 3.11, async FastAPI + asyncio trading loop
- Exchange: Bybit (paper + real via BybitClient)
- DB: MySQL (`pymysql.DictCursor`) — **never use pd.read_sql with this connection**
- Candles: `df` = pandas DataFrame with columns `[timestamp, open, high, low, close, volume]`
- All indicators must be computed with **pure numpy/pandas** — no TA-Lib, no pandas_ta imports unless already present in `backend/pandas_ta.py`

## FILES TO MODIFY
```
bybit-bot-v5/backend/strategies/
  base.py              ← extend BaseStrategy / TradingSignal
  all_strategies.py    ← S1, S2, S5, S7, S13, S14
  aggressive_momentum.py ← S15
  entry_filter.py      ← quality score + universal pre-trade filters
  tp_normalizer.py     ← smart TP/SL logic
bybit-bot-v5/backend/
  risk_manager.py      ← position sizing, daily loss limit, series risk
  trade_journal.py     ← logging schema extension
  main.py              ← cooldown logic, max open positions, late entry guard
```

## STRATEGY → SYMBOL MAP (do not change)
```python
"S1":  "BTCUSDT"       # EMACrossoverStrategy,   15m
"S2":  "ETHUSDT"       # BollingerBandsStrategy, 15m
"S5":  "DOGEUSDT"      # ScalperGridStrategy,    15m  ← PRIORITY
"S7":  "LINKUSDT"      # MultiConfirmStrategy,   15m
"S13": "1000PEPEUSDT"  # OverboughtShortStrategy, 15m
"S14": "WIFUSDT"       # OverboughtShortStrategy (aggressive), 15m
"S15": "OPUSDT"        # AggressiveMomentumStrategy, 1m
```

---

## PART 1 — UNIVERSAL PRE-TRADE METRIC BLOCK

Add function `compute_market_context(df: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict`
to `entry_filter.py`. Must return:

```python
{
  "trend_strength":         float,   # EMA20 slope normalised by ATR, range 0-1
  "adx":                    float,   # ADX(14)
  "atr":                    float,   # ATR(14) current
  "atr50":                  float,   # ATR(50) baseline
  "atr_ratio":              float,   # atr / atr50
  "rsi":                    float,   # RSI(14)
  "macd_hist":              float,   # MACD histogram (12,26,9)
  "ema20":                  float,
  "ema50":                  float,
  "ema200":                 float,
  "volume_ratio":           float,   # last candle volume / 20-bar avg
  "support":                float,   # low of last 8-12 candles ± 0.15-0.25 ATR
  "resistance":             float,   # high of last 8-12 candles ± 0.15-0.25 ATR
  "dist_to_support_atr":    float,   # (close - support) / atr
  "dist_to_resistance_atr": float,   # (resistance - close) / atr
  "market_regime":          str,     # "uptrend" | "downtrend" | "volatile" | "flat"
  "momentum_score":         float,   # composite 0-1: RSI+MACD+vol momentum
  "ema_slope_ok":           bool,    # EMA20 slope > threshold
}
```

### Level detection rules
- `support`    = min(low[-10:-1]) adjusted by +0.20×ATR (never more than 0.5×ATR from raw min)
- `resistance` = max(high[-10:-1]) adjusted by -0.20×ATR (never more than 0.5×ATR from raw max)
- If df has < 15 rows → return None (insufficient data)
- Levels must stay within 3×ATR of current price — discard otherwise

### Market regime rules
```
uptrend:   EMA20 > EMA50 > EMA200  AND  ADX > 18
downtrend: EMA20 < EMA50 < EMA200  AND  ADX > 18
volatile:  ATR/ATR50 > 1.5         OR   ADX > 35
flat:      ADX < 18  AND  ATR/ATR50 < 1.0
```

---

## PART 2 — QUALITY SCORE

Add `compute_quality_score(ctx: dict) -> float` to `entry_filter.py`.

```python
score = (
    ctx["trend_strength"]      * 25 +   # 0-1 → 0-25
    min(ctx["volume_ratio"]-1, 1) * 15 + # extra vol above 1× baseline
    min(ctx["atr_ratio"], 2)/2 * 15 +   # ATR expansion vs baseline
    (1 - min(dist_to_level/3, 1)) * 20 + # closeness to support/resistance
    rsi_quality * 10 +                   # RSI in 35-65 zone=1, extremes=0.5
    regime_bonus * 15                    # flat=0, trending=1, volatile=0.7
)
# score range 0-100
```

Where:
- `dist_to_level` = min(dist_to_support_atr, dist_to_resistance_atr)
- `rsi_quality`: RSI 35-65 → 1.0 ; RSI 25-35 or 65-75 → 0.7 ; else → 0.3
- `regime_bonus`: "uptrend"/"downtrend" → 1.0 ; "volatile" → 0.7 ; "flat" → 0.0

---

## PART 3 — UNIVERSAL FILTERS (apply in every strategy before signal)

Implement as `UniversalFilters.check(ctx: dict, side: str) -> tuple[bool, str]` in `entry_filter.py`:

```
QUALITY:  score < 70           → reject "quality_score_low"
ATR:      atr < atr50 * 0.8   → reject "atr_too_low"
VOLUME:   volume_ratio < 1.15 → reject "volume_insufficient"
REGIME:   flat market          → reject "flat_market"
LATE_ENTRY: signal_candles_ago > 3 → reject "late_entry"
```

Return `(True, "ok")` if all pass, else `(False, reason)`.

---

## PART 4 — SMART SL/TP

Replace `tp_normalizer.py` with `SmartLevels` class:

```python
class SmartLevels:
    @staticmethod
    def compute_sl(entry: float, side: str, ctx: dict) -> float:
        """
        SL = max(1.1×ATR distance, nearest swing extreme within 2×ATR).
        For BUY:  sl = min(entry - 1.1*atr, support - 0.1*atr)
        For SELL: sl = max(entry + 1.1*atr, resistance + 0.1*atr)
        SL must never be < 0.4% from entry (slippage floor).
        SL must never be > 2.5% from entry (risk cap).
        """

    @staticmethod
    def compute_tp(entry: float, side: str, ctx: dict, min_rr: float = 1.8) -> float:
        """
        TP = max(1.8×ATR, nearest liquidity level, next resistance/support).
        For BUY:  tp = max(entry + 1.8*atr, resistance)
        For SELL: tp = min(entry - 1.8*atr, support)
        RR = (tp-entry)/(entry-sl) must be >= min_rr, else scale tp outward.
        TP must never be < 0.7% from entry.
        TP must never be > 5.0% from entry.
        """

    @staticmethod
    def should_close_early(ctx: dict, entry: float, current: float,
                           tp: float, side: str) -> tuple[bool, str]:
        """
        Close early if:
        - reached 50% of TP distance AND momentum_score < 0.35
        Returns (True, reason) or (False, "")
        """

    @staticmethod
    def update_trailing(current: float, entry: float, tp: float,
                        current_sl: float, side: str, atr: float) -> float:
        """
        Trailing SL activated after BE:
        Step = 0.4×ATR.
        For BUY:  new_sl = current - 0.4*atr  (only if > current_sl)
        For SELL: new_sl = current + 0.4*atr  (only if < current_sl)
        """
```

---

## PART 5 — RISK MANAGER UPGRADES

Extend `RiskManager` in `risk_manager.py`:

```python
# New fields
self.daily_pnl: float = 0.0
self.daily_loss_limit_pct: float = 5.0   # stop all trading if hit
self.max_open_positions: int = 3
self.base_risk_pct: float = 1.0          # risk per trade (% of balance)
self.current_risk_pct: float = 1.0       # adjusted dynamically

# New method: dynamic risk scaling
def get_risk_pct(self, consecutive_losses: int, consecutive_wins: int) -> float:
    """
    consecutive_losses >= 4 → 0.5× base (50%)
    consecutive_losses == 3 → 0.5× base
    consecutive_wins   >= 5 → 0.75× base (25% reduction, avoid overconfidence)
    else                    → 1.0× base
    """

# New method: daily loss guard
def check_daily_loss(self, balance: float) -> bool:
    """Returns True if daily loss > daily_loss_limit_pct → stop all trading."""

# Update calculate_position_size to use get_risk_pct()
```

Leverage cap: **never exceed 5×** regardless of strategy setting.

---

## PART 6 — COOLDOWN LOGIC

Add to `main.py` trading loop (per-strategy cooldown dict `_cooldowns: dict[str, datetime]`):

```python
# After each SL hit, record consecutive SL count per strategy
# consecutive_sl[sid] += 1
# 2 SL → cooldown 20 min
# 3 SL → cooldown 40 min
# 4 SL → auto_disable strategy (strat.auto_disabled = True)
# Winning trade → reset consecutive_sl[sid] = 0
```

Check before any trade attempt:
```python
if sid in _cooldowns and datetime.utcnow() < _cooldowns[sid]:
    continue  # skip — in cooldown
```

---

## PART 7 — MAX OPEN POSITIONS GUARD

In trading loop, before opening:
```python
open_count = sum(1 for s in state.strategies.values() if s.current_position)
if open_count >= state.risk_manager.max_open_positions:
    continue  # do not open new position
```

---

## PART 8 — INDIVIDUAL STRATEGY SPECS

### S1 — EMACrossoverStrategy (BTCUSDT, 15m)

**LONG conditions:**
1. `ema20 > ema50 > ema200`
2. `adx > 22`
3. `volume_ratio > 1.2`
4. EMA20 slope > 0 (angle > 0.1 × ATR per candle)
5. Quality score ≥ 70

**SHORT conditions:**
1. `ema20 < ema50 < ema200`
2. `adx > 22`
3. `volume_ratio > 1.2`
4. EMA20 slope < 0

**REJECT:**
- `market_regime == "flat"` → no trade
- EMA20 slope weak (< 0.05 × ATR/candle) → no trade
- Late entry (> 3 candles after crossover) → no trade

SL/TP: use `SmartLevels`. Leverage: 5. Timeframe: 15m.

---

### S2 — BollingerBandsStrategy (ETHUSDT, 15m)

**LONG conditions:**
1. Close touches or crosses below lower BB (within 0.3 × ATR)
2. `rsi < 32`
3. `volume_ratio > 1.1`
4. `dist_to_support_atr < 1.5` (price near support)
5. Market regime is NOT downtrend (mean reversion only in ranging/volatile)

**SHORT conditions:**
1. Close touches or crosses above upper BB
2. `rsi > 68`
3. Volume falling (last 3 candles avg vol < 20-bar avg)
4. `dist_to_resistance_atr < 1.5`
5. Market regime is NOT uptrend

**REJECT:**
- Strong trend against trade direction (ADX > 30 + aligned EMAs) → no trade
- Quality score < 70

SL/TP: use `SmartLevels` with `min_rr=1.6`. Leverage: 4.

---

### S5 — ScalperGridStrategy (DOGEUSDT, 15m) ← PRIORITY STRATEGY

**LONG conditions:**
1. Price within 1.0 × ATR of `support` (not more than 2 candles ago)
2. `rsi < 37`
3. `volume_ratio > 1.2`
4. Confirmation candle: close > open (bullish) on current or previous candle
5. No 3+ consecutive impulse candles before entry
6. Quality score ≥ 65 (slightly relaxed for scalper)

**SHORT conditions:**
1. Price within 1.0 × ATR of `resistance`
2. `rsi > 63`
3. `volume_ratio > 1.2`
4. Confirmation: close < open

**REJECT:**
- Price already moved >2.5 × ATR from level → "chasing price"
- ATR filter required (atr >= atr50 × 0.8)
- Do NOT trade empty/thin candles (volume < 0.5 × avg)

SL: 1.1 × ATR. TP: resistance/support (adjust `min_rr=1.5`). Leverage: 5.
**Focus: more frequent but high-quality entries. TP taken quickly.**

---

### S7 — MultiConfirmStrategy (LINKUSDT, 15m)

Required confirmations: **4 of 6** (not 5 of 6 — reduce over-filtering):

1. EMA alignment (ema20 vs ema50 direction)
2. ATR expansion (atr > atr50 × 0.9)
3. Volume (volume_ratio > 1.15)
4. Trend (trend_strength > 0.4)
5. ADX (adx > 20)
6. Market regime (not flat)

**LONG:** 4/6 conditions met + rsi 35-60 + price above ema50
**SHORT:** 4/6 conditions met + rsi 40-65 + price below ema50

Quality score ≥ 70. Leverage: 5.

---

### S13 — OverboughtShortStrategy (1000PEPEUSDT, 15m)

**SHORT conditions (ALL required):**
1. `rsi > 72`
2. Volume falling: last 2 candles avg vol < 1.5-bar prior avg
3. RSI divergence: price made higher high, RSI made lower high (check last 5-8 candles)
4. Price at or above resistance (within 0.5 × ATR)
5. Close near or above upper BB (within 0.3 × ATR)
6. Reversal candle confirmation: bearish engulfing OR upper wick > 1.5 × body

**REJECT:**
- Strong uptrend (EMA20 > EMA50 > EMA200 AND ADX > 28) → do not short
- No divergence → no trade
- Quality score < 72

SL: above last swing high + 0.15 × ATR. Leverage: 4. Size: full.

---

### S14 — OverboughtShort Aggressive (WIFUSDT, 15m)

Same logic as S13 with:
- All S13 conditions apply
- Position size = S13 size × 0.70 (−30%)
- Confirmation required (no entry without reversal candle)
- `rsi > 75` (stricter threshold)
- Leverage: 3 (more conservative)

---

### S15 — AggressiveMomentumStrategy (OPUSDT, 1m)

**Keep existing logic, extend with:**
- If `momentum_score > 0.65` AND `adx > 25` AND `volume_ratio > 1.3`:
  - Do NOT cut TP early (disable early_tp or set threshold=95%)
  - Activate trailing stop after BE instead of fixed TP2/TP3
  - Allow hold up to 45 min
- Score ≥ 85: full size
- Score 75-84: 0.5× size
- Score < 75: no trade
- M15 trend filter still required (df_m15 alignment)

---

## PART 9 — TRADE LOGGING EXTENSION

Add these fields to `log_trade()` call dict (extend SQL schema via `_MIGRATION_COLUMNS`):

```python
"entry_reason":       str,   # why we entered (signal factors as JSON)
"exit_reason_detail": str,   # why we exited (TP/SL/timeout/trailing/early/momentum_weak)
"quality_score":      float, # pre-trade quality score
"market_regime":      str,   # uptrend/downtrend/volatile/flat
"adx_at_entry":       float,
"volume_ratio":       float,
"trend_strength":     float,
"support_at_entry":   float,
"resistance_at_entry":float,
"strategy_error":     str,   # populated on exception, else ""
```

`entry_reason` format (JSON string):
```json
{"ema":"aligned","rsi":28.4,"adx":24.1,"vol_ratio":1.35,"score":78.2,"regime":"uptrend"}
```

---

## PART 10 — AUTO-ANALYSIS (every 100 trades per strategy)

Add `StrategyAutoAnalyzer.maybe_run(strategy_id, journal)` called from main loop after trade close:

```python
# Runs if strategy.trades % 100 == 0 and trades >= 100
# Computes:
stats = {
  "win_rate":    float,   # %
  "profit_factor": float,
  "avg_win":     float,
  "avg_loss":    float,
  "expectancy":  float,   # avg_win*wr - avg_loss*(1-wr)
  "max_drawdown":float,   # max peak-to-trough in history
}
# Logs via logger.info("[AutoAnalysis] {sid}: {stats}")
# Adjustment rules:
if stats["profit_factor"] < 1.2:
    strategy.edge_wr_target = max(0.45, strategy.edge_wr_target - 0.02)
if stats["win_rate"] > 65 and stats["profit_factor"] > 1.8:
    strategy.edge_wr_target = min(0.70, strategy.edge_wr_target + 0.02)
```

---

## PART 11 — ML SNAPSHOT FEATURES (no lookahead bias)

In `ml/feature_extractor.py`, ensure snapshot features are computed **at bar close**:

Required fields in every snapshot:
```python
FEATURE_NAMES = [
  "adx", "rsi", "atr_ratio", "volume_ratio", "trend_strength",
  "ema20_vs_ema50", "ema50_vs_ema200",  # pct differences
  "dist_to_support_atr", "dist_to_resistance_atr",
  "macd_hist_norm",  # macd_hist / atr
  "momentum_score", "quality_score", "market_regime_encoded",
  "rolling_wr_20", "rolling_pnl_20", "consecutive_losses",
]
# market_regime_encoded: flat=0, volatile=1, downtrend=2, uptrend=3
```

**Anti-leakage rules:**
- Signal computed on `df[:-1]` (closed candles only)
- Never use close of signal candle as feature for that same signal
- Execution price = open of next candle

---

## IMPLEMENTATION CONSTRAINTS

1. **No breaking changes to** `TradingSignal` interface — `filters_passed` dict can be extended
2. **No external dependencies** — use only numpy, pandas, standard library
3. **Thread-safe** — strategies run in asyncio loop, no threading primitives in strategy code
4. **No pd.read_sql** with MySQL connections — always use `cursor.execute` + `pd.DataFrame([[row[c] for c in cols] for row in cur.fetchall()], columns=cols)`
5. All indicator functions must handle `df.shape[0] < required_period` gracefully (return None)
6. Keep `BaseStrategy.close_position()` signature intact
7. Preserve `restore_strategy_stats()` compatibility in `trade_journal.py`

---

## TARGET METRICS

```
Profit Factor  > 1.5
Win Rate       > 58%
Max Drawdown   < 15%
Noise trades   < 20% of all trades (score < 60 should be 0)
```

---

## EXECUTION ORDER

Implement in this order to minimize breakage:
1. `entry_filter.py` — `compute_market_context`, `compute_quality_score`, `UniversalFilters`
2. `tp_normalizer.py` → `SmartLevels` (keep old class as alias for BC)
3. `base.py` — add `signal_candle_timestamp` tracking for late-entry guard
4. `risk_manager.py` — `get_risk_pct`, `check_daily_loss`, leverage cap
5. `all_strategies.py` — rewrite S1, S2, S5, S7, S13, S14 using new components
6. `aggressive_momentum.py` — extend S15
7. `main.py` — cooldown dict, max-open guard, daily-loss check
8. `trade_journal.py` — migration columns, extended log_trade fields

Each file must be self-contained and testable independently.
