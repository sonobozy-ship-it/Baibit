"""
Бэктестер — прогон стратегии на исторических данных.
Загружает свечи из Bybit или CSV, имитирует торговлю, считает метрики.
"""
import pandas as pd
import numpy as np
from typing import Type, List, Dict
from datetime import datetime
import logging

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class BacktestResult:
    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.start_balance = 0
        self.end_balance = 0

    def calculate_metrics(self) -> Dict:
        """
        Полный набор метрик:
        PF, Sharpe, Sortino, MaxDD, WinRate, Expectancy, AvgRR,
        Monthly Returns, Equity Curve.
        """
        if not self.trades:
            return {"trades": 0, "win_rate": 0, "total_pnl": 0}

        pnls    = [t["pnl"] for t in self.trades]
        r_mults = [t.get("r_multiple", 0.0) for t in self.trades]
        wins    = [p for p in pnls if p > 0]
        losses  = [p for p in pnls if p < 0]

        total_pnl     = sum(pnls)
        n             = len(pnls)
        win_rate      = len(wins) / n * 100
        avg_win       = float(np.mean(wins))   if wins   else 0.0
        avg_loss      = float(np.mean(losses)) if losses else 0.0
        profit_factor = (
            abs(sum(wins) / sum(losses))
            if losses and sum(losses) != 0 else float("inf")
        )

        # ── Max Drawdown ────────────────────────────────────────────────────
        peak = self.equity_curve[0] if self.equity_curve else 0.0
        max_dd, max_dd_pct = 0.0, 0.0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd     = peak - eq
                max_dd_pct = dd_pct

        # ── Sharpe (annualised) ─────────────────────────────────────────────
        if n > 1 and self.start_balance > 0:
            rets    = np.array(pnls) / self.start_balance
            std_ret = float(rets.std())
            sharpe  = float((rets.mean() / std_ret) * np.sqrt(252)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        # ── Sortino (downside deviation) ────────────────────────────────────
        if self.start_balance > 0:
            rets_arr = np.array(pnls) / self.start_balance
            down     = rets_arr[rets_arr < 0]
            down_std = float(down.std()) if len(down) > 1 else 0.0
            sortino  = float((rets_arr.mean() / down_std) * np.sqrt(252)) if down_std > 0 else 0.0
        else:
            sortino = 0.0

        # ── Expectancy ($) ──────────────────────────────────────────────────
        wr_dec     = win_rate / 100
        expectancy = wr_dec * avg_win + (1 - wr_dec) * avg_loss  # avg_loss < 0

        # ── Average R-Multiple ──────────────────────────────────────────────
        avg_rr = float(np.mean(r_mults)) if r_mults else 0.0

        # ── Monthly Returns ─────────────────────────────────────────────────
        monthly: Dict[str, float] = {}
        for t in self.trades:
            ts = str(t.get("timestamp", ""))
            month = ts[:7] if len(ts) >= 7 else "unknown"
            monthly[month] = round(monthly.get(month, 0.0) + t["pnl"], 4)

        return {
            "trades":              n,
            "wins":                len(wins),
            "losses":              len(losses),
            "win_rate":            round(win_rate, 2),
            "total_pnl":           round(total_pnl, 2),
            "avg_win":             round(avg_win, 4),
            "avg_loss":            round(avg_loss, 4),
            "profit_factor":       round(profit_factor, 3),
            "expectancy":          round(expectancy, 4),
            "avg_rr":              round(avg_rr, 3),
            "max_drawdown":        round(max_dd, 2),
            "max_drawdown_pct":    round(max_dd_pct, 2),
            "sharpe":              round(sharpe, 3),
            "sortino":             round(sortino, 3),
            "start_balance":       round(self.start_balance, 2),
            "end_balance":         round(self.end_balance, 2),
            "roi_pct":             round(
                (self.end_balance - self.start_balance) / self.start_balance * 100, 2
            ) if self.start_balance else 0,
            "best_trade":          round(max(pnls), 4),
            "worst_trade":         round(min(pnls), 4),
            "monthly_returns":     monthly,
            "equity_curve":        [round(e, 2) for e in self.equity_curve],
            "longest_win_streak":  self._longest_streak(pnls, positive=True),
            "longest_loss_streak": self._longest_streak(pnls, positive=False),
        }

    @staticmethod
    def _longest_streak(pnls: List[float], positive: bool) -> int:
        max_streak = current = 0
        for p in pnls:
            if (p > 0) == positive:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak


class Backtester:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        fee_pct: float = 0.06,
        conservative: bool = True,
        slippage_pct: float = 0.05,
        funding_pct: float = 0.01,
    ):
        """
        fee_pct       — комиссия в % (Bybit taker = 0.06%)
        conservative  — при SL+TP на одной свече берём худший (SL) исход
        slippage_pct  — % от notional при входе
        funding_pct   — % от notional каждые 8ч (funding)
        """
        self.initial_balance = initial_balance
        self.fee_pct = fee_pct
        self.conservative = conservative
        self.slippage_pct = slippage_pct
        self.funding_pct = funding_pct

    def run(
        self,
        strategy_class: Type[BaseStrategy],
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        warmup_candles: int = 220,
        **strategy_kwargs,
    ) -> BacktestResult:
        """
        Запускает бэктест.
        df: OHLCV свечи (исторические).
        """
        strategy = strategy_class(symbol=symbol, **strategy_kwargs)
        result = BacktestResult()
        result.start_balance = self.initial_balance
        balance = self.initial_balance
        result.equity_curve.append(balance)

        strategy.timeframe_minutes = getattr(strategy, 'timeframe_minutes', 15)
        logger.info(f"🔬 Backtest {strategy.ID} {strategy.NAME} на {len(df)} свечей")

        for i in range(warmup_candles, len(df) - 1):  # -1 чтобы был next_candle для исполнения
            window = df.iloc[:i + 1].copy()           # фичи считаем НА ЗАКРЫТИИ свечи i
            current_candle = window.iloc[-1]
            next_candle = df.iloc[i + 1]              # исполнение на следующей свече (no look-ahead)

            # Если позиция открыта — проверяем SL/TP/BE/Trailing на NEXT свече
            if strategy.current_position:
                pos = strategy.current_position
                # Используем high/low СЛЕДУЮЩЕЙ свечи (реалистично)
                hit_sl = (pos["side"] == "Buy" and next_candle["low"] <= pos["sl"]) or \
                         (pos["side"] == "Sell" and next_candle["high"] >= pos["sl"])
                hit_tp = (pos["side"] == "Buy" and next_candle["high"] >= pos["tp"]) or \
                         (pos["side"] == "Sell" and next_candle["low"] <= pos["tp"])

                # Breakeven и trailing на open следующей
                next_open = float(next_candle["open"])
                strategy.check_breakeven(next_open)
                strategy.check_trailing_stop(next_open)

                # Conservative mode: если SL и TP задеты на одной свече — берём худший сценарий
                if self.conservative and hit_sl and hit_tp:
                    hit_tp = False  # считаем что сначала hit SL

                if hit_sl or hit_tp:
                    exit_price = pos["tp"] if hit_tp else pos["sl"]
                    entry = pos["entry"]
                    side = pos["side"]
                    qty = pos.get("qty", 1.0)

                    # КОРРЕКТНЫЙ PnL (без leverage hack)
                    if side == "Buy":
                        gross = qty * (exit_price - entry)
                    else:
                        gross = qty * (entry - exit_price)
                    notional_in = qty * entry
                    notional_out = qty * exit_price
                    fees = (notional_in + notional_out) * (self.fee_pct / 100)
                    pnl_usd = gross - fees

                    slippage = notional_in * (self.slippage_pct / 100)
                    pnl_usd -= slippage

                    # Funding: примерно 1 раз за 8 часов на удерживаемую позицию
                    candles_held = i - strategy.current_position.get("open_bar", i)
                    funding_periods = candles_held * (strategy.timeframe_minutes / 480)
                    funding_cost = notional_in * (self.funding_pct / 100) * max(0, funding_periods)
                    pnl_usd -= funding_cost

                    balance += pnl_usd

                    # R-multiple для ML
                    sl_distance = abs(entry - pos["sl"])
                    r_gain = (exit_price - entry) if side == "Buy" else (entry - exit_price)
                    r_mult = round(r_gain / sl_distance, 3) if sl_distance > 0 else 0

                    result.trades.append({
                        "timestamp": str(next_candle["timestamp"]),
                        "side": side,
                        "entry": entry,
                        "exit": exit_price,
                        "qty": qty,
                        "pnl": round(pnl_usd, 4),
                        "pnl_pct": round(pnl_usd / notional_in * 100, 4) if notional_in else 0,
                        "r_multiple": r_mult,
                        "fees": round(fees, 4),
                        "exit_reason": "TP" if hit_tp else "SL",
                        "balance_after": round(balance, 2),
                    })
                    result.equity_curve.append(balance)
                    strategy.current_position = None

                    if balance <= 0:
                        logger.error("💀 Депо обнулилось")
                        break
                continue

            # Нет позиции → ищем сигнал НА ЗАКРЫТИИ текущей свечи
            signal = strategy.analyze(window)
            if signal and signal.action in ("BUY", "SELL"):
                # Исполняем на OPEN следующей свечи (реалистично)
                entry_price = float(next_candle["open"])

                # Размер позиции из risk_per_trade%
                risk_pct = 1.0
                risk_usd = balance * (risk_pct / 100)
                sl_dist = abs(entry_price - signal.stop_loss)
                qty = risk_usd / sl_dist if sl_dist > 0 else 0
                if qty <= 0:
                    continue

                strategy.register_position(
                    side="Buy" if signal.action == "BUY" else "Sell",
                    entry=entry_price,
                    sl=signal.stop_loss,
                    tp=signal.take_profit,
                )
                strategy.current_position["qty"] = qty
                strategy.current_position["open_bar"] = i

        result.end_balance = balance
        logger.info(f"✅ Backtest завершён. Trades: {len(result.trades)}, Balance: {balance:.2f}")
        return result

    def compare_strategies(
        self,
        strategies: List[Type[BaseStrategy]],
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
    ) -> Dict[str, Dict]:
        """Сравнить несколько стратегий на одних и тех же данных."""
        results = {}
        for s_class in strategies:
            res = self.run(s_class, df, symbol)
            results[s_class.ID] = {
                "name": s_class.NAME,
                "metrics": res.calculate_metrics(),
            }
        return results
