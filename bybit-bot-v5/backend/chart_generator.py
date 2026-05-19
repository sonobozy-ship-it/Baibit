"""
Генерация графика свечей с точками входа/выхода.
Возвращает путь к PNG-файлу.
"""
import io
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import mplfinance as mpf
    import matplotlib
    matplotlib.use("Agg")  # без дисплея
    import matplotlib.pyplot as plt
    CHART_AVAILABLE = True
except ImportError:
    CHART_AVAILABLE = False
    logger.warning("[Chart] mplfinance не установлен — графики недоступны")


def generate_trade_chart(
    df: pd.DataFrame,
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    exit_price: Optional[float] = None,
    strategy_id: str = "",
    candles: int = 60,
) -> Optional[bytes]:
    """
    Генерирует PNG-график свечей с линиями входа, SL, TP и выхода.
    Возвращает bytes PNG или None при ошибке.
    """
    if not CHART_AVAILABLE:
        return None
    try:
        # Берём последние N свечей
        plot_df = df.tail(candles).copy()
        plot_df.index = pd.to_datetime(plot_df.index)
        plot_df = plot_df[["open", "high", "low", "close", "volume"]].astype(float)
        plot_df.columns = ["Open", "High", "Low", "Close", "Volume"]

        # Линии уровней
        hlines_prices = [entry, sl, tp]
        hlines_colors = ["blue", "red", "green"]
        hlines_labels = [f"Вход {entry:.4f}", f"SL {sl:.4f}", f"TP {tp:.4f}"]

        if exit_price:
            hlines_prices.append(exit_price)
            hlines_colors.append("orange")
            hlines_labels.append(f"Выход {exit_price:.4f}")

        hlines = dict(
            hlines=hlines_prices,
            colors=hlines_colors,
            linewidths=[1.5] * len(hlines_prices),
            linestyle=["--"] * len(hlines_prices),
        )

        side_emoji = "🟢 LONG" if side in ("BUY", "Buy") else "🔴 SHORT"
        title = f"{strategy_id} | {symbol} | {side_emoji}"

        # Стиль
        style = mpf.make_mpf_style(
            base_mpf_style="charles",
            rc={"font.size": 8},
            marketcolors=mpf.make_marketcolors(
                up="#26a69a", down="#ef5350",
                edge="inherit", wick="inherit", volume="in",
            ),
        )

        buf = io.BytesIO()
        mpf.plot(
            plot_df,
            type="candle",
            style=style,
            title=title,
            volume=True,
            hlines=hlines,
            savefig=dict(fname=buf, dpi=120, bbox_inches="tight"),
            figsize=(10, 6),
        )
        buf.seek(0)
        return buf.read()

    except Exception as e:
        logger.error(f"[Chart] Ошибка генерации: {e}")
        return None
