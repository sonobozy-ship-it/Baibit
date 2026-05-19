"""
Bybit API клиент — обёртка над pybit с асинхронной поддержкой.
Документация: https://bybit-exchange.github.io/docs/v5/intro
"""
import os
import asyncio
import logging
from typing import Optional, Dict, List
from pybit.unified_trading import HTTP, WebSocket
import pandas as pd

logger = logging.getLogger(__name__)


class BybitClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.session = HTTP(
            testnet=testnet,
            api_key=api_key or "",
            api_secret=api_secret or "",
        )
        self.testnet = testnet
        self.ws_public = None
        self.ws_private = None
        self.price_callbacks = {}  # symbol -> callback
        # Public-only mode: no credentials, only market data endpoints work
        self.public_only = not (api_key and api_secret)
        net = "TESTNET" if testnet else "MAINNET"
        mode = "read-only" if self.public_only else "authenticated"
        logger.info(f"Bybit клиент инициализирован ({net}, {mode})")

    # ========== БАЛАНС И АККАУНТ ==========
    def get_balance(self, coin: str = "USDT") -> float:
        """Получить баланс USDT."""
        if self.public_only:
            return 0.0
        try:
            res = self.session.get_wallet_balance(accountType="UNIFIED", coin=coin)
            if res["retCode"] == 0:
                wallets = res["result"]["list"][0]["coin"]
                for w in wallets:
                    if w["coin"] == coin:
                        return float(w["walletBalance"])
            return 0.0
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return 0.0

    def get_instrument_info(self, symbol: str) -> Dict:
        """Правила округления qty/price для символа (кешируется)."""
        if not hasattr(self, "_instrument_cache"):
            self._instrument_cache: Dict[str, Dict] = {}
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        try:
            res = self.session.get_instruments_info(category="linear", symbol=symbol)
            if res["retCode"] == 0 and res["result"]["list"]:
                info = res["result"]["list"][0]
                lot = info.get("lotSizeFilter", {})
                price_f = info.get("priceFilter", {})
                result = {
                    "min_qty":      float(lot.get("minOrderQty", "0.001")),
                    "qty_step":     float(lot.get("qtyStep", "0.001")),
                    "min_notional": float(lot.get("minNotionalValue", "5")),
                    "tick_size":    float(price_f.get("tickSize", "0.01")),
                    "max_leverage": float(info.get("leverageFilter", {}).get("maxLeverage", "100")),
                }
                self._instrument_cache[symbol] = result
                return result
        except Exception as e:
            logger.warning(f"get_instrument_info {symbol}: {e}")
        return {"min_qty": 0.001, "qty_step": 0.001, "min_notional": 5.0, "tick_size": 0.01, "max_leverage": 100.0}

    @staticmethod
    def round_to_step(value: float, step: float) -> float:
        """Округлить value вниз до кратного step."""
        if step <= 0:
            return value
        import math
        precision = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
        result = math.floor(value / step) * step
        return round(result, precision + 2)

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Получить открытые позиции."""
        if self.public_only:
            return []
        try:
            params = {"category": "linear", "settleCoin": "USDT"}
            if symbol:
                params["symbol"] = symbol
            res = self.session.get_positions(**params)
            if res["retCode"] == 0:
                return [p for p in res["result"]["list"] if float(p["size"]) > 0]
            return []
        except Exception as e:
            logger.error(f"Ошибка получения позиций: {e}")
            return []

    # ========== ТОРГОВЛЯ ==========
    def place_order(
        self,
        symbol: str,
        side: str,           # "Buy" или "Sell"
        qty: float,
        order_type: str = "Market",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: int = 1,
        reduce_only: bool = False,
    ) -> Dict:
        """Открыть позицию."""
        if self.public_only:
            return {"success": False, "error": "Public-only mode: no API credentials"}

        # ── Safety gate ──────────────────────────────────────────────
        # 1. Округление qty по правилам биржи
        info = self.get_instrument_info(symbol)
        qty = self.round_to_step(qty, info["qty_step"])
        if qty < info["min_qty"]:
            return {"success": False, "error": f"qty {qty} < minOrderQty {info['min_qty']}"}
        if qty * (price or 0) > 0 and qty * price < info["min_notional"]:
            # Проверка notional только если цена известна
            pass  # При market ордере цена неизвестна заранее

        # 2. Округление SL/TP по tick_size
        tick = info["tick_size"]
        if stop_loss is not None:
            stop_loss = self.round_to_step(stop_loss, tick)
        if take_profit is not None:
            take_profit = self.round_to_step(take_profit, tick)

        # 3. Hard cap на плечо
        max_lev = int(min(leverage, info["max_leverage"]))
        if max_lev != leverage:
            logger.warning(f"[SafetyGate] {symbol}: leverage {leverage}x → {max_lev}x (биржевой лимит)")
            leverage = max_lev

        # 4. Логирование перед отправкой
        logger.info(
            f"[ORDER] {symbol} {side} qty={qty} lev={leverage}x "
            f"SL={stop_loss} TP={take_profit} type={order_type}"
        )

        try:
            # Установить плечо
            try:
                self.session.set_leverage(
                    category="linear",
                    symbol=symbol,
                    buyLeverage=str(leverage),
                    sellLeverage=str(leverage),
                )
            except Exception:
                pass  # уже установлено

            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
                "reduceOnly": reduce_only,
            }
            if price and order_type == "Limit":
                params["price"] = str(price)
            if stop_loss:
                params["stopLoss"] = str(stop_loss)
            if take_profit:
                params["takeProfit"] = str(take_profit)

            res = self.session.place_order(**params)
            if res["retCode"] == 0:
                logger.info(f"Ордер открыт: {symbol} {side} {qty} @ {price or 'Market'}")
                return {"success": True, "data": res["result"]}
            else:
                logger.error(f"Ошибка ордера: {res['retMsg']}")
                return {"success": False, "error": res["retMsg"]}
        except Exception as e:
            logger.exception(f"Исключение при размещении ордера: {e}")
            return {"success": False, "error": str(e)}

    def close_position(self, symbol: str) -> Dict:
        """Закрыть позицию по рынку."""
        positions = self.get_positions(symbol)
        if not positions:
            return {"success": False, "error": "No position"}

        pos = positions[0]
        side = "Sell" if pos["side"] == "Buy" else "Buy"
        return self.place_order(
            symbol=symbol,
            side=side,
            qty=float(pos["size"]),
            reduce_only=True,
        )

    def update_stop_loss(self, symbol: str, stop_loss: float) -> Dict:
        """Перенести стоп-лосс (для безубытка / трейлинга)."""
        try:
            res = self.session.set_trading_stop(
                category="linear",
                symbol=symbol,
                stopLoss=str(stop_loss),
                positionIdx=0,
            )
            return {"success": res["retCode"] == 0, "data": res}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ========== СВЕЧИ ==========
    def get_klines(
        self,
        symbol: str,
        interval: str = "15",  # 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
        limit: int = 200,
    ) -> pd.DataFrame:
        """Получить свечи для анализа."""
        try:
            res = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit,
            )
            if res["retCode"] == 0:
                data = res["result"]["list"]
                df = pd.DataFrame(
                    data,
                    columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
                )
                df = df.iloc[::-1].reset_index(drop=True)  # развернуть, чтобы старые слева
                for col in ["open", "high", "low", "close", "volume", "turnover"]:
                    df[col] = pd.to_numeric(df[col])
                df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms")
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Ошибка получения свечей {symbol}: {e}")
            return pd.DataFrame()

    def get_ticker(self, symbol: str) -> Dict:
        """Получить текущую цену."""
        try:
            res = self.session.get_tickers(category="linear", symbol=symbol)
            if res["retCode"] == 0 and res["result"]["list"]:
                t = res["result"]["list"][0]
                return {
                    "symbol": symbol,
                    "price": float(t["lastPrice"]),
                    "change_24h": float(t["price24hPcnt"]) * 100,
                    "volume_24h": float(t["volume24h"]),
                    "open_interest": float(t.get("openInterest", 0)),
                    "funding_rate": float(t.get("fundingRate", 0)),
                }
            return {}
        except Exception as e:
            logger.error(f"Ошибка тикера {symbol}: {e}")
            return {}

    def get_top_usdt_symbols(
        self,
        top_n: int = 30,
        exclude: set = None,
        min_volume_usdt: float = 5_000_000,
    ) -> List[str]:
        """
        Возвращает топ-N USDT-перп символов по объёму за 24ч.
        Отфильтровывает малоликвидные пары (< min_volume_usdt).
        exclude — символы которые уже заняты другими стратегиями.
        """
        try:
            res = self.session.get_tickers(category="linear")
            if res["retCode"] != 0:
                return []
            items = res["result"]["list"]
            exclude = exclude or set()
            candidates = []
            for t in items:
                sym = t.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                vol = float(t.get("turnover24h") or t.get("volume24h") or 0)
                if vol < min_volume_usdt:
                    continue
                if sym in exclude:
                    continue
                candidates.append((sym, vol))
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [sym for sym, _ in candidates[:top_n]]
        except Exception as e:
            logger.error(f"get_top_usdt_symbols: {e}")
            return []

    def get_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """Получить ордербук (для slippage и orderbook фич)."""
        try:
            res = self.session.get_orderbook(category="linear", symbol=symbol, limit=limit)
            if res["retCode"] == 0:
                return {
                    "symbol": symbol,
                    "bids": res["result"]["b"],   # [[price, size], ...]
                    "asks": res["result"]["a"],
                    "timestamp": res["result"]["ts"],
                }
            return {}
        except Exception as e:
            logger.error(f"Ошибка orderbook {symbol}: {e}")
            return {}

    def get_funding_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """История ставок финансирования."""
        try:
            res = self.session.get_funding_rate_history(
                category="linear", symbol=symbol, limit=limit,
            )
            if res["retCode"] == 0:
                return res["result"]["list"]
            return []
        except Exception as e:
            logger.error(f"Funding history {symbol}: {e}")
            return []

    def get_market_meta(self, symbol: str) -> Dict:
        """Объединённые метаданные: funding, OI, тикер."""
        ticker = self.get_ticker(symbol)
        if not ticker:
            return {}

        # Получаем 2 последних свечи OI для расчёта изменения (опционально)
        oi_change = 0.0
        try:
            # На некоторых эндпойнтах OI растёт/падает
            pass  # simplified — OI берётся из тикера
        except Exception:
            pass

        return {
            "funding_rate": ticker.get("funding_rate", 0),
            "open_interest": ticker.get("open_interest", 0),
            "oi_change_pct": oi_change,
            "price": ticker.get("price"),
            "volume_24h": ticker.get("volume_24h"),
            "long_short_ratio": 1.0,  # Bybit не отдаёт напрямую; можно через отдельный endpoint
        }

    # ========== WEBSOCKET ==========
    def start_websocket(self, symbols: List[str], on_price_update):
        """Запустить WebSocket для live цен."""
        self.ws_public = WebSocket(testnet=self.testnet, channel_type="linear")
        for sym in symbols:
            self.ws_public.ticker_stream(
                symbol=sym,
                callback=lambda msg, s=sym: on_price_update(s, msg),
            )
        logger.info(f"WebSocket запущен для {symbols}")

    def stop_websocket(self):
        if self.ws_public:
            self.ws_public.exit()
        if self.ws_private:
            self.ws_private.exit()
