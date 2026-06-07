"""
╔══════════════════════════════════════════════════════════╗
║           براند 22 — PAXG/USDT Gold Radar Server        ║
║         Real-Time WebSocket + Technical Analysis         ║
╚══════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import pandas_ta as ta
import websockets
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
SYMBOL          = "paxgusdt"
BINANCE_WS_URL  = f"wss://stream.binance.com:9443/ws/{SYMBOL}@trade"
CANDLE_INTERVAL = 60        # seconds per candle (1-Minute)
MAX_CANDLES     = 200       # history depth for indicator accuracy
RSI_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
RSI_OVERSOLD    = 35        # BUY zone threshold
RSI_OVERBOUGHT  = 65        # SELL zone threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("brand22")

# ─────────────────────────────────────────────
#  CANDLE ENGINE
# ─────────────────────────────────────────────
class CandleEngine:
    """Aggregates raw trade ticks into OHLCV 1-minute candles."""

    def __init__(self):
        self.candles: deque[dict] = deque(maxlen=MAX_CANDLES)
        self._open:   Optional[float] = None
        self._high:   float = 0.0
        self._low:    float = float("inf")
        self._close:  float = 0.0
        self._volume: float = 0.0
        self._start:  Optional[int] = None   # epoch second of current candle

    def _current_minute(self, ts_ms: int) -> int:
        """Floor timestamp (ms) to the minute epoch second."""
        return (ts_ms // 1000) - ((ts_ms // 1000) % CANDLE_INTERVAL)

    def add_trade(self, price: float, qty: float, ts_ms: int) -> Optional[dict]:
        """
        Feed a trade tick. Returns a completed candle dict if the minute
        boundary was crossed, otherwise None.
        """
        minute = self._current_minute(ts_ms)

        # First tick ever
        if self._start is None:
            self._start  = minute
            self._open   = price
            self._high   = price
            self._low    = price
            self._close  = price
            self._volume = qty
            return None

        # Same candle → update
        if minute == self._start:
            if price > self._high: self._high = price
            if price < self._low:  self._low  = price
            self._close  = price
            self._volume += qty
            return None

        # New candle → finalise previous
        completed = {
            "ts":     self._start,
            "open":   self._open,
            "high":   self._high,
            "low":    self._low,
            "close":  self._close,
            "volume": round(self._volume, 6),
        }
        self.candles.append(completed)

        # Start fresh candle
        self._start  = minute
        self._open   = price
        self._high   = price
        self._low    = price
        self._close  = price
        self._volume = qty

        log.info(
            f"Candle closed | O:{completed['open']:.2f}  H:{completed['high']:.2f}"
            f"  L:{completed['low']:.2f}  C:{completed['close']:.2f}"
            f"  V:{completed['volume']:.4f}"
        )
        return completed

    def to_dataframe(self) -> pd.DataFrame:
        if not self.candles:
            return pd.DataFrame()
        df = pd.DataFrame(list(self.candles))
        df.set_index("ts", inplace=True)
        df.sort_index(inplace=True)
        return df


# ─────────────────────────────────────────────
#  TECHNICAL ANALYSIS ENGINE
# ─────────────────────────────────────────────
class TechnicalAnalysis:
    """Computes RSI + MACD and emits trading signals."""

    @staticmethod
    def compute(df: pd.DataFrame) -> dict:
        """
        Requires at least (RSI_PERIOD + MACD_SLOW + MACD_SIGNAL) rows
        for meaningful results.
        """
        min_rows = MACD_SLOW + MACD_SIGNAL + 5
        if len(df) < min_rows:
            return {"ready": False, "reason": f"Need ≥{min_rows} candles (have {len(df)})"}

        close = df["close"].astype(float)

        # ── RSI ──────────────────────────────────
        rsi_series = ta.rsi(close, length=RSI_PERIOD)
        rsi = float(rsi_series.iloc[-1]) if rsi_series is not None else None

        # ── MACD ─────────────────────────────────
        macd_df = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        if macd_df is None or macd_df.empty:
            return {"ready": False, "reason": "MACD calculation failed"}

        macd_col   = [c for c in macd_df.columns if c.startswith("MACD_")  and "s" not in c.lower() and "h" not in c.lower()][0]
        signal_col = [c for c in macd_df.columns if "MACDs" in c][0]
        hist_col   = [c for c in macd_df.columns if "MACDh" in c][0]

        macd_val    = float(macd_df[macd_col].iloc[-1])
        signal_val  = float(macd_df[signal_col].iloc[-1])
        hist_val    = float(macd_df[hist_col].iloc[-1])
        hist_prev   = float(macd_df[hist_col].iloc[-2])

        # ── Signal Logic ─────────────────────────
        signal, confidence, reasons = TechnicalAnalysis._decide(
            rsi, macd_val, signal_val, hist_val, hist_prev
        )

        return {
            "ready":      True,
            "rsi":        round(rsi, 2) if rsi else None,
            "macd":       round(macd_val, 4),
            "macd_signal": round(signal_val, 4),
            "macd_hist":  round(hist_val, 4),
            "signal":     signal,          # "BUY" | "SELL" | "HOLD"
            "confidence": confidence,      # "HIGH" | "MEDIUM" | "LOW"
            "reasons":    reasons,
        }

    @staticmethod
    def _decide(
        rsi: float, macd: float, macd_sig: float,
        hist: float, hist_prev: float
    ) -> tuple[str, str, list[str]]:

        reasons = []
        buy_score  = 0
        sell_score = 0

        # ── RSI scoring ──────────────────────────
        if rsi <= RSI_OVERSOLD:
            buy_score += 2
            reasons.append(f"RSI oversold ({rsi:.1f} ≤ {RSI_OVERSOLD})")
        elif rsi >= RSI_OVERBOUGHT:
            sell_score += 2
            reasons.append(f"RSI overbought ({rsi:.1f} ≥ {RSI_OVERBOUGHT})")
        elif rsi < 45:
            buy_score += 1
            reasons.append(f"RSI leaning bearish ({rsi:.1f})")
        elif rsi > 55:
            sell_score += 1
            reasons.append(f"RSI leaning bullish ({rsi:.1f})")

        # ── MACD cross scoring ────────────────────
        macd_cross_up   = (hist > 0) and (hist_prev <= 0)
        macd_cross_down = (hist < 0) and (hist_prev >= 0)

        if macd_cross_up:
            buy_score += 2
            reasons.append("MACD bullish crossover")
        elif macd > macd_sig and hist > hist_prev:
            buy_score += 1
            reasons.append("MACD momentum positive")

        if macd_cross_down:
            sell_score += 2
            reasons.append("MACD bearish crossover")
        elif macd < macd_sig and hist < hist_prev:
            sell_score += 1
            reasons.append("MACD momentum negative")

        # ── Verdict ───────────────────────────────
        if buy_score >= 3:
            return "BUY",  "HIGH"   if buy_score >= 4  else "MEDIUM", reasons
        if sell_score >= 3:
            return "SELL", "HIGH"   if sell_score >= 4 else "MEDIUM", reasons
        if buy_score == 2:
            return "BUY",  "LOW",  reasons
        if sell_score == 2:
            return "SELL", "LOW",  reasons
        return "HOLD", "LOW", reasons or ["No clear signal"]


# ─────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────
engine  = CandleEngine()
ta_eng  = TechnicalAnalysis()

state = {
    "last_price":   None,
    "last_update":  None,
    "last_signal":  None,
    "candle_count": 0,
    "connected":    False,
}


# ─────────────────────────────────────────────
#  BINANCE WEBSOCKET LISTENER
# ─────────────────────────────────────────────
async def binance_listener():
    """Persistent WebSocket loop with auto-reconnect."""
    backoff = 1
    while True:
        try:
            log.info(f"Connecting to Binance WS → {BINANCE_WS_URL}")
            async with websockets.connect(
                BINANCE_WS_URL,
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                state["connected"] = True
                backoff = 1
                log.info("✅ Connected to Binance WebSocket")

                async for raw in ws:
                    msg = json.loads(raw)
                    # Trade stream: {"e":"trade","T":timestamp_ms,"p":"price","q":"qty"}
                    price  = float(msg["p"])
                    qty    = float(msg["q"])
                    ts_ms  = int(msg["T"])

                    state["last_price"]  = price
                    state["last_update"] = ts_ms

                    completed = engine.add_trade(price, qty, ts_ms)
                    if completed:
                        state["candle_count"] += 1
                        df = engine.to_dataframe()
                        analysis = ta_eng.compute(df)
                        if analysis.get("ready"):
                            state["last_signal"] = {
                                **analysis,
                                "price":      price,
                                "candle":     completed,
                                "timestamp":  datetime.now(timezone.utc).isoformat(),
                            }
                            log.info(
                                f"📡 Signal → {analysis['signal']} "
                                f"[{analysis['confidence']}] | "
                                f"RSI:{analysis['rsi']}  MACD:{analysis['macd']}"
                            )

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            state["connected"] = False
            log.warning(f"WS disconnected: {e}. Retrying in {backoff}s…")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

        except Exception as e:
            state["connected"] = False
            log.error(f"Unexpected error: {e}", exc_info=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


# ─────────────────────────────────────────────
#  FASTAPI APPLICATION
# ─────────────────────────────────────────────
app = FastAPI(
    title="براند 22 — PAXG/USDT Gold Radar API",
    version="1.0.0",
    description="Real-time gold trading signals powered by RSI + MACD analysis",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your Netlify domain in production
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    asyncio.create_task(binance_listener())
    log.info("🚀 براند 22 server started")


# ── Endpoints ────────────────────────────────

@app.get("/signal", summary="Latest trading signal")
async def get_signal():
    """
    Returns the most recent BUY / SELL / HOLD signal with
    RSI, MACD values, confidence level, and reasoning.
    """
    if not state["last_signal"]:
        return JSONResponse(
            status_code=202,
            content={
                "status":  "warming_up",
                "message": "Collecting candles… check back shortly.",
                "candles_collected": state["candle_count"],
                "connected": state["connected"],
            },
        )
    return {
        "status": "ok",
        **state["last_signal"],
    }


@app.get("/price", summary="Current live price")
async def get_price():
    """Lightweight endpoint — just the current PAXG/USDT price."""
    return {
        "symbol":     "PAXG/USDT",
        "price":      state["last_price"],
        "updated_ms": state["last_update"],
        "connected":  state["connected"],
    }


@app.get("/candles", summary="Recent 1-minute candles (OHLCV)")
async def get_candles(limit: int = 50):
    """Returns the last `limit` completed 1-minute candles."""
    limit = min(limit, MAX_CANDLES)
    candles = list(engine.candles)[-limit:]
    return {
        "symbol":  "PAXG/USDT",
        "interval": "1m",
        "count":   len(candles),
        "candles": candles,
    }


@app.get("/health", summary="Server health check")
async def health():
    return {
        "status":           "running",
        "connected":        state["connected"],
        "candles_collected": state["candle_count"],
        "last_price":       state["last_price"],
        "server_time":      datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
