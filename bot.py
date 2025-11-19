import os
import time
import math
import logging
import traceback
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import requests
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

# ---------------- load env ----------------
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

# ---------------- config ----------------
SYMBOL = "TONUSDT"
TESTNET = os.getenv("TESTNET", "false").lower() in ("1", "true", "yes")
USD_PER_TRADE = 1.0
# INTERVAL = "1"
# KLINE_LIMIT = 200
RSI_PERIOD = 14
EMA_FAST = 5
EMA_SLOW = 20
VOLUME_MULTIPLIER = 2.0
SL_BUFFER_PCT = 0.003   # 0.3%
TP_BUFFER_PCT = 0.004   # 0.4%
MAX_RETRY = 5
BASE_SLEEP = 1.0
INTERVAL = "1"     # 1 phút
KLINE_LIMIT = 50

# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------- session ----------------
session = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET)
BASE_TELE_URL = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"

# ---------------- helpers ----------------
def send_telegram(msg: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        logging.debug("Telegram not configured.")
        return
    try:
        requests.post(BASE_TELE_URL, data={"chat_id": TG_CHAT_ID, "text": msg}, timeout=6)
    except Exception as e:
        logging.error("Telegram send error: %s", e)

def backoff(attempt: int):
    sleep_t = BASE_SLEEP * (2 ** (attempt - 1))
    logging.warning("Backing off for %.1f seconds (attempt %d)", sleep_t, attempt)
    time.sleep(sleep_t)

# ---------------- indicators ----------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["close"] = df["close"].astype(float)
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["vol_ma"] = df["volume"].rolling(window=20, min_periods=1).mean()
    return df

# ---------------- support/resistance ----------------
def find_support_resistance(df: pd.DataFrame, lookback: int = 20) -> Tuple[Optional[float], Optional[float]]:
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    close = df["close"].astype(float).values
    if len(close) < lookback:
        return None, None
    resistance = float(np.max(highs[-lookback:]))
    support = float(np.min(lows[-lookback:]))
    price = float(close[-1])
    if not (resistance > price):
        resistance = max([h for h in highs if h > price], default=None)
    if not (support < price):
        support = min([l for l in lows if l < price], default=None)
    return support, resistance

# ---------------- kline helpers ----------------
def get_klines(symbol: str, interval: str = INTERVAL, limit: int = KLINE_LIMIT) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
            data = resp.get("result", {}).get("list", [])
            if not data:
                return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            df = pd.DataFrame(data)
            ts_col = next((c for c in df.columns if "time" in str(c).lower() or "start" in str(c).lower()), df.columns[0])
            df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
            unit = "ms" if df[ts_col].median(skipna=True) > 1e12 else "s"
            df["time"] = pd.to_datetime(df[ts_col], unit=unit, errors="coerce")
            for col in ["open","high","low","close","volume"]:
                df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
            df = df[["time","open","high","low","close","volume"]]
            return df
        except Exception as e:
            logging.error("get_klines attempt %d error: %s", attempt, e)
            backoff(attempt)
    return pd.DataFrame(columns=["time","open","high","low","close","volume"])

def get_mark_price(symbol: str) -> float:
    for attempt in range(1, MAX_RETRY + 1):
        try:
            t = session.get_tickers(category="linear", symbol=symbol)
            return float(t["result"]["list"][0]["lastPrice"])
        except Exception as e:
            logging.error("get_mark_price attempt %d error: %s", attempt, e)
            backoff(attempt)
    return 0.0

def get_position_info(symbol: str) -> Optional[dict]:
    try:
        res = session.get_positions(category="linear", symbol=symbol)
        lst = res.get("result", {}).get("list", [])
        return lst[0] if lst else None
    except Exception as e:
        logging.error("get_position_info error: %s", e)
        return None

def get_wallet_equity() -> float:
    try:
        res = session.get_wallet_balance(accountType="UNIFIED")
        total = float(res["result"]["list"][0]["totalEquity"])
        return total
    except Exception as e:
        logging.error("get_wallet_equity error: %s", e)
        return 0.0

def calc_qty_for_usd(symbol: str, usd_value: float) -> float:
    price = get_mark_price(symbol)
    raw_qty = usd_value / price
    return max(round(raw_qty, 6), 0.000001)

def place_market_order(symbol: str, side: str, qty: float) -> Optional[dict]:
    for attempt in range(1, MAX_RETRY + 1):
        try:
            res = session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                timeInForce="GoodTillCancel",
                reduceOnly=False,
                positionIdx=1
            )
            logging.info("Order placed: %s %.6f %s", side, qty, symbol)
            return res
        except Exception as e:
            logging.error("place_market_order attempt %d error: %s", attempt, e)
            backoff(attempt)
    return None

def set_sl_tp(symbol: str, sl_price: float, tp_price: float):
    try:
        session.set_trading_stop(category="linear", symbol=symbol, stopLoss=str(sl_price), takeProfit=str(tp_price))
        logging.info("SL=%.6f TP=%.6f set", sl_price, tp_price)
    except Exception as e:
        logging.error("set_sl_tp error: %s", e)

# ---------------- main bot loop ----------------
def trade_loop():
    send_telegram(f"🤖 Bot started for {SYMBOL} | {USD_PER_TRADE}$ per trade | Testnet={TESTNET}")
    open_position = None
    equity_start = get_wallet_equity()

    while True:
        try:
            df = get_klines(SYMBOL)
            if df.empty:
                time.sleep(5)
                continue

            df = compute_indicators(df)
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df)>=2 else last

            support, resistance = find_support_resistance(df)
            price = float(last["close"])

            trend_up = last["ema_fast"] > last["ema_slow"]
            trend_down = last["ema_fast"] < last["ema_slow"]
            rsi_val = float(last["rsi"]) if not pd.isna(last["rsi"]) else None
            vol_spike = last["volume"] > last["vol_ma"] * VOLUME_MULTIPLIER

            signal = None
            if open_position is None:
                if trend_up and rsi_val and rsi_val > 50 and (last["close"] > prev["high"] or vol_spike):
                    signal = "BUY"
                elif trend_down and rsi_val and rsi_val < 50 and (last["close"] < prev["low"] or vol_spike):
                    signal = "SELL"

                if signal:
                    qty = calc_qty_for_usd(SYMBOL, USD_PER_TRADE)
                    if qty <= 0:
                        logging.error("Qty<=0, skip")
                        time.sleep(5)
                        continue

                    if signal=="BUY":
                        sl = support*(1-SL_BUFFER_PCT) if support else price*(1-SL_BUFFER_PCT)
                        tp = resistance*(1+TP_BUFFER_PCT) if resistance else price*(1+TP_BUFFER_PCT)
                    else:
                        sl = resistance*(1+SL_BUFFER_PCT) if resistance else price*(1+SL_BUFFER_PCT)
                        tp = support*(1-TP_BUFFER_PCT) if support else price*(1-TP_BUFFER_PCT)

                    place_market_order(SYMBOL, "Buy" if signal=="BUY" else "Sell", qty)
                    set_sl_tp(SYMBOL, sl, tp)
                    open_position = {"side": signal, "qty": qty, "entry_price": price, "sl": sl, "tp": tp}
                    send_telegram(f"🟢 ENTER {signal} {qty} @ {price:.6f} | SL={sl:.6f} TP={tp:.6f}")

            else:
                pos = get_position_info(SYMBOL)
                if not pos or float(pos.get("size",0))==0:
                    equity_now = get_wallet_equity()
                    pnl_real = equity_now - equity_start
                    send_telegram(f"🟣 Position closed. Realized PnL: {pnl_real:.4f} USDT")
                    logging.info(f"Position closed. PnL: {pnl_real:.4f}")
                    open_position = None
                    equity_start = equity_now
                else:
                    upl = float(pos.get("unrealisedPnl",0.0))
                    logging.info(f"Open {open_position['side']} | Unrealized PnL: {upl:.4f} USDT")

            time.sleep(60)

        except Exception as e:
            logging.error("Main loop exception: %s", e)
            logging.error(traceback.format_exc())
            send_telegram(f"❗Error: {e}")
            time.sleep(10)


if __name__=="__main__":
    trade_loop()