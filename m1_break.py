import os
import time
import logging

import numpy as np
import requests
import pandas as pd
from datetime import date, datetime
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

# ------------------ load env ------------------
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TG_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

# ------------------ config ------------------
SYMBOL = "DOGEUSDT"
INTERVAL = "5"           # M5
USD_PER_TRADE = 10       # base usd mỗi lệnh (anh chỉnh được)
LEVERAGE = 10            # leverage đặt cho isolated
R_MULTIPLIER = 0.002     # 1R = 0.2% (SL), TP = 2R
SLEEP_SECONDS = 55

# vốn ban đầu để tính % PnL (nếu muốn)
INITIAL_CAPITAL = 500.0

# ------------------ logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ------------------ session ------------------
session = HTTP(testnet=True, api_key=API_KEY, api_secret=API_SECRET)
BASE_URL = "https://api.telegram.org/bot" + TG_TOKEN

# ====== HÀM HỖ TRỢ ======
def send_tele(text):
    try:
        requests.get(f"{BASE_URL}/sendMessage?chat_id={TG_CHAT_ID}&text={text}")
    except:
        print("Không gửi được Telegram")

def get_klines(symbol, interval=1, limit=100):
    res = session.get_kline(category="linear", symbol=symbol, interval=str(interval), limit=limit)
    df = pd.DataFrame(res['result']['list'], columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df[::-1].reset_index(drop=True)

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def find_support_resistance(df, window=10):
    highs = df['high']
    lows = df['low']
    resistances = [highs[i] for i in range(window, len(highs)-window)
                   if highs[i] == max(highs[i-window:i+window])]
    supports = [lows[i] for i in range(window, len(lows)-window)
                if lows[i] == min(lows[i-window:i+window])]
    if not resistances or not supports:
        return None, None
    return resistances[-1], supports[-1]

def set_isolated_leverage(symbol, leverage):
    try:
        session.set_leverage(category="linear", symbol=symbol, buyLeverage=str(leverage), sellLeverage=str(leverage))
        # session.switch_margin_mode(category="linear", symbol=symbol, tradeMode=0)  # 0=Isolated
    except Exception as e:
        print("Set leverage error:", e)

def get_qty(price):
    notional_usd = float(USD_PER_TRADE) * float(LEVERAGE)
    qty = notional_usd / float(price)
    qty_rounded = round(qty, 0)
    return max(int(qty_rounded), 1)

def place_order(side, price):
    qty = get_qty(price)
    try:
        result = session.place_order(
            category="linear",
            symbol=SYMBOL,
            side=side,
            orderType="Market",
            qty=qty,
            reduceOnly=False
        )
        print(f"✅ Order placed: {side} {qty} {SYMBOL}")
        return result
    except Exception as e:
        print(f"❌ Place order error: {e}")
        send_tele(f"⚠️ Lỗi vào lệnh {side}: {e}")
        return None

# ====== MAIN LOOP ======
def main():
    set_isolated_leverage(SYMBOL, LEVERAGE)
    send_tele("🤖 Bot đã khởi động!")

    last_signal = None
    base_equity = INITIAL_CAPITAL
    daily_pnl = 0

    while True:
        try:
            df = get_klines(SYMBOL, INTERVAL)
            df['rsi'] = calc_rsi(df['close'])
            price = float(df['close'].iloc[-1])

            resistance, support = find_support_resistance(df)
            if not resistance or not support:
                print("Chưa tìm thấy hỗ trợ/kháng cự")
                time.sleep(10)
                continue

            if df['rsi'].iloc[-1] > 70 and price >= resistance and last_signal != "SELL":
                place_order("Sell", price)
                last_signal = "SELL"
                send_tele(f"📉 SELL {SYMBOL} tại {price}, kháng cự {resistance}")

            elif df['rsi'].iloc[-1] < 30 and price <= support and last_signal != "BUY":
                place_order("Buy", price)
                last_signal = "BUY"
                send_tele(f"📈 BUY {SYMBOL} tại {price}, hỗ trợ {support}")

            # Giả lập PNL
            pnl_today = round(np.random.uniform(-1, 1), 2)
            daily_pnl += pnl_today
            total_pnl = base_equity + daily_pnl
            print(f"{datetime.now()} | Price: {price} | PNL ngày: {daily_pnl} | Tổng vốn: {total_pnl}")

            time.sleep(60)

        except Exception as e:
            print("Main loop error:", e)
            send_tele(f"❗Lỗi vòng lặp chính: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()