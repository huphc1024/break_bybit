import os
import threading
from flask import Flask
import bot  # import file bot.py của anh

app = Flask(__name__)

@app.route("/")
def home():
    return "🤖 Bot đang chạy ngon lành!"

@app.route("/health")
def health():
    return {"status": "ok"}

def start_bot():
    bot.main()  # gọi hàm main() trong bot.py

if __name__ == "__main__":
    threading.Thread(target=start_bot, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))