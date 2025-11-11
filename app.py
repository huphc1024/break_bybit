from flask import Flask
import threading
from bot import main  # import hàm main() của anh

app = Flask(__name__)

@app.route('/')
def index():
    return "✅ Bot Bybit đang chạy trên Render!"

@app.route('/health')
def health():
    return {"status": "ok"}

# Chạy bot trên thread nền
def run_background():
    try:
        print("🚀 Khởi động bot...")
        main()
    except Exception as e:
        print("🔥 Lỗi chạy bot:", e)

threading.Thread(target=run_background, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
