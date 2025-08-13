# collector.py
import websocket
import json
import pandas as pd
from datetime import datetime
import threading
import time
import os

lob_data = []

def on_message(ws, message):
    global lob_data
    data = json.loads(message)
    bids = data['bids'][:5]
    asks = data['asks'][:5]

    mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    spread = best_ask - best_bid
    bid_price, bid_volume = float(data['bids'][0][0]), float(data['bids'][0][1])
    ask_price, ask_volume = float(data['asks'][0][0]), float(data['asks'][0][1])
    imbalance = sum(float(b[1]) for b in bids) - sum(float(a[1]) for a in asks)

    lob_data.append({
        "timestamp": datetime.now(),
        "mid_price": mid_price,
        "spread": spread,
        "imbalance": imbalance,
        "bid_price": bid_price,
        "bid_volume": bid_volume,
        "ask_price": ask_price,
        "ask_volume": ask_volume

    })

def on_error(ws, error):
    print("Error:", error)

def on_close(ws):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection opened")

def save_to_csv_every_n_seconds(n, file_name="lob_data3.csv"):
    global lob_data
    first_write = True if not os.path.exists(file_name) else False
    while True:
        if lob_data:
            df = pd.DataFrame(lob_data)
            df.to_csv(file_name, mode="a", header=first_write, index=False)
            print(f"[Saved] {len(lob_data)} rows → {file_name}")
            lob_data = []
            first_write = False
        time.sleep(n)

def main():
    # Saving thread
    t = threading.Thread(target=save_to_csv_every_n_seconds, args=(10,))
    t.daemon = True
    t.start()

    # WebSocket thread
    socket = "wss://stream.binance.com:9443/ws/btcusdt@depth5@100ms"
    ws = websocket.WebSocketApp(socket,
                                 on_message=on_message,
                                 on_error=on_error,
                                 on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

if __name__ == "__main__":
    main()
