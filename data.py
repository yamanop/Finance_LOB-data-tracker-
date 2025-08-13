import websocket
import json
import pandas as pd
from datetime import datetime

lob_data = []

def on_message(ws, message):
    data = json.loads(message)
    bids = data['bids'][:5]
    asks = data['asks'][:5]

    mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    spread = best_ask - best_bid
    imbalance = sum(float(b[1]) for b in bids) - sum(float(a[1]) for a in asks)

    lob_data.append({
        "timestamp": datetime.now(),
        "mid_price": mid_price,
        "spread": spread,
        "imbalance": imbalance
        
    })

    # Print last row
    print(lob_data[-1])

def on_error(ws, error):
    print("Error:", error)

def on_close(ws):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection opened")

socket = "wss://stream.binance.com:9443/ws/btcusdt@depth5@100ms"
ws = websocket.WebSocketApp\
(socket, on_message=on_message, on_error=on_error, on_close=on_close)
ws.on_open = on_open
ws.run_forever()


df = pd.DataFrame(lob_data)
df.to_csv("lob_data.csv", index=False)
