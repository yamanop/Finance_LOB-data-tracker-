from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow frontend to connect
model = joblib.load("btc_xgboost_model.joblib")
label_map = {0: -1, 1: 0, 2: 1}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    bid_price = float(data['bid_price'])
    ask_price = float(data['ask_price'])
    bid_volume = float(data['bid_volume'])
    ask_volume = float(data['ask_volume'])

    spread = ask_price - bid_price
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

    features = np.array([[bid_price, ask_price, bid_volume, ask_volume, spread, imbalance]])
    prediction = model.predict(features)[0]
    return jsonify({
        "prediction": label_map[prediction]
    })

if __name__ == "__main__":
    app.run(debug=True)
