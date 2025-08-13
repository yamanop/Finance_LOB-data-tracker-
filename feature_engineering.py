# feature_engineering.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("lob_data3.csv")

# 🧮 Basic LOB features
df['spread'] = df['ask_price'] - df['bid_price']
df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
df['imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'] + 1e-6)
df['depth_ratio'] = df['bid_volume'] / (df['ask_volume'] + 1e-6)
df['price_derivative'] = df['mid_price'].diff()
df['order_flow_imbalance'] = df['bid_volume'].diff() - df['ask_volume'].diff()

# 🔁 Rolling features
window = 10  # ~1 second if 100ms sampling
df['volatility_1s'] = df['mid_price'].rolling(window).std()
df['rolling_spread'] = df['spread'].rolling(window).mean()
df['rolling_imbalance_std'] = df['imbalance'].rolling(window).std()

# 🎯 Target variable: future return
lookahead = 10  # ~1 sec ahead
df['future_return'] = df['mid_price'].shift(-lookahead) - df['mid_price']

# 🏷️ Label: -1 (down), 0 (flat), 1 (up)
threshold = df['future_return'].std()
df['label'] = df['future_return'].apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))

# Drop unused / nan rows
df.dropna(inplace=True)

# ✅ Save for modeling
df.to_csv("engineered_lob2.csv", index=False)
print("✅ Feature engineering done → saved to engineered_lob2.csv")

print("Label distribution:")
print(df['label'].value_counts())

# Optional: Plot distribution
df['label'].value_counts().plot(kind='bar', title='Label Distribution')
plt.xlabel("Label")
plt.ylabel("Count")
plt.savefig("plot.png")
