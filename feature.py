import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt

# Parameters
FUTURE_WINDOW = 10        # seconds ahead
THRESHOLD = 0.0004       # 0.15% price change

# Load CSV
df = pd.read_csv("lob_data2.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp').reset_index(drop=True)

labels = []

for i in range(len(df)):
    current_time = df.loc[i, 'timestamp']
    current_price = df.loc[i, 'mid_price']

    # Get index of first timestamp > current + 5s
    future_index = df['timestamp'].searchsorted(current_time + timedelta(seconds=FUTURE_WINDOW))
    
    if future_index >= len(df):
        labels.append(None)
        continue

    future_price = df.loc[future_index, 'mid_price']
    change_pct = (future_price - current_price) / current_price

    # Debug print (you can comment later)
    print(f"[{i}] Now: {current_price:.2f}, Future: {future_price:.2f}, Change: {change_pct:.5f}")

    if change_pct > THRESHOLD:
        labels.append(1)
    elif change_pct < -THRESHOLD:
        labels.append(-1)
    else:
        labels.append(0)

df['label'] = labels
df.dropna(inplace=True)

# Save processed data
df.to_csv("engineered_lob.csv", index=False)
print("\nFeature engineering done. Saved → engineered_lob.csv")
print("Label distribution:")
print(df['label'].value_counts())

# Optional: Plot distribution
df['label'].value_counts().plot(kind='bar', title='Label Distribution')
plt.xlabel("Label")
plt.ylabel("Count")
plt.savefig("plot.png")
