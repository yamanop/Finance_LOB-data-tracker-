import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib

# 🧠 Load trained model (ya use already trained clf)
# clf = joblib.load("xgb_model.joblib")  # agar saved hai

# 📥 Load dataset again
df = pd.read_csv("engineered_lob2.csv")
df['label'] = df['label'].map({-1: 0, 0: 1, 1: 2})

features = [
    'bid_price', 'ask_price',
    'bid_volume', 'ask_volume',
    'spread', 'imbalance'
]

X = df[features]

# ⚙️ Retrain model (ya load trained one)
clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
clf.fit(X, df['label'])

# 📊 Feature importance
importances = clf.feature_importances_
indices = importances.argsort()[::-1]
sorted_features = [features[i] for i in indices]

plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=sorted_features, palette="viridis")
plt.title("XGBoost Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)

