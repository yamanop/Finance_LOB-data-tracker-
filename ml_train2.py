import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample

# 📥 Load engineered data
df = pd.read_csv("engineered_lob2.csv")

# 🔁 Label encode: -1 → 0, 0 → 1, 1 → 2
label_map = {-1: 0, 0: 1, 1: 2}
reverse_map = {0: -1, 1: 0, 2: 1}
df['label'] = df['label'].map(label_map)

# 🎯 Define features
features = [
    'bid_price', 'ask_price',
    'bid_volume', 'ask_volume',
    'spread', 'imbalance'
]

# ⚖️ Split and balance each class
df_0 = df[df['label'] == 0]  # -1 → DOWN
df_1 = df[df['label'] == 1]  # 0 → STABLE
df_2 = df[df['label'] == 2]  # +1 → UP

target_count = max(len(df_0), len(df_1), len(df_2))

df_0_bal = resample(df_0, replace=True, n_samples=target_count, random_state=42)
df_1_bal = resample(df_1, replace=True, n_samples=target_count, random_state=42)
df_2_bal = resample(df_2, replace=True, n_samples=target_count, random_state=42)

# ✅ Merge balanced data
df_balanced = pd.concat([df_0_bal, df_1_bal, df_2_bal])

# 🔀 Train-test split on balanced data
X = df_balanced[features]
y = df_balanced['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ⚙️ Train XGBoost classifier
clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=150,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.8,
    random_state=42
)
clf.fit(X_train, y_train)

# 🔮 Predict and decode labels
y_pred = clf.predict(X_test)
y_test_decoded = y_test.map(reverse_map)
y_pred_decoded = pd.Series(y_pred).map(reverse_map)

# ✅ Accuracy and metrics
acc = accuracy_score(y_test_decoded, y_pred_decoded)
print(f"\n Accuracy: {acc:.4f}\n")
print(" Classification Report:")
print(classification_report(y_test_decoded, y_pred_decoded))

# 📊 Confusion Matrix
cm = confusion_matrix(y_test_decoded, y_pred_decoded)
labels = [-1, 0, 1]

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='crest', xticklabels=labels, yticklabels=labels)
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix6.png", dpi=300)
import joblib
joblib.dump(clf, "btc_xgboost_model.joblib")



