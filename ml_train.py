# train_xgboost_balanced.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess
df = pd.read_csv("engineered_lob.csv")
X = df[['spread', 'imbalance']]
y = df['label']

# Remap labels for XGBoost
label_map = {-1: 0, 0: 1, 1: 2}
reverse_map = {0: -1, 1: 0, 2: 1}
df['label'] = df['label'].map(label_map)

# Split classes
df_0 = df[df['label'] == 0]  # DOWN
df_1 = df[df['label'] == 1]  # STABLE
df_2 = df[df['label'] == 2]  # UP

# Count how many samples we want per class (balance based on STABLE count)
target_count = max(len(df_0), len(df_1), len(df_2))

df_0_bal = resample(df_0, replace=True, n_samples=target_count, random_state=42)
df_1_bal = resample(df_1, replace=True, n_samples=target_count, random_state=42)
df_2_bal = resample(df_2, replace=True, n_samples=target_count, random_state=42)

# Combine
df_balanced = pd.concat([df_0_bal, df_1_bal, df_2_bal])
X_bal = df_balanced[['spread', 'imbalance']]
y_bal = df_balanced['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

# Train XGBoost
clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_pred_decoded = pd.Series(y_pred).map(reverse_map)
y_test_decoded = y_test.map(reverse_map)

# Evaluation
print("Classification Report:\n", classification_report(y_test_decoded, y_pred_decoded))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test_decoded, y_pred_decoded), annot=True, fmt='d')
plt.title("XGBoost Confusion Matrix (Balanced)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix2.png")

# Save model
joblib.dump(clf, "xgb_model_balanced.pkl")
print("Balanced XGBoost model saved as xgb_model_balanced.pkl")

# shap_plot.py
import pandas as pd
import joblib
import shap_analysis
import matplotlib.pyplot as plt

# Load model and data
df = pd.read_csv("engineered_lob.csv")
X = df[['spread', 'imbalance']]
model = joblib.load(".pkl")

# SHAP
explainer = shap_analysis.Explainer(model, X)
shap_values = explainer(X)

# Summary Plot
shap_analysis.summary_plot(shap_values, X)

# Optional: Bar Plot
shap_analysis.plots.bar(shap_values)

