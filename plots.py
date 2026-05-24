import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Exact same split as marl_creditcard.py
# Any difference here would mean evaluating on a different test set
df = pd.read_csv('data.csv', index_col=0, encoding='UTF-8')
X = df.drop(['class'], axis=1)
y = pd.DataFrame([0 if x == 0 else 1 for x in df['class']], columns=['class'])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

neg = len(y_train[y_train['class'] == 0])
pos = len(y_train[y_train['class'] == 1])
scale_pos_weight = neg / pos

print("Data loaded. Training models for plots...")

def get_xgb():
    return xgb.XGBClassifier(
        random_state=42, eval_metric='logloss',
        n_estimators=100, max_depth=6, n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

# Updated to match the 17 features selected by MARL in the latest run
marl_features = [
    'V2', 'V3', 'V4', 'V5', 'V7', 'V10', 'V12', 'V13', 'V14',
    'V18', 'V20', 'V21', 'V22', 'V23', 'V25', 'V28', 'Amount'
]

# Train baseline on all features — evaluated on same test set as MARL run
print("Training baseline (all features)...")
clf_all = get_xgb()
clf_all.fit(X_train, y_train.values.ravel())
pred_all = clf_all.predict(X_test)
cm_all = confusion_matrix(y_test.values.ravel(), pred_all)
f1_all = f1_score(y_test.values.ravel(), pred_all, average='macro')

# Train on MARL-selected 17 features — same test set
print("Training MARL model (17 features)...")
clf_marl = get_xgb()
clf_marl.fit(X_train[marl_features], y_train.values.ravel())
pred_marl = clf_marl.predict(X_test[marl_features])
cm_marl = confusion_matrix(y_test.values.ravel(), pred_marl)
f1_marl = f1_score(y_test.values.ravel(), pred_marl, average='macro')

print(f"Baseline F1 : {round(f1_all, 4)}")
print(f"MARL F1     : {round(f1_marl, 4)}")
print("Building plots...")

# Load actual training history exported by marl_creditcard.py
f1_log = pd.read_csv('f1_history.csv')
episodes = f1_log['episode'].tolist()
f1_curve = f1_log['f1'].tolist()

plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})
c_marl = '#1565C0'

# Plot 1: Training curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(episodes, f1_curve, color=c_marl, linewidth=1.5, alpha=0.85, label='MARL F1')
ax.axhline(y=f1_all * 100, color='red', linestyle='--',
           linewidth=1.3, label=f'All Features Baseline ({round(f1_all, 4)})')
ax.axhline(y=f1_marl * 100, color='green', linestyle='--',
           linewidth=1.3, label=f'MARL Converged ({round(f1_marl, 4)})')
ax.set_xlabel('Episode')
ax.set_ylabel('Macro F1 Score (%)')
ax.set_title('MARL Training Curve — Credit Card Fraud Detection', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, max(episodes))
ax.set_ylim(40, 100)
plt.tight_layout()
plt.savefig('plot_training_curve.png', dpi=150, bbox_inches='tight')
print("Saved: plot_training_curve.png")
plt.close()

# Plot 2: Side-by-side confusion matrices
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle('Confusion Matrices — All Features vs MARL Selected',
              fontsize=13, fontweight='bold')
for ax, cm, title in zip(
    axes2,
    [cm_all, cm_marl],
    [f'All Features — {X.shape[1]} (Baseline)', f'MARL Selected — {len(marl_features)} Features']
):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'],
                linewidths=0.5, cbar=False, annot_kws={'size': 13})
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('plot_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved: plot_confusion_matrix.png")
plt.close()

# Plot 3: Feature importance — V14 dominates by a wide margin
print("Computing feature importance...")
importances = clf_marl.feature_importances_
feat_imp = pd.Series(importances, index=marl_features).sort_values(ascending=True)

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.barh(feat_imp.index, feat_imp.values,
         color=c_marl, edgecolor='black', linewidth=0.5)
ax3.set_xlabel('Feature Importance (XGBoost)')
ax3.set_title('MARL Selected Features — Importance Ranking', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('plot_feature_importance.png', dpi=150, bbox_inches='tight')
print("Saved: plot_feature_importance.png")
plt.close()

print("\nAll 3 plots saved.")
