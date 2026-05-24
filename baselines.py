import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Exact same split as marl_creditcard.py and plots_only.py
df = pd.read_csv('data.csv', index_col=0, encoding='UTF-8')
X = df.drop(['class'], axis=1)
y = pd.DataFrame([0 if x == 0 else 1 for x in df['class']], columns=['class'])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

neg = len(y_train[y_train['class'] == 0])
pos = len(y_train[y_train['class'] == 1])
scale_pos_weight = neg / pos

# Number of features to select — matches MARL output
N_FEATURES = 17

def evaluate(X_tr, X_te, y_tr, y_te, label, features):
    clf = xgb.XGBClassifier(
        random_state=42, eval_metric='logloss',
        n_estimators=100, max_depth=6, n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    clf.fit(X_tr, y_tr.values.ravel())
    pred = clf.predict(X_te)

    f1 = f1_score(y_te.values.ravel(), pred, average='macro')
    cm = confusion_matrix(y_te.values.ravel(), pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n{label}")
    print(f"  Features    : {len(features)}")
    print(f"  Selected    : {features}")
    print(f"  Macro F1    : {round(f1, 4)}")
    print(f"  Precision   : {round(precision * 100, 2)}%")
    print(f"  Recall      : {round(recall * 100, 2)}%")
    print(f"  FP / FN     : {fp} / {fn}")

    return {
        'method': label,
        'n_features': len(features),
        'macro_f1': round(f1, 4),
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'features': features
    }

results = []

print("="*60)
print("BASELINE COMPARISONS")
print(f"All methods select top {N_FEATURES} features, same train/test split")
print("="*60)

# 1. All features baseline
print("\nAll Features (Baseline)...")
all_features = list(X.columns)
r = evaluate(X_train, X_test, y_train, y_test, 'All Features (29)', all_features)
results.append(r)

# 2. Random Forest feature importance top-17
print("\nRandom Forest Importance...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                            class_weight='balanced')
rf.fit(X_train, y_train.values.ravel())
rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
rf_features = list(rf_importances.nlargest(N_FEATURES).index)
r = evaluate(X_train[rf_features], X_test[rf_features],
             y_train, y_test, f'Random Forest Importance (top {N_FEATURES})', rf_features)
results.append(r)

# 3. Mutual Information top-17
print("\nMutual Information...")
mi_scores = mutual_info_classif(X_train, y_train.values.ravel(), random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns)
mi_features = list(mi_series.nlargest(N_FEATURES).index)
r = evaluate(X_train[mi_features], X_test[mi_features],
             y_train, y_test, f'Mutual Information (top {N_FEATURES})', mi_features)
results.append(r)

# 4. RFE with XGBoost
print("\nRFE with XGBoost (this may take a few minutes)...")
xgb_rfe = xgb.XGBClassifier(
    random_state=42, eval_metric='logloss',
    n_estimators=50, max_depth=4, n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)
rfe = RFE(xgb_rfe, n_features_to_select=N_FEATURES)
rfe.fit(X_train, y_train.values.ravel())
rfe_features = list(X.columns[rfe.support_])
r = evaluate(X_train[rfe_features], X_test[rfe_features],
             y_train, y_test, f'RFE with XGBoost (top {N_FEATURES})', rfe_features)
results.append(r)

# 5. MARL results
print("\nMARL (our method)...")
marl_features = [
    'V2', 'V3', 'V4', 'V5', 'V7', 'V10', 'V12', 'V13', 'V14',
    'V18', 'V20', 'V21', 'V22', 'V23', 'V25', 'V28', 'Amount'
]
r = evaluate(X_train[marl_features], X_test[marl_features],
             y_train, y_test, f'MARL (ours, {N_FEATURES} features)', marl_features)
results.append(r)

# Summary table
print("\n" + "="*75)
print(f"{'Method':<35} {'F1':>7} {'Precision':>10} {'Recall':>8} {'FP':>5} {'FN':>5}")
print("="*75)
best_f1 = max(r['macro_f1'] for r in results)
for r in results:
    marker = " *" if r['macro_f1'] == best_f1 else ""
    print(f"{r['method']:<35} {r['macro_f1']:>7} "
          f"{str(r['precision'])+'%':>10} "
          f"{str(r['recall'])+'%':>8} "
          f"{r['false_positives']:>5} "
          f"{r['false_negatives']:>5}{marker}")
print("="*75)
print("* best Macro F1")

# Save results
results_df = pd.DataFrame(results).drop('features', axis=1)
results_df.to_csv('baseline_results.csv', index=False)
print("\nSaved: baseline_results.csv")
