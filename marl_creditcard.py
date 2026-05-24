import shap
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report, confusion_matrix

random.seed(42)
np.random.seed(42)

# Load preprocessed data
df = pd.read_csv('data.csv', index_col=0, encoding='UTF-8')

# Remove any duplicate columns that might have crept in during preprocessing
drop_list = list(set(df.columns) - set(df.T.drop_duplicates(keep='first').T.columns))
df = df.drop(drop_list, axis=1)

X = df.drop(['class'], axis=1)
y = pd.DataFrame([0 if x == 0 else 1 for x in df['class']], columns=['class'])

# Three-way split: train / validation / test
# Validation is used during MARL training for reward signal
# Test set is completely blind until final evaluation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Compute mutual information on training data only
# Using the full dataset here would leak test information into the reward signal
print("Computing mutual information on training data...")
mi_scores = mutual_info_classif(X_train, y_train.values.ravel(), random_state=42)
mutual_df = pd.DataFrame({
    'col_name': X_train.columns,
    'mutual_info': mi_scores
}).reset_index(drop=True)

# scale_pos_weight tells XGBoost how much to penalize missing a fraud case
# relative to a false alarm — handles class imbalance without oversampling
neg = len(y_train[y_train['class'] == 0])
pos = len(y_train[y_train['class'] == 1])
scale_pos_weight = neg / pos

print(f"Dataset     : {df.shape[0]} rows, {X.shape[1]} features")
print(f"Train       : {len(X_train)} rows ({neg} not-fraud, {pos} fraud)")
print(f"Validation  : {len(X_val)} rows")
print(f"Test        : {len(X_test)} rows (blind until final eval)")
print(f"scale_pos_weight: {round(scale_pos_weight, 2)}")


def reward_weight(input_features):
    x_train = X_train.iloc[:, input_features]
    x_val   = X_val.iloc[:, input_features]

    clf = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=6,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    clf.fit(x_train, y_train.values.ravel())

    # Reward is computed on the validation set — test set never touched here
    pred = clf.predict(x_val)
    f1 = f1_score(y_val.values.ravel(), pred, average='macro')

    # SHAP computed on training minority samples only
    # These are the unique fraud cases in X_train — fast and no leakage
    minority_mask = y_train['class'] == 1
    minority_train = x_train[minority_mask.values]

    if len(minority_train) == 0:
        shap_weight = [0.0] * len(input_features)
    else:
        explainer = shap.Explainer(clf)
        shap_values = explainer(minority_train)
        shap_result = pd.DataFrame(shap_values.values)
        shap_weight = list(shap_result.abs().mean())

    print("F1_score:", round(f1, 4))
    return f1, clf.feature_importances_, shap_weight


def get_reward(features):
    if len(features) == 0:
        return 0, [], []
    f1, importance, shap_weight = reward_weight(features)
    return f1 * 100, importance, shap_weight


# Each feature gets its own Q-learning agent with two possible actions:
# action 0 = deselect this feature, action 1 = select this feature
num_agents = X_train.shape[1]
Q_values = [[-1, -1] for _ in range(num_agents)]

epsilon = 0.05
alpha = 0.2
epsilon_decay_rate = 0.995
alpha_decay_rate = 0.995

num_episodes = 500
current_actions = [0] * num_agents
previous_action = [1] * num_agents
previous_R = 47.0

# Log F1 per episode so plots_only.py can read actual training history
f1_history = []

for episode in range(num_episodes):
    print("--------------------------------------------------")
    print("episode:", episode)

    for agent in range(num_agents):
        rand_val = random.uniform(0, 1)
        current_actions[agent] = (
            np.argmax(Q_values[agent]) if rand_val > epsilon
            else random.choice([0, 1])
        )

    selected_features = [i for i, a in enumerate(current_actions) if a == 1]
    Current_R, feature_importance, shap_weight = get_reward(selected_features)
    print("R:", Current_R)

    f1_history.append({'episode': episode, 'f1': round(Current_R, 4)})

    # Only update agents that changed their action since last episode
    different_indices = [
        i for i in range(len(current_actions))
        if current_actions[i] != previous_action[i]
    ]

    Improvement_R = Current_R - previous_R
    divide_R = Improvement_R / len(different_indices) if len(different_indices) > 0 else 0

    # Map SHAP weights back to full feature index space
    mapped_shap_weight = np.zeros(len(current_actions))
    for i, weight in zip([i for i, x in enumerate(current_actions) if x == 1], shap_weight):
        mapped_shap_weight[i] = weight

    total_mapped_shap_weight = sum(mapped_shap_weight)
    mutual_info_values = mutual_df.loc[different_indices, 'mutual_info']
    total_mutual_info_values = sum(mutual_info_values)

    # Q-value update: reward = F1 improvement + MI weight + SHAP weight
    for agent in different_indices:
        if Improvement_R > 0:
            if total_mapped_shap_weight > 0 and total_mutual_info_values > 0:
                Q_values[agent][current_actions[agent]] += alpha * (
                    divide_R
                    + (mapped_shap_weight[agent] / total_mapped_shap_weight)
                    + (mutual_info_values[agent] / total_mutual_info_values)
                    - Q_values[agent][current_actions[agent]]
                )
            else:
                Q_values[agent][current_actions[agent]] += alpha * (
                    divide_R - Q_values[agent][current_actions[agent]]
                )
        else:
            Q_values[agent][current_actions[agent]] += alpha * (
                divide_R - Q_values[agent][current_actions[agent]]
            )

    previous_R = Current_R
    previous_action = current_actions.copy()
    alpha *= alpha_decay_rate
    epsilon *= epsilon_decay_rate

    print("alpha:", round(alpha, 6))
    print("epsilon:", round(epsilon, 6))

# Save training history so plots_only.py can use actual run data
pd.DataFrame(f1_history).to_csv('f1_history.csv', index=False)
print("Training history saved to f1_history.csv")

# Pick the final feature set based on which action each agent settled on
last_feature = [np.argmax(Q_values[i]) for i in range(num_agents)]
selected_features = [X_train.columns[i] for i in range(num_agents) if last_feature[i] == 1]

print("\n==================================================")
print("FINAL RESULTS")
print("==================================================")
print(f"Selected features ({len(selected_features)}): {selected_features}")

# Final evaluation on test set — first and only time test data is used
clf_final = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    n_estimators=100,
    max_depth=6,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)
clf_final.fit(X_train[selected_features], y_train.values.ravel())

pred = clf_final.predict(X_test[selected_features])

print("F1_score:", round(f1_score(y_test.values.ravel(), pred, average='macro'), 4))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["Not Fraud", "Fraud"], digits=4))
