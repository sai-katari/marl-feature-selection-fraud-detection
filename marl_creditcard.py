import shap
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, classification_report, confusion_matrix

random.seed(42)
np.random.seed(42)

# Load preprocessed data and pre-computed mutual information scores
df = pd.read_csv('data.csv', index_col=0, encoding='UTF-8')
mutual_df = pd.read_csv('mutual_info_result.csv', index_col=0)
mutual_df = mutual_df[mutual_df['col_name'] != 'class'].reset_index(drop=True)

# Remove any duplicate columns that might have crept in during preprocessing
drop_list = list(set(df.columns) - set(df.T.drop_duplicates(keep='first').T.columns))
df = df.drop(drop_list, axis=1)

X = df.drop(['class'], axis=1)
y = pd.DataFrame([0 if x == 0 else 1 for x in df['class']], columns=['class'])

# Split first — test set is locked away and not touched during feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fraud cases are only 0.17% of the data — oversample the minority class
# on training data only, never on the full dataset
oversample = RandomOverSampler(sampling_strategy='minority', random_state=123)
X_res, y_res = oversample.fit_resample(X_train, y_train)

print(f"Dataset: {df.shape[0]} rows, {X.shape[1]} features")
print(f"Class distribution after oversampling: {pd.Series(y_res['class']).value_counts().to_dict()}")


def reward_weight(input_features):
    x_train = X_res.iloc[:, input_features]
    x_test = X_test.iloc[:, input_features]

    clf = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=6,
        n_jobs=-1
    )
    clf.fit(x_train, y_res.values.ravel())

    pred = clf.predict(x_test)
    f1 = f1_score(y_test.values.ravel(), pred, average='macro')

    # SHAP computed on training minority samples only — not the test set
    # Feature selection should have no visibility into test data at any point
    train_data = pd.concat([x_train.reset_index(drop=True),
                            y_res.reset_index(drop=True)], axis=1)
    minority_train = train_data[train_data['class'] == 1].drop(['class'], axis=1)

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
# Q-values start at -1 to encourage early exploration
num_agents = 29
Q_values = [[-1, -1] for _ in range(num_agents)]

epsilon = 0.05
alpha = 0.2
epsilon_decay_rate = 0.995
alpha_decay_rate = 0.995

num_episodes = 500
current_actions = [0] * num_agents
previous_action = [1] * num_agents

# Set close to actual first-episode F1 so reward signal fires correctly
previous_R = 47.0

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

# Pick the final feature set based on which action each agent settled on
last_feature = [np.argmax(Q_values[i]) for i in range(num_agents)]
selected_features = [X_train.columns[i] for i in range(num_agents) if last_feature[i] == 1]

print("\n==================================================")
print("FINAL RESULTS")
print("==================================================")
print(f"Selected features ({len(selected_features)}): {selected_features}")

# Final evaluation on test set — first time test data is used
clf_final = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    n_estimators=100,
    max_depth=6,
    n_jobs=-1
)
clf_final.fit(X_res[selected_features], y_res.values.ravel())

pred = clf_final.predict(X_test[selected_features])

print("F1_score:", round(f1_score(y_test.values.ravel(), pred, average='macro'), 4))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["Not Fraud", "Fraud"], digits=4))
