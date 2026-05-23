import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Load the raw Kaggle dataset
df = pd.read_csv('creditcard.csv')

# The 'Time' column is just seconds elapsed since the first transaction
# in the dataset - basically a row index with no fraud signal, so drop it
df = df.drop('Time', axis=1)

# Rename 'Class' to 'class' to match what the main training script expects
df = df.rename(columns={'Class': 'class'})

# Save the cleaned version - this becomes the input for marl_creditcard.py
df.to_csv('data.csv', index=True)

# Compute mutual information between each feature and the target
# MI scores tell us how much each feature is statistically associated with fraud
# These scores are used as part of the reward function during MARL training
X = df.drop('class', axis=1)
y = df['class']

mi_scores = mutual_info_classif(X, y, random_state=42)

mi_df = pd.DataFrame({
    'col_name': X.columns,
    'mutual_info': mi_scores
})

mi_df.to_csv('mutual_info_result.csv', index=True)

print("Done. Files saved: data.csv, mutual_info_result.csv")
print(mi_df.sort_values('mutual_info', ascending=False))
