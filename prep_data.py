import pandas as pd

# Load the raw Kaggle dataset
df = pd.read_csv('creditcard.csv')

# The 'Time' column is just seconds elapsed since the first transaction
# in the dataset - basically a row index with no fraud signal, so drop it
df = df.drop('Time', axis=1)

# Rename 'Class' to 'class' to match what the main training script expects
df = df.rename(columns={'Class': 'class'})

# Save the cleaned version - this becomes the input for marl_creditcard.py
df.to_csv('data.csv', index=True)

print("Done. data.csv saved.")
print(f"Shape: {df.shape}")
print(f"Fraud rate: {df['class'].mean()*100:.4f}%")
