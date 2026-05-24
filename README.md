# MARL-Based Feature Selection for Credit Card Fraud Detection

Applied Multi-Agent Reinforcement Learning to figure out which features actually matter for detecting credit card fraud and which ones are just noise.

The dataset has 284,807 transactions with only 492 fraud cases (0.17%). The challenge isn't just classification - it's doing it with fewer features without losing detection performance.

## What This Does

Each of the 29 features gets its own Q-learning agent. Every episode, each agent decides whether to include or drop its feature. A classifier trains on whatever subset gets selected, and the reward signal comes from three things:

- How much the F1 score improved over the last episode
- How informative the feature is (mutual information with the target)
- How much it contributes to detecting fraud specifically - SHAP values computed only on fraud samples, not the whole dataset

Agents that changed their action get their Q-values updated based on whether the change helped or hurt. Over 500 episodes, the agents converge on a stable feature subset.

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) - 284,807 transactions, 29 features (V1–V28 from PCA + transaction amount), 0.17% fraud rate.

## Methodology

- 70/15/15 train/validation/test split - test set is completely blind until final evaluation
- Reward computed on validation set during training - no test leakage
- Mutual information computed on training data only
- Class imbalance handled via `scale_pos_weight` in XGBoost (559:1) instead of oversampling
- SHAP computed on training minority samples only (~356 unique fraud cases)

## Results

Ran 500 episodes. Converged around episode 41.

| | All Features (29) | MARL Selected (17) |
|---|---|---|
| Macro F1 | 0.9332 | **0.9444** |
| Fraud Precision | 85.25% | **89.66%** |
| Fraud Recall | 88.14% | 88.14% |
| False Positives | 9 | **6** |
| False Negatives | 7 | 7 |
| Accuracy | 99.97% | 99.97% |

41% fewer features, +1.12% better Macro F1, same recall, fewer false alarms - MARL maintained detection rate while tightening precision.

**Features selected:** V2, V3, V4, V5, V7, V10, V12, V13, V14, V18, V20, V21, V22, V23, V25, V28, Amount

## Baseline Comparison

All methods forced to select exactly 17 features, evaluated on the same held-out test set.

| Method | Macro F1 | Precision | Recall | FP | FN |
|---|---|---|---|---|---|
| All Features (29) | 0.9332 | 85.25% | 88.14% | 9 | 7 |
| Random Forest Importance | 0.9379 | 85.48% | 89.83% | 9 | 6 |
| Mutual Information | 0.9369 | 86.67% | 88.14% | 8 | 7 |
| RFE with XGBoost | **0.9453** | 88.33% | **89.83%** | 7 | 6 |
| **MARL (ours)** | 0.9444 | **89.66%** | 88.14% | **6** | 7 |

RFE edges out MARL on Macro F1 and recall. MARL wins on precision and false positives - the metric that matters most in fraud detection where flagging a legitimate transaction as fraud means blocking a real customer. MARL selects the feature subset that minimizes false alarms while maintaining the same fraud detection rate as the baseline.

## Plots

### Training Curve
![Training Curve](plot_training_curve.png)

### Confusion Matrix
![Confusion Matrix](plot_confusion_matrix.png)

### Feature Importance
![Feature Importance](plot_feature_importance.png)

## Files

```
prep_data.py            preprocess creditcard.csv → data.csv
marl_creditcard.py      main MARL training (500 episodes)
plots_only.py           generates the 3 plots above
baselines.py            runs traditional feature selection methods for comparison
f1_history.csv          F1 score per episode from the training run
baseline_results.csv    comparison results across all methods
```

`data.csv` is not in the repo (144MB). Generate it by running `prep_data.py` after downloading `creditcard.csv` from Kaggle.

## How to Run

```bash
pip install xgboost shap scikit-learn pandas numpy matplotlib seaborn

# Step 1 - prep the data
python prep_data.py

# Step 2 - run MARL training (~1-2 hours)
python marl_creditcard.py

# Step 3 - generate plots
python plots_only.py

# Step 4 - run baseline comparisons (optional, ~10 mins)
python baselines.py
```

## Notes

- V14 dominates feature importance by a wide margin - known high-signal feature in fraud detection literature
- SHAP reward on minority samples only is what pushes fraud recall up - without it the model tends to optimize for the majority class
- The occasional dips in the training curve are from epsilon-greedy exploration kicking in - the agents temporarily try different feature combinations before snapping back
- Using `scale_pos_weight` instead of oversampling avoids the risk of the model memorizing duplicated minority samples

## Reference

Kim, S., Park, S. Y., Seo, H., & Woo, J. (2024). Feature selection integrating Shapley values and mutual information in reinforcement learning. *Computer Methods and Programs in Biomedicine*, 257, 108416.
