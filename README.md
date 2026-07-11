# Mental Health Severity Prediction — Depression, Anxiety & Stress

Reproduction and critical extension of *"Predicting Mental Health Outcomes: A Machine 
Learning Approach to Depression, Anxiety, and Stress"* (Norouzi & Machado, 2024), using 
the DASS-21 dataset (39,775 responses, OpenPsychometrics).

## Key finding: identified and fixed label-derivation leakage

The original paper (and my first reproduction, Track A below) predicts DASS severity 
using the *same* question items that are summed to compute the severity label — making 
the task an arithmetic lookup rather than genuine prediction. This explains the ~99–100% 
accuracy reported by the paper and reproduced here.

To get a methodologically honest result, I re-engineered the task (Track B) to predict 
severity from **demographic and personality (TIPI) features only** — independent of the 
scored items — and benchmarked against a majority-class baseline.

## Results

| Scale | Track A (item-based, inflated) | Track B (demographic, realistic) | Majority-class baseline |
|---|---|---|---|
| Depression | F1 = 1.000 | **F1 = 0.684** | F1 = 0.559 |
| Anxiety | F1 = 1.000 | F1 = 0.763 | F1 = 0.742 |
| Stress | F1 = 0.999 | **F1 = 0.588** | F1 = 0.432 |

**Depression and Stress:** demographic/personality features carry real predictive signal — 
Random Forest meaningfully beats the baseline on both.

**Anxiety:** the demographic model (82.2% accuracy) is essentially tied with the baseline 
(82.3% accuracy). Age, gender, and personality traits alone don't meaningfully predict 
anxiety severity in this dataset — a genuine negative result, reported rather than hidden.

## Methodology
- Models: KNN, SVM (RBF kernel), Random Forest — tuned via GridSearchCV
- 5-fold stratified cross-validation
- Class imbalance handled via `class_weight="balanced"`
- Random Forest feature importance analysis per scale
- Fixed a secondary preprocessing leak (StandardScaler previously fit on the full 
  dataset before the train/test split)

## Tech stack
Python, scikit-learn, pandas, NumPy, matplotlib, seaborn

## Dataset
[DASS Responses — Kaggle](https://www.kaggle.com/datasets/lucasgreenwell/depression-anxiety-stress-scales-responses)

## Setup
\`\`\`bash
pip install -r requirements.txt
jupyter notebook model_training_fixed.ipynb
\`\`\`
