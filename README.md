# Task 6 — The Binary Decision
### PlaceMux · Altrodav Technologies · Phase 1 Industry Immersion

## Objective
Build a binary classifier and evaluate it beyond raw accuracy.

## What I Built
- Trained **Logistic Regression** and **Random Forest** classifiers
- Handled **class imbalance** (85% vs 15%) using `class_weight="balanced"`
- Evaluated using **Confusion Matrix, Precision, Recall, F1-Score**
- Plotted **ROC Curve** and **Precision-Recall Curve**
- Selected threshold **0.565** based on business cost reasoning

## Final Results (Random Forest on Test Set)
| Metric | Score |
|---|---|
| Accuracy | 96.8% |
| Precision | 91.4% |
| Recall | 86.9% |
| F1-Score | 89.1% |
| Threshold | 0.565 |

## Tools Used
- Python, scikit-learn, matplotlib, numpy, pandas

## How to Run
pip install scikit-learn matplotlib pandas numpy
python classifier.py

## Output
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/pr_curve.png`
- `results/final_confusion_matrix.png`