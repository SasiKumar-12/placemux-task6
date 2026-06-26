import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_curve, classification_report
)
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
os.makedirs("results", exist_ok=True)

# 1. Create data (imbalanced: 85% class 0, 15% class 1)
X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=6,
    weights=[0.85, 0.15], flip_y=0.01, random_state=SEED
)
print(f"Dataset shape  : {X.shape}")
print(f"Class counts   : {np.bincount(y)}")

# 2. Split into train / validation / test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp)
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# 3. Baseline
print(f"Baseline accuracy (always guess 0): {np.sum(y_val==0)/len(y_val):.3f}")

# 4. Scale + Train
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

lr = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=500)
lr.fit(X_train_s, y_train)

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED)
rf.fit(X_train_s, y_train)

# 5. Evaluate on validation
def evaluate(model, X, y, threshold=0.5, label=""):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    print(f"\n--- {label} ---")
    print(classification_report(y, preds, target_names=["No","Yes"]))
    return proba, preds

lr_proba_val, lr_preds_val = evaluate(lr, X_val_s, y_val, label="Logistic Regression - Validation")
rf_proba_val, rf_preds_val = evaluate(rf, X_val_s, y_val, label="Random Forest - Validation")

# 6. Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, name in zip(axes, [lr_preds_val, rf_preds_val], ["Logistic Regression", "Random Forest"]):
    cm = confusion_matrix(y_val, preds)
    ConfusionMatrixDisplay(cm, display_labels=["No","Yes"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} - Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.show()

# 7. ROC Curve
fig, ax = plt.subplots(figsize=(8, 6))
for proba, name, color in [(lr_proba_val,"Logistic Regression","steelblue"),(rf_proba_val,"Random Forest","darkorange")]:
    fpr, tpr, _ = roc_curve(y_val, proba)
    ax.plot(fpr, tpr, label=f"{name} AUC={auc(fpr,tpr):.3f}", color=color)
ax.plot([0,1],[0,1],"k--", label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend(); ax.grid(alpha=0.3)
plt.savefig("results/roc_curve.png", dpi=150)
plt.show()

# 8. PR Curve + Threshold
fig, ax = plt.subplots(figsize=(8, 6))
for proba, name, color in [(lr_proba_val,"Logistic Regression","steelblue"),(rf_proba_val,"Random Forest","darkorange")]:
    prec, rec, _ = precision_recall_curve(y_val, proba)
    ax.plot(rec, prec, label=name, color=color)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve"); ax.legend(); ax.grid(alpha=0.3)
plt.savefig("results/pr_curve.png", dpi=150)
plt.show()

# Pick best threshold (recall >= 0.85)
prec_rf, rec_rf, thresh_rf = precision_recall_curve(y_val, rf_proba_val)
mask = rec_rf[:-1] >= 0.85
chosen_threshold = thresh_rf[mask][np.argmax(prec_rf[:-1][mask])] if mask.any() else 0.3
print(f"\nChosen threshold: {chosen_threshold:.3f}")

# 9. Final test evaluation
rf_proba_test, rf_preds_test = evaluate(rf, X_test_s, y_test, threshold=chosen_threshold, label=f"Random Forest - TEST (threshold={chosen_threshold:.2f})")

cm_test = confusion_matrix(y_test, rf_preds_test)
ConfusionMatrixDisplay(cm_test, display_labels=["No","Yes"]).plot(colorbar=False)
plt.title(f"Final Test Confusion Matrix (threshold={chosen_threshold:.2f})")
plt.savefig("results/final_confusion_matrix.png", dpi=150)
plt.show()

# 10. Final summary
tn, fp, fn, tp = cm_test.ravel()
print("\n========== FINAL RESULTS ==========")
print(f"Threshold  : {chosen_threshold:.3f}")
print(f"Accuracy   : {accuracy_score(y_test, rf_preds_test):.3f}")
print(f"Precision  : {precision_score(y_test, rf_preds_test):.3f}")
print(f"Recall     : {recall_score(y_test, rf_preds_test):.3f}")
print(f"F1-Score   : {f1_score(y_test, rf_preds_test):.3f}")
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print("====================================")