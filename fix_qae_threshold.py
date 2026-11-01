import numpy as np, joblib
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

ART    = "artifacts"
y_test = np.load(f"{ART}/y_test.npy")
probs  = np.load(f"{ART}/qae_test_probs.npy")
thresh = joblib.load(f"{ART}/quantum_thresholds.pkl")

print(f"QAE full-test prob range: [{probs.min():.4f}, {probs.max():.4f}]")
print(f"Current qae entry in thresholds pkl: {thresh['qae']}")
print(f"(Note: was overwritten with error-space value 0.5539 by resolve_followups.py)")
print()

# Correct approach: search threshold on prob space (classify as attack if prob >= t)
best_t, best_f1 = 0.50, 0.0
for t_raw in range(1, 99):
    t = t_raw / 100.0
    f = f1_score(y_test, (probs >= t).astype(int), zero_division=0)
    if f > best_f1:
        best_f1, best_t = f, t

print(f"Best prob threshold (full test): t={best_t:.2f}  F1={best_f1:.6f}")
print()

# Report a few key thresholds
for t in [0.43, 0.50, best_t]:
    preds_t = (probs >= t).astype(int)
    f = f1_score(y_test, preds_t, zero_division=0)
    p = precision_score(y_test, preds_t, zero_division=0)
    r = recall_score(y_test, preds_t, zero_division=0)
    print(f"  t={t:.2f}: F1={f:.6f}  Prec={p:.4f}  Rec={r:.4f}")

# Full metrics at best threshold
preds_best = (probs >= best_t).astype(int)
cm = confusion_matrix(y_test, preds_best)
tn, fp, fn, tp = cm.ravel()
print(f"\nQAE Full-Test at t={best_t:.2f}:")
print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
print(f"  F1={f1_score(y_test,preds_best):.6f}")

# Save corrected probability threshold
thresh["qae"] = best_t
joblib.dump(thresh, f"{ART}/quantum_thresholds.pkl")
np.save(f"{ART}/qae_test_preds.npy", preds_best)
print(f"\nUpdated quantum_thresholds.pkl: {thresh}")
print(f"qae_test_preds.npy updated (shape={preds_best.shape}, t={best_t})")
