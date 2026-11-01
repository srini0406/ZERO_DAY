"""Diagnostic 1: Threshold tuning on existing saved probs (no retraining)."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_test_full = np.load("artifacts/y_test.npy")
qsnn_probs  = np.load("artifacts/qsnn_test_probs.npy")
qmlp_probs  = np.load("artifacts/qmlp_test_probs.npy")
qae_probs   = np.load("artifacts/qae_test_probs.npy")

# Reconstruct QMLP/QAE subsample labels using the same RNG seed used in training
rng = np.random.default_rng(42)
atk_te = np.where(y_test_full == 1)[0]
nrm_te = np.where(y_test_full == 0)[0]
te_idx = np.concatenate([
    rng.choice(atk_te, 500, replace=False),
    rng.choice(nrm_te, 500, replace=False)
])
y_sub = y_test_full[te_idx]
print(f"QMLP/QAE subsample: attack={int((y_sub==1).sum())}  normal={int((y_sub==0).sum())}")


def tune_and_report(probs, y_true, model_name, current_t=0.5):
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    def metrics_at(t):
        p = (probs >= t).astype(int)
        return (accuracy_score(y_true, p),
                precision_score(y_true, p, zero_division=0),
                recall_score(y_true, p, zero_division=0),
                f1_score(y_true, p, zero_division=0))

    acc, prec, rec, f1 = metrics_at(current_t)
    print(f"  At t={current_t:.2f}: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

    # Full threshold scan
    best_t, best_f1 = current_t, 0.0
    scan = []
    for t in np.arange(0.05, 0.96, 0.02):
        a, p, r, f = metrics_at(t)
        scan.append((round(float(t), 2), a, p, r, f))
        if f > best_f1:
            best_f1, best_t = f, round(float(t), 2)

    acc, prec, rec, f1 = metrics_at(best_t)
    delta_f1   = f1   - metrics_at(current_t)[3]
    delta_prec = prec - metrics_at(current_t)[1]
    delta_rec  = rec  - metrics_at(current_t)[2]
    print(f"  At t={best_t:.2f}: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    print(f"  Delta vs t=0.5:  dPrec={delta_prec:+.4f}  dRec={delta_rec:+.4f}  dF1={delta_f1:+.4f}")

    # Print scan around optimal
    print(f"\n  Threshold scan (window around best):")
    print(f"  {'T':>5}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
    for (t, a, p, r, f) in scan:
        if best_t - 0.14 <= t <= best_t + 0.14:
            marker = " <-- best" if t == best_t else ""
            print(f"  {t:>5.2f}  {a:>7.4f}  {p:>7.4f}  {r:>7.4f}  {f:>7.4f}{marker}")
    return best_t


print("\n" + "="*60)
print("DIAGNOSTIC 1: THRESHOLD TUNING (no retraining needed)")
print("="*60)

best_t_qsnn = tune_and_report(qsnn_probs, y_test_full, "QSNN  (full test set, n=82332)")
best_t_qmlp = tune_and_report(qmlp_probs, y_sub,       "QMLP  (1000-sample subsample)")
best_t_qae  = tune_and_report(qae_probs,  y_sub,       "QAE   (1000-sample subsample)")

print(f"\n  Summary: QSNN best_t={best_t_qsnn}  QMLP best_t={best_t_qmlp}  QAE best_t={best_t_qae}")
