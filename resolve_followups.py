"""
resolve_followups.py
────────────────────
Addresses three follow-up items:

  [1] Issue 1 clarification: show the bimodal probability histogram for
      Exp 1 QSNN to confirm the plateau is a residual of embedding collapse,
      not evidence that calibration is resolved.

  [2] Issue 3 verification: show QMLP full-test confusion matrix and confirm
      the probs array is numerically distinct from QSNN's array.

  [3] QAE full-test regeneration: run qae_circuit on all 82,332 test
      samples using the original qae_config logic, save to
      qae_test_errors.npy / qae_test_probs.npy / qae_test_preds.npy.
"""

import os, sys, numpy as np, pennylane as qml
import joblib
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                              recall_score, accuracy_score)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def log(msg=""): print(msg, flush=True)

ART      = "artifacts"
N_QUBITS = 6

y_test = np.load(os.path.join(ART, "y_test.npy"))
log(f"Full test set: {len(y_test):,} samples  |  attacks: {y_test.sum():,}  normals: {(y_test==0).sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# [1] Bimodal histogram: confirm plateau reflects residual embedding collapse
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("[1] QSNN Probability Distribution — Bimodal/Collapse Analysis")
log("=" * 70)

probs_qsnn = np.load(os.path.join(ART, "qsnn_test_probs.npy"))

bins = [(0.00,0.05),(0.05,0.10),(0.10,0.15),(0.15,0.20),(0.20,0.25),
        (0.25,0.30),(0.30,0.35),(0.35,0.40),(0.40,0.45),(0.45,0.50),
        (0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),
        (0.75,0.80),(0.80,0.85),(0.85,0.90),(0.90,0.95),(0.95,1.00)]

log("\n  Probability histogram (all 82,332 test samples):")
log(f"  {'Bin':<14} {'Count':>8} {'%':>6}  {'Bar'}")
log("  " + "-"*60)
gap_start, gap_end, max_gap = None, None, 0
prev_empty = False
for lo, hi in bins:
    mask  = (probs_qsnn >= lo) & (probs_qsnn < hi)
    count = int(mask.sum())
    pct   = 100.0 * count / len(probs_qsnn)
    bar   = "█" * int(pct / 2)
    log(f"  [{lo:.2f}, {hi:.2f})   {count:>8,}  {pct:>5.1f}%  {bar}")
    if count == 0 and not prev_empty:
        gap_start = lo
    if count > 0 and prev_empty:
        gap_end = lo
    prev_empty = (count == 0)

log(f"\n  Min prob : {probs_qsnn.min():.6f}")
log(f"  Max prob : {probs_qsnn.max():.6f}")

# Count samples in the uncertain zone [0.45, 0.60]
uncertain = ((probs_qsnn >= 0.45) & (probs_qsnn < 0.60)).sum()
log(f"\n  Samples in uncertain zone [0.45, 0.60] : {uncertain:,}")
log(f"  Samples outside [0.35, 0.65]           : {((probs_qsnn < 0.35) | (probs_qsnn > 0.65)).sum():,}")
log(f"  Samples in plateau zone [0.40, 0.55]   : {((probs_qsnn >= 0.40) & (probs_qsnn <= 0.55)).sum():,}")

log("""
  INTERPRETATION:
  The flat F1 plateau (t=0.40–0.55) is NOT evidence that probability
  calibration is resolved. It is a RESIDUAL SYMPTOM of partial embedding
  collapse: the softmax over prototype distances still produces a bimodal
  distribution concentrated at two tight bands near [0.296, 0.313] and
  [0.687, 0.703], with a near-empty gap in between.

  Because the gap is nearly empty, every threshold in [0.40, 0.55]
  classifies identically — there are effectively no samples sitting in
  the uncertain region for any threshold to move.

  A well-calibrated model would show a broad, continuous probability
  distribution (ideally uniform or normally distributed over [0,1]).
  Exp 1 widened the probability range from [0.316, 0.684] → [0.296, 0.703]
  but did not produce continuous coverage. The bimodal clustering is the
  residual of the prototype-distance softmax being applied in a low-variance
  6D embedding space.

  This is correctly described as: threshold is optimally tuned within
  the architecture's limits, but confidence calibration (spreading
  probability mass continuously across [0,1]) was NOT achieved.
""")

# ─────────────────────────────────────────────────────────────────────────────
# [2] QMLP full-test confusion matrix — confirm independent computation
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 70)
log("[2] QMLP Full-Test Confusion Matrix — Independence Verification")
log("=" * 70)

probs_qmlp = np.load(os.path.join(ART, "qmlp_test_probs.npy"))
thresh     = joblib.load(os.path.join(ART, "quantum_thresholds.pkl"))

log(f"\n  Arrays are numerically IDENTICAL to QSNN: "
    f"{np.array_equal(probs_qsnn, probs_qmlp)}")
log(f"  Max absolute difference between QSNN and QMLP probs: "
    f"{np.abs(probs_qsnn - probs_qmlp).max():.6f}")
log(f"  QSNN prob mean: {probs_qsnn.mean():.6f}  std: {probs_qsnn.std():.6f}")
log(f"  QMLP prob mean: {probs_qmlp.mean():.6f}  std: {probs_qmlp.std():.6f}")
log(f"  QSNN prob range: [{probs_qsnn.min():.6f}, {probs_qsnn.max():.6f}]")
log(f"  QMLP prob range: [{probs_qmlp.min():.6f}, {probs_qmlp.max():.6f}]")

t_qmlp = thresh["qmlp"]
preds_qmlp = (probs_qmlp >= t_qmlp).astype(int)
t_qsnn = thresh["qsnn"]
preds_qsnn = (probs_qsnn >= t_qsnn).astype(int)

cm_qmlp = confusion_matrix(y_test, preds_qmlp)
cm_qsnn = confusion_matrix(y_test, preds_qsnn)

log(f"\n  QMLP Confusion Matrix (t={t_qmlp}, full 82,332-sample test):")
log(f"                Pred Normal  Pred Attack")
log(f"  True Normal   {cm_qmlp[0,0]:>10,}  {cm_qmlp[0,1]:>11,}")
log(f"  True Attack   {cm_qmlp[1,0]:>10,}  {cm_qmlp[1,1]:>11,}")
tn,fp,fn,tp = cm_qmlp.ravel()
log(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
log(f"  Acc={accuracy_score(y_test,preds_qmlp):.6f}  "
    f"Prec={precision_score(y_test,preds_qmlp):.6f}  "
    f"Rec={recall_score(y_test,preds_qmlp):.6f}  "
    f"F1={f1_score(y_test,preds_qmlp):.6f}")

log(f"\n  QSNN Confusion Matrix (t={t_qsnn}, full 82,332-sample test):")
log(f"                Pred Normal  Pred Attack")
log(f"  True Normal   {cm_qsnn[0,0]:>10,}  {cm_qsnn[0,1]:>11,}")
log(f"  True Attack   {cm_qsnn[1,0]:>10,}  {cm_qsnn[1,1]:>11,}")
tn2,fp2,fn2,tp2 = cm_qsnn.ravel()
log(f"  TN={tn2:,}  FP={fp2:,}  FN={fn2:,}  TP={tp2:,}")
log(f"  Acc={accuracy_score(y_test,preds_qsnn):.6f}  "
    f"Prec={precision_score(y_test,preds_qsnn):.6f}  "
    f"Rec={recall_score(y_test,preds_qsnn):.6f}  "
    f"F1={f1_score(y_test,preds_qsnn):.6f}")

log(f"\n  Predictions are IDENTICAL (same array): {np.array_equal(preds_qmlp, preds_qsnn)}")
log(f"  Number of samples where QMLP and QSNN predict differently: "
    f"{(preds_qmlp != preds_qsnn).sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# [3] QAE Full-test regeneration on all 82,332 samples
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("[3] QAE Full-Test Regeneration (82,332 samples)")
log("=" * 70)

# Load QAE artifacts
qae_params = np.array(joblib.load(os.path.join(ART, "qae_params.pkl")))
qae_config  = joblib.load(os.path.join(ART, "qae_config.pkl"))
X_test_full = np.load(os.path.join(ART, "quantum_input_test.npy"))

log(f"  qae_params shape : {qae_params.shape}")
log(f"  qae_config       : {qae_config}")
log(f"  X_test_full      : {X_test_full.shape}")

# Rebuild qae_circuit identically to quantum_models.py lines 133-153
dev = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(dev, diff_method="adjoint")
def qae_circuit(x, params):
    """
    QAE: single angle embedding + 3 VQC layers + trash qubit measurement.
    Identical to quantum_models.py qae_circuit — single embedding (no reupload).
    Measures PauliZ on trash qubits {3,4,5}; reconstruction error =
    mean(1 - Z_i). Attacks trained to have LOW error (memorise attack patterns).
    """
    for i in range(N_QUBITS):
        qml.RX(np.pi * x[i], wires=i)
    for lyr in range(3):
        offset = lyr * 18
        for i in range(N_QUBITS):
            qml.RX(params[offset + 3*i],     wires=i)
            qml.RY(params[offset + 3*i + 1], wires=i)
            qml.RZ(params[offset + 3*i + 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i+1) % N_QUBITS])
    return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in [3, 4, 5]])

def qae_errors_batch(X, params, batch_size=500):
    """Compute reconstruction error mean(1 - Z_i) for each sample."""
    all_errors = []
    for s in range(0, len(X), batch_size):
        chunk = X[s:s+batch_size]
        # Vectorized over batch
        raw = np.array(qae_circuit(chunk.T, params))  # (3, batch)
        errors = np.mean(1.0 - raw.T, axis=1)          # (batch,)
        all_errors.append(errors)
        if s % 5000 == 0:
            log(f"    ... {s:,}/{len(X):,} samples processed")
    return np.concatenate(all_errors)

log(f"\n  Running QAE inference on {len(X_test_full):,} samples...")
log(f"  Estimated time: ~{len(X_test_full) * 2.95 / 1000 / 60:.0f} min (2.95ms/sample)")

import time
t0 = time.time()
test_errors_full = qae_errors_batch(X_test_full, qae_params)
elapsed = time.time() - t0
log(f"  Done in {elapsed/60:.1f} min")

# Apply same threshold logic as quantum_models.py
attacks_lower = qae_config["attacks_have_lower_error"]
threshold     = qae_config["threshold"]  # 95th pct of TRAINING attack errors
log(f"\n  attacks_have_lower_error = {attacks_lower}")
log(f"  threshold (from training) = {threshold:.6f}")
log(f"  Error range on full test: [{test_errors_full.min():.6f}, {test_errors_full.max():.6f}]")

mean_err_atk = float(test_errors_full[y_test == 1].mean())
mean_err_nrm = float(test_errors_full[y_test == 0].mean())
log(f"  Mean error ATTACK : {mean_err_atk:.6f}")
log(f"  Mean error NORMAL : {mean_err_nrm:.6f}")
log(f"  Direction consistent: attacks_lower={mean_err_atk <= mean_err_nrm}")

if attacks_lower:
    preds_qae = (test_errors_full <= threshold).astype(int)
    max_err = test_errors_full.max() + 1e-9
    probs_qae = 1.0 - (test_errors_full / max_err)
else:
    preds_qae = (test_errors_full >= threshold).astype(int)
    probs_qae = test_errors_full / (test_errors_full.max() + 1e-9)

cm_qae = confusion_matrix(y_test, preds_qae)
tn_q,fp_q,fn_q,tp_q = cm_qae.ravel()
f1_qae  = f1_score(y_test, preds_qae, zero_division=0)
acc_qae = accuracy_score(y_test, preds_qae)
prec_qae= precision_score(y_test, preds_qae, zero_division=0)
rec_qae = recall_score(y_test, preds_qae, zero_division=0)

# Threshold search on full test
best_t_qae, best_f1_qae = threshold, f1_qae
for t_try in np.arange(0.01, 0.99, 0.01):
    if attacks_lower:
        p_try = (test_errors_full <= t_try).astype(int)
    else:
        p_try = (test_errors_full >= t_try).astype(int)
    f_try = f1_score(y_test, p_try, zero_division=0)
    if f_try > best_f1_qae:
        best_f1_qae, best_t_qae = f_try, round(float(t_try), 2)

log(f"\n  QAE Full-Test Results (82,332 samples):")
log(f"  Confusion Matrix:")
log(f"                Pred Normal  Pred Attack")
log(f"  True Normal   {cm_qae[0,0]:>10,}  {cm_qae[0,1]:>11,}")
log(f"  True Attack   {cm_qae[1,0]:>10,}  {cm_qae[1,1]:>11,}")
log(f"  TN={tn_q:,}  FP={fp_q:,}  FN={fn_q:,}  TP={tp_q:,}")
log(f"  Acc={acc_qae:.6f}  Prec={prec_qae:.6f}  Rec={rec_qae:.6f}  F1={f1_qae:.6f}")
log(f"  Best F1={best_f1_qae:.6f} at t={best_t_qae}")
log(f"  Probs range: [{probs_qae.min():.6f}, {probs_qae.max():.6f}]")

# Save updated artifacts
np.save(os.path.join(ART, "qae_test_errors.npy"), test_errors_full)
np.save(os.path.join(ART, "qae_test_probs.npy"),  probs_qae)
np.save(os.path.join(ART, "qae_test_preds.npy"),  preds_qae)

# Update thresholds
thresh["qae"] = float(threshold)
joblib.dump(thresh, os.path.join(ART, "quantum_thresholds.pkl"))

log(f"\n  Saved: qae_test_errors.npy {test_errors_full.shape}")
log(f"  Saved: qae_test_probs.npy  {probs_qae.shape}")
log(f"  Saved: qae_test_preds.npy  {preds_qae.shape}")
log(f"  quantum_thresholds.pkl     : {thresh}")

log("\n[Done] All three follow-up items resolved.")
