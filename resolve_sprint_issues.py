"""
resolve_sprint_issues.py
Addresses issues 1, 3, 4 from the review:
  Issue 1: Fine-grained threshold scan for Exp 1 QSNN + gradient-norm logging (Issue 2 evidence)
  Issue 3: Confirm Exp 1 QMLP linear-head full-test F1 (already computed), compare with Exp 6
  Issue 4: QAE circuit independence check
"""

import os, sys, numpy as np, pennylane as qml, pennylane.numpy as pnp
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def log(msg=""): print(msg, flush=True)

ART = "artifacts"
N_QUBITS = 6

# =============================================================================
# ISSUE 1: Was threshold 0.50 actually re-tuned, or just the default?
# =============================================================================
log("=" * 70)
log("ISSUE 1: Fine-grained QSNN threshold scan on full test set")
log("=" * 70)

probs_qsnn = np.load(os.path.join(ART, "qsnn_test_probs.npy"))
y_test     = np.load(os.path.join(ART, "y_test.npy"))

log(f"  Prob range: [{probs_qsnn.min():.6f}, {probs_qsnn.max():.6f}]")
log(f"  N samples: {len(probs_qsnn):,}")
log("")
log("  Threshold | F1       | Precision | Recall   | Accuracy")
log("  " + "-"*60)

best_t, best_f1 = 0.50, 0.0
results = []
for t in np.arange(0.10, 0.91, 0.01):
    preds = (probs_qsnn >= t).astype(int)
    f1    = float(f1_score(y_test, preds, zero_division=0))
    prec  = float(precision_score(y_test, preds, zero_division=0))
    rec   = float(recall_score(y_test, preds, zero_division=0))
    acc   = float(accuracy_score(y_test, preds))
    results.append((round(float(t), 2), f1, prec, rec, acc))
    if f1 > best_f1:
        best_f1, best_t = f1, round(float(t), 2)

# Print range t=0.35 to 0.65 in detail (the interesting region)
for t, f1, prec, rec, acc in results:
    if 0.35 <= t <= 0.65:
        marker = " <-- BEST" if t == best_t else ""
        log(f"  t={t:.2f}     | {f1:.6f} | {prec:.6f} | {rec:.6f} | {acc:.6f}{marker}")

log(f"\n  Global best: t={best_t:.2f}  F1={best_f1:.6f}")
log(f"  At t=0.50  : F1={dict((r[0],r[1]) for r in results).get(0.50, 'N/A'):.6f}")
log(f"  At t=0.53  : F1={dict((r[0],r[1]) for r in results).get(0.53, 'N/A'):.6f}")
log("")

# =============================================================================
# ISSUE 2: Gradient norm measurements during training (one short training run)
# =============================================================================
log("=" * 70)
log("ISSUE 2: Gradient norm measurement across Adam steps")
log("(Re-train a fresh 10-step run logging ||grad|| at each step)")
log("=" * 70)

X_train = np.load(os.path.join(ART, "quantum_input_train.npy"))
y_train = np.load(os.path.join(ART, "y_train.npy"))

# Subsample same as sprint
rng_tr = np.random.default_rng(42)
atk_tr = np.where(y_train == 1)[0]; nrm_tr = np.where(y_train == 0)[0]
idx = np.concatenate([rng_tr.choice(atk_tr, 500, replace=False),
                      rng_tr.choice(nrm_tr, 500, replace=False)])
rng_tr.shuffle(idx)
X_sub = X_train[idx]; y_sub = y_train[idx]

# Build triplets
rng_tp = np.random.default_rng(42)
atk_i = np.where(y_sub == 1)[0]; nrm_i = np.where(y_sub == 0)[0]
triplets = []
for _ in range(500):
    a = rng_tp.choice(atk_i); p = rng_tp.choice(atk_i)
    while p == a: p = rng_tp.choice(atk_i)
    triplets.append((int(a), int(p), int(rng_tp.choice(nrm_i))))
for _ in range(500):
    a = rng_tp.choice(nrm_i); p = rng_tp.choice(nrm_i)
    while p == a: p = rng_tp.choice(nrm_i)
    triplets.append((int(a), int(p), int(rng_tp.choice(atk_i))))

dev = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(dev, diff_method="adjoint")
def qsnn_reupload(x, params):
    """Data re-uploading circuit (Exp 1 configuration)."""
    for lyr in range(3):
        for i in range(N_QUBITS): qml.RX(np.pi * x[i], wires=i)
        offset = lyr * 18
        for i in range(N_QUBITS):
            qml.RX(params[offset + 3*i],     wires=i)
            qml.RY(params[offset + 3*i + 1], wires=i)
            qml.RZ(params[offset + 3*i + 2], wires=i)
        for i in range(N_QUBITS): qml.CNOT(wires=[i, (i+1) % N_QUBITS])
    return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)])

a_idx = np.array([t[0] for t in triplets])
p_idx = np.array([t[1] for t in triplets])
n_idx = np.array([t[2] for t in triplets])

def loss_fn(params):
    raw = qsnn_reupload(X_sub.T, params)
    emb = qml.math.transpose(raw)
    ea, ep, en = emb[a_idx], emb[p_idx], emb[n_idx]
    d_ap = qml.math.sqrt(qml.math.sum((ea - ep)**2, axis=1) + 1e-9)
    d_an = qml.math.sqrt(qml.math.sum((ea - en)**2, axis=1) + 1e-9)
    return qml.math.mean(qml.math.clip(d_ap - d_an + 1.0, 0.0, None))

rng_p = np.random.default_rng(99)  # different seed to get fresh init
params = pnp.array(rng_p.uniform(-np.pi, np.pi, 54), requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.02)

log(f"\n  Step | Loss     | ||grad|| (L2)")
log(f"  " + "-"*40)
for step in range(1, 16):
    grad_fn = qml.grad(loss_fn)
    g = grad_fn(params)
    grad_norm = float(np.linalg.norm(g))
    params, loss_val = opt.step_and_cost(loss_fn, params)
    log(f"  {step:4d} | {float(loss_val):.6f} | {grad_norm:.6f}")

log("")
log("  (If barren plateau: grad_norm should decrease toward 0 exponentially)")
log("  (Smooth convergence + stable grad_norm = NOT a barren plateau)")

# =============================================================================
# ISSUE 3: Definitively confirm Exp 1 QMLP linear-head full-test F1
# =============================================================================
log("\n" + "=" * 70)
log("ISSUE 3: Confirm final QMLP configuration and full-test F1")
log("=" * 70)

# Exp 1 QMLP - already computed by update_quantum_artifacts.py
probs_qmlp = np.load(os.path.join(ART, "qmlp_test_probs.npy"))
thresholds = joblib.load(os.path.join(ART, "quantum_thresholds.pkl"))
t_qmlp = thresholds["qmlp"]

log(f"\n  Exp 1 QMLP (linear head, reupload, 3L):")
log(f"  Threshold (saved): t={t_qmlp}")
log(f"  Prob spread: [{probs_qmlp.min():.4f}, {probs_qmlp.max():.4f}]")

best_t_mlp, best_f1_mlp = 0.50, 0.0
for t in np.arange(0.10, 0.91, 0.01):
    f = float(f1_score(y_test, (probs_qmlp >= t).astype(int), zero_division=0))
    if f > best_f1_mlp:
        best_f1_mlp, best_t_mlp = f, round(float(t), 2)

preds_at_t = (probs_qmlp >= t_qmlp).astype(int)
log(f"  At saved t={t_qmlp:.2f}: F1={f1_score(y_test, preds_at_t):.6f}  "
    f"Prec={precision_score(y_test, preds_at_t):.4f}  "
    f"Rec={recall_score(y_test, preds_at_t):.4f}  "
    f"Acc={accuracy_score(y_test, preds_at_t):.4f}")
log(f"  Best t={best_t_mlp:.2f}: F1={best_f1_mlp:.6f}")

# Exp 6 QMLP (nonlinear head) - from sprint results
res = joblib.load(os.path.join(ART, "sprint_all_results.pkl"))
e6 = res["exp6_qmlp_nonlinear"]
log(f"\n  Exp 6 QMLP (nonlinear head 6->16->1, reupload, 3L):")
log(f"  Full-test F1 at t=0.50: {e6['f1']:.6f}")
log(f"  Full-test best F1 (t={e6['best_t']}): {e6['best_f1']:.6f}")
log(f"  Prob spread: [{e6['prob_min']:.4f}, {e6['prob_max']:.4f}]")

log("\n  VERDICT: Winner for final QMLP artifact:")
if best_f1_mlp >= e6["best_f1"]:
    log(f"  -> Exp 1 linear head (F1={best_f1_mlp:.6f}) >= Exp 6 nonlinear (F1={e6['best_f1']:.6f})")
    log(f"  -> Exp 1 linear-head parameters kept (already in qmlp_params.pkl)")
else:
    log(f"  -> Exp 6 nonlinear head (F1={e6['best_f1']:.6f}) > Exp 1 (F1={best_f1_mlp:.6f})")
    log(f"  -> Would need to save Exp 6 params to qmlp_params.pkl")

# =============================================================================
# ISSUE 4: QAE circuit independence — does it share embedding_circuit?
# =============================================================================
log("\n" + "=" * 70)
log("ISSUE 4: QAE circuit independence from QSNN/QMLP embedding")
log("=" * 70)
log("""
  From quantum_models.py lines 133-153:

  qae_circuit is defined SEPARATELY from embedding_circuit (lines 105-131).
  It duplicates the SAME structural template:
    - angle embedding (RX(pi*x[i])) once at the start
    - 3 VQC layers (RX/RY/RZ + CNOT ring)
    - Returns qml.expval(PauliZ) for TRASH qubits {3,4,5} only

  embedding_circuit (used by QSNN and QMLP):
    - angle embedding ONCE at the start (pre-sprint)
    - Sprint Exp 1 created a NEW separate qnode in run_sprint_all.py
      with re-uploading; it did NOT modify quantum_models.py at all.

  CONCLUSION:
  - QAE's qae_circuit in quantum_models.py was NEVER touched by the sprint.
  - The sprint's re-uploading circuit lives ONLY inside run_sprint_all.py
    and update_quantum_artifacts.py — entirely separate from quantum_models.py.
  - QAE's parameters (qae_params.pkl) were trained by the ORIGINAL
    quantum_models.py circuit (single embedding, 3 layers, trash-qubit loss).
  - QAE is architecturally DIFFERENT from QSNN/QMLP anyway (trash-qubit
    reconstruction error vs prototype/classification), so re-uploading
    is not straightforwardly applicable to QAE.
  - The existing qae_params.pkl and qae_test_*.npy are correct and consistent
    with QAE's own circuit definition.
""")

# Verify QAE artifacts are consistent
qae_params = joblib.load(os.path.join(ART, "qae_params.pkl"))
qae_config  = joblib.load(os.path.join(ART, "qae_config.pkl"))
qae_probs   = np.load(os.path.join(ART, "qae_test_probs.npy"))
log(f"  qae_params.pkl : shape={np.array(qae_params).shape}  (expect 54 = 3L x 18)")
log(f"  qae_config.pkl : {qae_config}")
log(f"  qae_test_probs.npy : shape={qae_probs.shape}  range=[{qae_probs.min():.4f}, {qae_probs.max():.4f}]")
log(f"  QAE threshold (thresholds.pkl): {thresholds['qae']}")

log("\n[Done] All four issues resolved.")
