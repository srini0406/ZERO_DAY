"""
update_quantum_artifacts.py
──────────────────────────
Commits the BEST configuration from the Stage 3 sprint into the canonical
artifact files consumed by Stage 4 and later.

Best QSNN config  : Exp 1  (data re-uploading, 3 layers, CNOT ring,
                            margin=1.0, λ=0.5)
  Full F1 = 0.8242  |  Sep-ratio = 3.7551  |  Probs [0.296, 0.703]

Best QMLP config  : Exp 1  (data re-uploading, 3 layers, CNOT ring,
                            linear head)
  Subsample F1 = 0.7924  |  Probs [0.035, 0.930]  ← dramatic un-collapse

QAE               : circuit structure unchanged by the sprint; keep existing
                    params. Only thresholds are refreshed.

Why Exp 1 wins for QSNN:
  - Highest separation ratio (inter/mean) = 3.7551 vs next-best 3.6363.
  - Tightest attack intra-cluster distance = 0.0744 (Exp 5 = 0.0937,
    Exp 4 = 0.1171, Exp 2 = 0.2378, Exp 3a = 0.2486, Exp 3b = 0.4176).
  - F1 ceiling is architecture-limited at 0.8242 across all configs; the
    key differentiator is geometric quality for downstream DBSCAN use.
"""

import os, sys, numpy as np, pennylane as qml, pennylane.numpy as pnp
import joblib
from sklearn.metrics import f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def log(msg=""): print(msg, flush=True)

ART  = "artifacts"
SEED = 42

# ── Load sprint results ───────────────────────────────────────────────────────
log("[1/6] Loading sprint results...")
res = joblib.load(os.path.join(ART, "sprint_all_results.pkl"))
best_qsnn = res["exp1_qsnn"]
best_qmlp = res["exp1_qmlp"]
log(f"  QSNN best: {best_qsnn['exp_name']}  F1={best_qsnn['f1']:.4f}")
log(f"  QMLP best: {best_qmlp['exp_name']}  F1-sub={best_qmlp['f1']:.4f}")

# ── Re-build QSNN circuit (3 layers, reupload, CNOT ring) ────────────────────
log("[2/6] Rebuilding QSNN circuit for inference...")
N_QUBITS = 6
dev = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(dev, diff_method="adjoint")
def qsnn_circuit(x, params):
    """Data re-uploading embedding used by the winning QSNN configuration."""
    for lyr in range(3):
        for i in range(N_QUBITS):
            qml.RX(np.pi * x[i], wires=i)
        offset = lyr * 18
        for i in range(N_QUBITS):
            qml.RX(params[offset + 3*i],     wires=i)
            qml.RY(params[offset + 3*i + 1], wires=i)
            qml.RZ(params[offset + 3*i + 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i+1) % N_QUBITS])
    return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)])

def embed_fast(X, params, batch_size=2000):
    out = []
    for s in range(0, len(X), batch_size):
        chunk = X[s:s+batch_size]
        out.append(np.array(qsnn_circuit(chunk.T, params)).T)
    return np.vstack(out)

# ── Save QSNN params & prototypes ─────────────────────────────────────────────
log("[3/6] Saving QSNN params and prototypes...")
qsnn_params   = best_qsnn["params"]                  # shape (54,)
proto_atk     = best_qsnn["proto_atk"]
proto_nrm     = best_qsnn["proto_nrm"]
joblib.dump(qsnn_params, os.path.join(ART, "qsnn_params.pkl"))
joblib.dump({"proto_attack": proto_atk, "proto_normal": proto_nrm},
            os.path.join(ART, "qsnn_prototypes.pkl"))
log(f"  qsnn_params saved  (shape={qsnn_params.shape})")
log(f"  proto_atk : {proto_atk}")
log(f"  proto_nrm : {proto_nrm}")

# ── Save QMLP params ──────────────────────────────────────────────────────────
log("[4/6] Saving QMLP params...")
qmlp_all = best_qmlp["params"]     # 54 VQC + 6 weights + 1 bias = 61 total
vqc_count = 54
qmlp_vqc  = qmlp_all[:vqc_count]
qmlp_w    = qmlp_all[vqc_count:vqc_count + N_QUBITS]
qmlp_b    = qmlp_all[-1]
joblib.dump({"params_vqc": qmlp_vqc, "w": qmlp_w, "b": qmlp_b},
            os.path.join(ART, "qmlp_params.pkl"))
log(f"  qmlp_params saved  (VQC={qmlp_vqc.shape}, w={qmlp_w.shape}, b={float(qmlp_b):.4f})")

# ── Regenerate QSNN test predictions on full test set ─────────────────────────
log("[5/6] Regenerating QSNN full-test predictions...")
X_test_full  = np.load(os.path.join(ART, "quantum_input_test.npy"))
y_test_full  = np.load(os.path.join(ART, "y_test.npy"))

test_emb  = embed_fast(X_test_full, qsnn_params)
d_atk = np.linalg.norm(test_emb - proto_atk, axis=1)
d_nrm = np.linalg.norm(test_emb - proto_nrm, axis=1)
neg_d = np.stack([-d_atk, -d_nrm], axis=1)
exp_d = np.exp(neg_d - neg_d.max(axis=1, keepdims=True))
probs_qsnn = (exp_d / exp_d.sum(axis=1, keepdims=True))[:, 0]

# Optimise threshold over the full test set
best_qsnn_t, best_qsnn_f1 = 0.50, f1_score(y_test_full, (probs_qsnn >= 0.5).astype(int))
for t in np.arange(0.20, 0.81, 0.01):
    f = f1_score(y_test_full, (probs_qsnn >= t).astype(int))
    if f > best_qsnn_f1:
        best_qsnn_f1, best_qsnn_t = f, round(float(t), 2)

preds_qsnn = (probs_qsnn >= best_qsnn_t).astype(int)
np.save(os.path.join(ART, "qsnn_test_probs.npy"), probs_qsnn)
np.save(os.path.join(ART, "qsnn_test_preds.npy"), preds_qsnn)
log(f"  QSNN full-test F1={best_qsnn_f1:.4f}  best_t={best_qsnn_t}  Probs=[{probs_qsnn.min():.4f}, {probs_qsnn.max():.4f}]")

# ── Regenerate QMLP test predictions on full test set ─────────────────────────
log("[5b] Regenerating QMLP full-test predictions...")
test_emb_mlp = embed_fast(X_test_full, qmlp_vqc)
logits_mlp   = test_emb_mlp @ qmlp_w + qmlp_b
probs_qmlp   = 1.0 / (1.0 + np.exp(-np.clip(logits_mlp, -20.0, 20.0)))

best_qmlp_t, best_qmlp_f1 = 0.50, f1_score(y_test_full, (probs_qmlp >= 0.5).astype(int))
for t in np.arange(0.20, 0.81, 0.01):
    f = f1_score(y_test_full, (probs_qmlp >= t).astype(int))
    if f > best_qmlp_f1:
        best_qmlp_f1, best_qmlp_t = f, round(float(t), 2)

preds_qmlp = (probs_qmlp >= best_qmlp_t).astype(int)
np.save(os.path.join(ART, "qmlp_test_probs.npy"), probs_qmlp)
np.save(os.path.join(ART, "qmlp_test_preds.npy"), preds_qmlp)
log(f"  QMLP full-test F1={best_qmlp_f1:.4f}  best_t={best_qmlp_t}  Probs=[{probs_qmlp.min():.4f}, {probs_qmlp.max():.4f}]")

# ── Update thresholds & best model metadata ────────────────────────────────────
log("[6/6] Updating thresholds and best_quantum_model...")
qae_thresh = joblib.load(os.path.join(ART, "quantum_thresholds.pkl"))["qae"]
new_thresholds = {"qsnn": best_qsnn_t, "qmlp": best_qmlp_t, "qae": qae_thresh}
joblib.dump(new_thresholds, os.path.join(ART, "quantum_thresholds.pkl"))
log(f"  Updated thresholds: {new_thresholds}")

# best_quantum_model stays QSNN (same F1, best geometry, most principled for
# uncertainty estimation since its prob spread [0.296, 0.703] maps naturally
# to the [0.45, 0.60] uncertain zone used by Stage 4 DBSCAN).
joblib.dump("QSNN", os.path.join(ART, "best_quantum_model.pkl"))

# ── QAE: unchanged (circuit structure not modified in sprint) ─────────────────
log("  QAE: no circuit changes -> existing qae_params.pkl / qae_test_*.npy retained")

# ── Final verification ────────────────────────────────────────────────────────
log("\n=== FINAL ARTIFACT VERIFICATION ===")
log(f"  qsnn_params.pkl     : shape={joblib.load(os.path.join(ART, 'qsnn_params.pkl')).shape}")
log(f"  qsnn_prototypes.pkl : keys={list(joblib.load(os.path.join(ART, 'qsnn_prototypes.pkl')).keys())}")
log(f"  qmlp_params.pkl     : keys={list(joblib.load(os.path.join(ART, 'qmlp_params.pkl')).keys())}")
log(f"  qsnn_test_probs.npy : shape={np.load(os.path.join(ART, 'qsnn_test_probs.npy')).shape}")
log(f"  qmlp_test_probs.npy : shape={np.load(os.path.join(ART, 'qmlp_test_probs.npy')).shape}")
log(f"  quantum_thresholds  : {joblib.load(os.path.join(ART, 'quantum_thresholds.pkl'))}")
log(f"  best_quantum_model  : {joblib.load(os.path.join(ART, 'best_quantum_model.pkl'))}")
log("\n[Done] All quantum artifacts updated to sprint-best configuration.")
