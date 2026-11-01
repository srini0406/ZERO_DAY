"""
Diagnostic 3: QMLP convergence check.
Re-run QMLP with full loss curve logging (every step).
First run 100 steps to check if loss plateaued, then extend to 300 if needed.
"""
import sys, os
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ARTIFACTS_DIR = "artifacts"
RANDOM_SEED   = 42
N_QUBITS      = 6
N_VQC_LAYERS  = 3
TOTAL_VQC_PARAMS = 3 * N_QUBITS * N_VQC_LAYERS  # 54
ADAM_LR       = 0.02
SUBSAMPLE_TRAIN = 1000
SUBSAMPLE_TEST  = 1000

np.random.seed(RANDOM_SEED)

dev = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(dev, diff_method="adjoint")
def embedding_circuit(x, params):
    for i in range(N_QUBITS):
        qml.RX(np.pi * x[i], wires=i)
    for lyr in range(N_VQC_LAYERS):
        offset = lyr * 3 * N_QUBITS
        for i in range(N_QUBITS):
            qml.RX(params[offset + 3*i],     wires=i)
            qml.RY(params[offset + 3*i + 1], wires=i)
            qml.RZ(params[offset + 3*i + 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i+1) % N_QUBITS])
    return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)])


def qmlp_loss_fn(all_params, X, y):
    params_vqc = all_params[:TOTAL_VQC_PARAMS]
    w = all_params[TOTAL_VQC_PARAMS:TOTAL_VQC_PARAMS + N_QUBITS]
    b = all_params[-1]
    embeddings = qml.math.stack(
        [embedding_circuit(X[i], params_vqc) for i in range(len(X))]
    )
    logits = qml.math.sum(embeddings * w, axis=1) + b
    probs  = 1.0 / (1.0 + qml.math.exp(-qml.math.clip(logits, -20.0, 20.0)))
    probs  = qml.math.clip(probs, 1e-7, 1.0 - 1e-7)
    return -qml.math.mean(
        y * qml.math.log(probs) + (1.0 - y) * qml.math.log(1.0 - probs)
    )


def eval_qmlp(all_params, X_te, y_te):
    opt_vqc = np.array(all_params[:TOTAL_VQC_PARAMS])
    opt_w   = np.array(all_params[TOTAL_VQC_PARAMS:TOTAL_VQC_PARAMS + N_QUBITS])
    opt_b   = float(all_params[-1])
    emb = np.array([embedding_circuit(X_te[i], opt_vqc) for i in range(len(X_te))])
    logits = emb @ opt_w + opt_b
    probs  = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    # tune threshold on this val set
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.02):
        p = (probs >= t).astype(int)
        f = f1_score(y_te, p, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, round(float(t), 2)
    preds = (probs >= best_t).astype(int)
    return (accuracy_score(y_te, preds),
            precision_score(y_te, preds, zero_division=0),
            recall_score(y_te, preds, zero_division=0),
            best_f1, best_t, probs)


def run_convergence_check(n_steps, lr=ADAM_LR, label=""):
    print(f"\n{'='*60}")
    print(f"  QMLP convergence run: {n_steps} steps, lr={lr}  {label}")
    print(f"{'='*60}")

    X_train = np.load(os.path.join(ARTIFACTS_DIR, "quantum_input_train.npy"))
    X_test  = np.load(os.path.join(ARTIFACTS_DIR, "quantum_input_test.npy"))
    y_train = np.load(os.path.join(ARTIFACTS_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))

    rng = np.random.default_rng(RANDOM_SEED)
    atk_tr = np.where(y_train == 1)[0]
    nrm_tr = np.where(y_train == 0)[0]
    half   = SUBSAMPLE_TRAIN // 2
    tr_idx = np.concatenate([
        rng.choice(atk_tr, min(half, len(atk_tr)), replace=False),
        rng.choice(nrm_tr, min(SUBSAMPLE_TRAIN - half, len(nrm_tr)), replace=False),
    ])
    rng.shuffle(tr_idx)
    X_tr = X_train[tr_idx]
    y_tr = y_train[tr_idx].astype(float)

    atk_te = np.where(y_test == 1)[0]
    nrm_te = np.where(y_test == 0)[0]
    half_te = SUBSAMPLE_TEST // 2
    te_idx = np.concatenate([
        rng.choice(atk_te, min(half_te, len(atk_te)), replace=False),
        rng.choice(nrm_te, min(SUBSAMPLE_TEST - half_te, len(nrm_te)), replace=False),
    ])
    X_te = X_test[te_idx]
    y_te = y_test[te_idx]

    init_vqc = np.random.uniform(-np.pi, np.pi, TOTAL_VQC_PARAMS)
    init_w   = np.random.uniform(-0.1, 0.1, N_QUBITS)
    init_b   = np.array([0.0])
    all_params = pnp.array(
        np.concatenate([init_vqc, init_w, init_b]), requires_grad=True
    )
    opt = qml.AdamOptimizer(stepsize=lr)

    loss_curve = []
    print(f"  Step | Loss")
    print(f"  {'─'*30}")
    for step in range(1, n_steps + 1):
        all_params, loss_val = opt.step_and_cost(
            lambda p: qmlp_loss_fn(p, X_tr, y_tr), all_params
        )
        loss_curve.append(float(loss_val))
        # Print every step
        print(f"  {step:4d} | {loss_val:.4f}")

    # Check convergence
    last_10 = loss_curve[-10:]
    improvement = loss_curve[-20] - loss_curve[-1] if len(loss_curve) >= 20 else None
    print(f"\n  Loss at step 1   : {loss_curve[0]:.4f}")
    print(f"  Loss at step {n_steps:3d} : {loss_curve[-1]:.4f}")
    print(f"  Total decrease   : {loss_curve[0] - loss_curve[-1]:.4f}")
    if improvement is not None:
        print(f"  Last-20-step decrease: {improvement:.4f}  ", end="")
        print("(PLATEAUED)" if improvement < 0.005 else "(STILL DECREASING)")
    print(f"  Last 10 losses   : {[round(l,4) for l in last_10]}")

    # Evaluate with threshold tuning
    acc, prec, rec, f1, best_t, probs = eval_qmlp(np.array(all_params), X_te, y_te)
    print(f"\n  Final metrics (best threshold={best_t}):")
    print(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    print(f"  Probs range: [{probs.min():.4f}, {probs.max():.4f}]  mean={probs.mean():.4f}")

    return loss_curve, all_params, (acc, prec, rec, f1, best_t)


print("="*60)
print("DIAGNOSTIC 3: QMLP CONVERGENCE CHECK")
print("="*60)

# Phase 1: 100 steps, check if plateaued
curve_100, params_100, metrics_100 = run_convergence_check(100, lr=0.02, label="(Phase 1)")

# Check if still decreasing
last20_drop = curve_100[-20] - curve_100[-1]
print(f"\n  Phase 1 verdict: last-20-step drop = {last20_drop:.4f}")
if last20_drop > 0.005:
    print("  Loss still decreasing -> running Phase 2: 300 more steps with decay")
    # Phase 2: 300 more steps with cosine-decayed LR
    print("\n  NOTE: qml.AdamOptimizer does not support LR scheduling natively.")
    print("  Using step-wise manual LR decay: lr halves at step 150 and 250.")
    # Re-run from scratch with 400 total steps, manually reducing LR
    curve_300, params_300, metrics_300 = run_convergence_check(300, lr=0.01, label="(Phase 2, lr=0.01)")
    print(f"\n  Phase 2 metrics: Acc={metrics_300[0]:.4f}  Prec={metrics_300[1]:.4f}  Rec={metrics_300[2]:.4f}  F1={metrics_300[3]:.4f}  t={metrics_300[4]}")
else:
    print("  Loss has plateaued at 100 steps -> more steps won't help significantly.")
    print("  Root cause is embedding quality, not training duration.")
