"""
Diagnostic 5: Larger subsamples for QMLP and QAE.
Run each configuration SEPARATELY and report independently.
Configs: 1000 (baseline), 3000, 5000 samples.
Also: QSNN with 3000 and 5000 triplets.

Each run uses the same seed and eval methodology for fair comparison.
"""
import sys, os, time
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
TOTAL_VQC_PARAMS = 3 * N_QUBITS * N_VQC_LAYERS   # 54
ADAM_LR       = 0.02
ADAM_STEPS    = 100
TEST_SIZE     = 1000

np.random.seed(RANDOM_SEED)

dev = qml.device("lightning.qubit", wires=N_QUBITS)
print(f"  Device: lightning.qubit (adjoint)")

@qml.qnode(dev, diff_method="adjoint")
def embedding_circuit(x, params):
    for i in range(N_QUBITS):
        qml.RX(np.pi * x[i], wires=i)
    for lyr in range(N_VQC_LAYERS):
        off = lyr * 3 * N_QUBITS
        for i in range(N_QUBITS):
            qml.RX(params[off + 3*i],     wires=i)
            qml.RY(params[off + 3*i + 1], wires=i)
            qml.RZ(params[off + 3*i + 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i+1) % N_QUBITS])
    return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)])


@qml.qnode(dev, diff_method="adjoint")
def qae_circuit(x, params):
    for i in range(N_QUBITS):
        qml.RX(np.pi * x[i], wires=i)
    for lyr in range(N_VQC_LAYERS):
        off = lyr * 3 * N_QUBITS
        for i in range(N_QUBITS):
            qml.RX(params[off + 3*i],     wires=i)
            qml.RY(params[off + 3*i + 1], wires=i)
            qml.RZ(params[off + 3*i + 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i+1) % N_QUBITS])
    return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in [3, 4, 5]])


def tune_threshold(probs, y_true):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.02):
        p = (probs >= t).astype(int)
        f = f1_score(y_true, p, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, round(float(t), 2)
    return best_t, best_f1


def report(y_true, probs, label):
    best_t, _ = tune_threshold(probs, y_true)
    preds = (probs >= best_t).astype(int)
    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    f1   = f1_score(y_true, preds, zero_division=0)
    # also at 0.5 for comparison
    p05  = (probs >= 0.5).astype(int)
    f05  = f1_score(y_true, p05, zero_division=0)
    print(f"  {label}")
    print(f"    t=0.50: F1={f05:.4f}  |  t={best_t:.2f} (tuned): Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    print(f"    Probs: [{probs.min():.3f}, {probs.max():.3f}]  mean={probs.mean():.3f}")
    return acc, prec, rec, f1, best_t


# ── Load data once ────────────────────────────────────────────────────────────
X_train = np.load(os.path.join(ARTIFACTS_DIR, "quantum_input_train.npy"))
X_test  = np.load(os.path.join(ARTIFACTS_DIR, "quantum_input_test.npy"))
y_train = np.load(os.path.join(ARTIFACTS_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))

atk_tr = np.where(y_train == 1)[0]
nrm_tr = np.where(y_train == 0)[0]
atk_te = np.where(y_test  == 1)[0]
nrm_te = np.where(y_test  == 0)[0]

# Fixed test subsample (same across all experiments)
rng_te = np.random.default_rng(RANDOM_SEED + 99)
te_idx = np.concatenate([
    rng_te.choice(atk_te, TEST_SIZE // 2, replace=False),
    rng_te.choice(nrm_te, TEST_SIZE // 2, replace=False),
])
X_te   = X_test[te_idx]
y_te   = y_test[te_idx]


# =============================================================================
# EXPERIMENT A: QMLP at different subsample sizes
# =============================================================================

def qmlp_loss(all_params, X, y):
    p_vqc = all_params[:TOTAL_VQC_PARAMS]
    w = all_params[TOTAL_VQC_PARAMS:TOTAL_VQC_PARAMS + N_QUBITS]
    b = all_params[-1]
    emb = qml.math.stack([embedding_circuit(X[i], p_vqc) for i in range(len(X))])
    logits = qml.math.sum(emb * w, axis=1) + b
    probs  = 1.0 / (1.0 + qml.math.exp(-qml.math.clip(logits, -20.0, 20.0)))
    probs  = qml.math.clip(probs, 1e-7, 1.0 - 1e-7)
    return -qml.math.mean(y * qml.math.log(probs) + (1.0 - y) * qml.math.log(1.0 - probs))


def run_qmlp(n_train):
    rng = np.random.default_rng(RANDOM_SEED)
    half = n_train // 2
    tr_idx = np.concatenate([
        rng.choice(atk_tr, min(half, len(atk_tr)), replace=False),
        rng.choice(nrm_tr, min(n_train - half, len(nrm_tr)), replace=False),
    ])
    rng.shuffle(tr_idx)
    X_tr = X_train[tr_idx]
    y_tr = y_train[tr_idx].astype(float)

    init = pnp.array(
        np.concatenate([
            np.random.uniform(-np.pi, np.pi, TOTAL_VQC_PARAMS),
            np.random.uniform(-0.1, 0.1, N_QUBITS),
            [0.0]
        ]), requires_grad=True
    )
    opt = qml.AdamOptimizer(stepsize=ADAM_LR)

    t0 = time.time()
    for step in range(1, ADAM_STEPS + 1):
        init, _ = opt.step_and_cost(lambda p: qmlp_loss(p, X_tr, y_tr), init)
    elapsed = time.time() - t0

    opt_a = np.array(init)
    opt_vqc = opt_a[:TOTAL_VQC_PARAMS]
    opt_w   = opt_a[TOTAL_VQC_PARAMS:TOTAL_VQC_PARAMS + N_QUBITS]
    opt_b   = float(opt_a[-1])
    emb = np.array([embedding_circuit(X_te[i], opt_vqc) for i in range(len(X_te))])
    logits = emb @ opt_w + opt_b
    probs  = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    return probs, elapsed


print("\n" + "="*60)
print("DIAGNOSTIC 5A: QMLP - varying subsample size")
print("="*60)
print(f"  Test set: {TEST_SIZE} samples (fixed, same across experiments)")
print(f"  Adam steps: {ADAM_STEPS}  lr={ADAM_LR}\n")

for n_tr in [1000, 3000, 5000]:
    print(f"\n  --- QMLP n_train={n_tr} ---")
    probs, elapsed = run_qmlp(n_tr)
    report(y_te, probs, f"n_train={n_tr}  time={elapsed/60:.1f}min")


# =============================================================================
# EXPERIMENT B: QAE at different subsample sizes
# =============================================================================

def qae_loss(params, X):
    trash = qml.math.stack([qae_circuit(X[i], params) for i in range(len(X))])
    return qml.math.mean(1.0 - trash)


def run_qae(n_train):
    rng = np.random.default_rng(RANDOM_SEED)
    tr_idx = rng.choice(atk_tr, min(n_train, len(atk_tr)), replace=False)
    X_tr_atk = X_train[tr_idx]

    params = pnp.array(
        np.random.uniform(-np.pi, np.pi, TOTAL_VQC_PARAMS), requires_grad=True
    )
    opt = qml.AdamOptimizer(stepsize=ADAM_LR)

    t0 = time.time()
    for step in range(1, ADAM_STEPS + 1):
        params, _ = opt.step_and_cost(lambda p: qae_loss(p, X_tr_atk), params)
    elapsed = time.time() - t0

    opt_p = np.array(params)
    # Compute test errors
    trash = np.array([qae_circuit(X_te[i], opt_p) for i in range(len(X_te))])
    errors = np.mean(1.0 - trash, axis=1)
    train_errs = np.array([
        float(np.mean(1.0 - qae_circuit(X_tr_atk[i], opt_p)))
        for i in range(len(X_tr_atk))
    ])
    mean_atk = float(errors[y_te == 1].mean())
    mean_nrm = float(errors[y_te == 0].mean())
    attacks_lower = mean_atk <= mean_nrm
    if attacks_lower:
        threshold = float(np.percentile(train_errs, 95))
        attack_probs = 1.0 - errors / (errors.max() + 1e-9)
    else:
        threshold = float(np.percentile(train_errs, 5))
        attack_probs = errors / (errors.max() + 1e-9)
    return attack_probs, elapsed, mean_atk, mean_nrm, attacks_lower


print("\n" + "="*60)
print("DIAGNOSTIC 5B: QAE - varying subsample size")
print("="*60)
print(f"  Test set: {TEST_SIZE} samples (fixed)")
print(f"  Adam steps: {ADAM_STEPS}  lr={ADAM_LR}\n")

for n_tr in [1000, 3000, 5000]:
    print(f"\n  --- QAE n_train={n_tr} (attacks only) ---")
    probs, elapsed, mean_atk, mean_nrm, dir_ok = run_qae(n_tr)
    print(f"    Error direction: atk={mean_atk:.4f}  nrm={mean_nrm:.4f}  confirmed={dir_ok}")
    report(y_te, probs, f"n_train={n_tr}  time={elapsed/60:.1f}min")


# =============================================================================
# EXPERIMENT C: QSNN triplet count scaling
# =============================================================================

def build_triplets(X, y, n_triplets, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    atk = np.where(y == 1)[0]
    nrm = np.where(y == 0)[0]
    half = n_triplets // 2
    trips = []
    for _ in range(half):
        a = rng.choice(atk); p = rng.choice(atk)
        while p == a and len(atk) > 1: p = rng.choice(atk)
        n = rng.choice(nrm)
        trips.append((int(a), int(p), int(n)))
    for _ in range(n_triplets - half):
        a = rng.choice(nrm); p = rng.choice(nrm)
        while p == a and len(nrm) > 1: p = rng.choice(nrm)
        n = rng.choice(atk)
        trips.append((int(a), int(p), int(n)))
    rng.shuffle(trips)
    return trips


def qsnn_loss(params, X, y, triplets, margin=1.0, lam=0.5):
    embs = qml.math.stack([embedding_circuit(X[i], params) for i in range(len(X))])
    trip_l = []
    for (ai, pi, ni) in triplets:
        ea, ep, en = embs[ai], embs[pi], embs[ni]
        d_ap = qml.math.sqrt(qml.math.sum((ea - ep)**2) + 1e-9)
        d_an = qml.math.sqrt(qml.math.sum((ea - en)**2) + 1e-9)
        trip_l.append(qml.math.clip(d_ap - d_an + margin, 0.0, None))
    tl = qml.math.mean(qml.math.stack(trip_l))
    atk_e = embs[y == 1]; nrm_e = embs[y == 0]
    def mid(e):
        c = qml.math.mean(e, axis=0)
        return qml.math.mean(qml.math.sqrt(qml.math.sum((e - c)**2, axis=1) + 1e-9))
    spread = (mid(atk_e) + mid(nrm_e)) / 2.0
    return tl + lam * spread


def run_qsnn_triplets(n_sub, n_triplets):
    rng = np.random.default_rng(RANDOM_SEED)
    half = n_sub // 2
    sub_idx = np.concatenate([
        rng.choice(atk_tr, min(half, len(atk_tr)), replace=False),
        rng.choice(nrm_tr, min(n_sub - half, len(nrm_tr)), replace=False),
    ])
    rng.shuffle(sub_idx)
    X_sub = X_train[sub_idx]
    y_sub = y_train[sub_idx]

    triplets = build_triplets(X_sub, y_sub, n_triplets)

    params = pnp.array(
        np.random.uniform(-np.pi, np.pi, TOTAL_VQC_PARAMS), requires_grad=True
    )
    opt = qml.AdamOptimizer(stepsize=ADAM_LR)

    t0 = time.time()
    loss_start = None
    for step in range(1, ADAM_STEPS + 1):
        params, loss_val = opt.step_and_cost(
            lambda p: qsnn_loss(p, X_sub, y_sub, triplets), params
        )
        if step == 1:
            loss_start = float(loss_val)
    elapsed = time.time() - t0
    loss_end = float(loss_val)

    opt_p = np.array(params)
    # Prototypes from subsample
    sub_emb = np.array([embedding_circuit(X_sub[i], opt_p) for i in range(len(X_sub))])
    proto_atk = sub_emb[y_sub == 1].mean(axis=0)
    proto_nrm = sub_emb[y_sub == 0].mean(axis=0)
    inter = float(np.linalg.norm(proto_atk - proto_nrm))
    intra_a = float(np.mean(np.linalg.norm(sub_emb[y_sub==1] - proto_atk, axis=1)))
    intra_n = float(np.mean(np.linalg.norm(sub_emb[y_sub==0] - proto_nrm, axis=1)))
    sep = inter / (max(intra_a, intra_n) + 1e-9)

    # Test inference on fixed subsample
    te_emb = np.array([embedding_circuit(X_te[i], opt_p) for i in range(len(X_te))])
    d_a = np.linalg.norm(te_emb - proto_atk, axis=1)
    d_n = np.linalg.norm(te_emb - proto_nrm, axis=1)
    neg = np.stack([-d_a, -d_n], axis=1)
    exp_d = np.exp(neg - neg.max(axis=1, keepdims=True))
    attack_probs = (exp_d / exp_d.sum(axis=1, keepdims=True))[:, 0]

    return attack_probs, elapsed, inter, intra_a, intra_n, sep, loss_start, loss_end


print("\n" + "="*60)
print("DIAGNOSTIC 5C: QSNN - varying triplet count")
print("="*60)
print(f"  Adam steps: {ADAM_STEPS}  lr={ADAM_LR}")
print(f"  Subsample = triplet_count (1:1 ratio)\n")

for n_trips in [1000, 3000, 5000]:
    print(f"\n  --- QSNN n_triplets={n_trips}  subsample={n_trips} ---")
    probs, elapsed, inter, intra_a, intra_n, sep, l0, l1 = run_qsnn_triplets(n_trips, n_trips)
    print(f"    Loss: {l0:.4f} -> {l1:.4f}  time={elapsed/60:.1f}min")
    print(f"    Separation: inter={inter:.4f}  intra_atk={intra_a:.4f}  intra_nrm={intra_n:.4f}  ratio={sep:.4f}")
    report(y_te, probs, f"n_triplets={n_trips}")
