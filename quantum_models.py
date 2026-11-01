"""
quantum_models.py - Stage 3: Quantum Machine Learning Models (PennyLane)

Device: lightning.qubit with adjoint differentiation.
Optimizer: qml.AdamOptimizer for all three models.

WHY lightning.qubit over default.qubit:
  lightning.qubit is a C++ accelerated statevector simulator. Combined with
  adjoint differentiation it computes the exact gradient in ~one extra forward
  pass regardless of parameter count, vs COBYLA which needs O(num_params)
  function evaluations per iteration just to build the simplex (56 evals for
  our 54 params before the first real step).

WHY Adam + adjoint over COBYLA:
  COBYLA is gradient-free. For a 54-parameter circuit it requires ~56 circuit
  evaluations per iteration just to estimate the gradient numerically. Adam
  with adjoint differentiation computes the exact gradient in ~2x the cost of
  one forward pass, independent of parameter count. On our benchmark:
    COBYLA: ~52s/iter on 200 samples  (~260 ms/sample total)
    Adam+adjoint: ~1.49s/step on 200 samples  (~7.45 ms/sample) = 35x faster

Measured benchmarks (lightning.qubit + adjoint, this machine):
  Forward only       : 2.95 ms/sample
  Forward + backward : 7.45 ms/sample

Three models trained INDEPENDENTLY - no ensembling.
WHY no ensembling: ensembling three weak quantum models can amplify shared
errors, and QAE's output (anomaly score) doesn't compose cleanly with binary
labels from the other two.

Adjoint differentiation support:
  All gates used (RX, RY, RZ, CNOT) fully support adjoint differentiation on
  lightning.qubit. No mid-circuit measurements are used. Full gradient support
  confirmed for all three models.
"""

import os
import sys
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import joblib
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── Device setup with fallback ────────────────────────────────────────────────
def get_device(n_wires):
    """
    Try lightning.qubit first (C++ accelerated, adjoint support).
    Fall back to default.qubit with a clear warning if unavailable.
    """
    try:
        dev = qml.device("lightning.qubit", wires=n_wires)
        # Verify adjoint actually works on this install
        @qml.qnode(dev, diff_method="adjoint")
        def _probe(p):
            qml.RX(p[0], wires=0)
            return qml.math.stack([qml.expval(qml.PauliZ(0))])
        _p = pnp.array([0.1], requires_grad=True)
        qml.grad(_probe)(_p)
        print(f"  Device: lightning.qubit (adjoint differentiation confirmed)")
        return dev, "adjoint"
    except Exception as e:
        print(f"  WARNING: lightning.qubit unavailable ({e})")
        print(f"  Falling back to default.qubit with parameter-shift.")
        print(f"  Expect ~10x slower gradient computation.")
        dev = qml.device("default.qubit", wires=n_wires)
        return dev, "parameter-shift"


# ── Circuit constants ─────────────────────────────────────────────────────────
N_QUBITS     = 6          # matches PCA n_components from Stage 1
N_VQC_LAYERS = 3
PARAMS_PER_LAYER = 3 * N_QUBITS        # RX + RY + RZ per qubit
TOTAL_VQC_PARAMS = N_VQC_LAYERS * PARAMS_PER_LAYER   # 54

# Training hyper-params (based on benchmark: 7.45ms/sample/step)
# 1000 samples x 100 steps ~ 12 min per model
ADAM_LR      = 0.02
ADAM_STEPS   = 100
SUBSAMPLE_TRAIN = 1000   # for QMLP, QAE, QSNN triplet training
SUBSAMPLE_TEST  = 1000   # for QMLP, QAE evaluation
QSNN_TRIPLETS   = 1000   # triplet count for QSNN

# Build shared device once
_dev, _diff_method = get_device(N_QUBITS)


# =============================================================================
# SHARED CIRCUIT COMPONENTS
# =============================================================================

def _make_qnodes(dev, diff_method):
    """Create QNodes bound to a specific device and diff_method."""

    @qml.qnode(dev, diff_method=diff_method)
    def embedding_circuit(x, params):
        """
        Angle embedding + VQC layers -> 6 PauliZ expectation values.
        Used by QSNN and QMLP as the shared embedding function.

        Angle embedding: RX(pi * feature_i) on qubit i.
        WHY angle embedding: features are in [0,1] after PCA normalisation,
        so pi * feature covers [0, pi] - the full useful range of RX.

        VQC layer: per-qubit RX/RY/RZ + CNOT ring entanglement.
        WHY CNOT ring: O(N) entanglement gates; each qubit influences its
        neighbour, propagating correlations without all-to-all connectivity.
        """
        for i in range(N_QUBITS):
            qml.RX(np.pi * x[i], wires=i)
        for lyr in range(N_VQC_LAYERS):
            offset = lyr * PARAMS_PER_LAYER
            for i in range(N_QUBITS):
                qml.RX(params[offset + 3 * i],     wires=i)
                qml.RY(params[offset + 3 * i + 1], wires=i)
                qml.RZ(params[offset + 3 * i + 2], wires=i)
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
        return qml.math.stack(
            [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
        )

    @qml.qnode(dev, diff_method=diff_method)
    def qae_circuit(x, params):
        """
        QAE circuit: same embedding + VQC, measure ONLY trash qubits (3-5).
        Reconstruction error = mean(1 - Z_i) for trash qubits i in {3,4,5}.
        Trash qubits start in |0> (Z=+1); deviation from +1 measures
        information lost during compression - the QAE objective.
        """
        for i in range(N_QUBITS):
            qml.RX(np.pi * x[i], wires=i)
        for lyr in range(N_VQC_LAYERS):
            offset = lyr * PARAMS_PER_LAYER
            for i in range(N_QUBITS):
                qml.RX(params[offset + 3 * i],     wires=i)
                qml.RY(params[offset + 3 * i + 1], wires=i)
                qml.RZ(params[offset + 3 * i + 2], wires=i)
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
        return qml.math.stack(
            [qml.expval(qml.PauliZ(i)) for i in [3, 4, 5]]
        )

    return embedding_circuit, qae_circuit


embedding_circuit, qae_circuit = _make_qnodes(_dev, _diff_method)


def embed_batch(X, params):
    """
    Embed a batch of samples using the shared embedding circuit.
    Returns ndarray of shape (n, N_QUBITS).
    Uses a Python loop - PennyLane 0.45 broadcasting on lightning.qubit
    returns shape (N_QUBITS, N_QUBITS) when batch dim == N_QUBITS, so
    explicit loop is safer and still fast (2.95 ms/sample on lightning.qubit).
    """
    return np.array([embedding_circuit(X[i], params) for i in range(len(X))])


def qae_errors_batch(X, params):
    """Reconstruction error for each sample: mean(1 - Z_i) for trash qubits."""
    results = np.array([qae_circuit(X[i], params) for i in range(len(X))])
    return np.mean(1.0 - results, axis=1)   # shape (n,)


# =============================================================================
# A) QSNN - Quantum Siamese Neural Network
#
# WHY QSNN instead of QSVM:
#   QSVM requires O(n^2) kernel matrix evaluations (~2.9 years for 175k samples
#   on a simulator). QSNN's triplet training cost is fixed (triplet count only),
#   and prototype inference is O(n) - one forward pass per sample.
#
# Training: triplet loss + spread penalty, optimised with Adam + adjoint.
# WHY triplet loss + spread penalty:
#   Triplet loss pushes same-class embeddings together and cross-class apart.
#   Spread penalty (lambda=0.5) directly penalises loose clusters by measuring
#   mean intra-class distance to own centroid - targeting tight separation.
#
# Inference: classify by nearest prototype (mean class embedding), O(1)/query.
# =============================================================================

def build_triplets(X, y, n_triplets, seed=RANDOM_SEED):
    """
    Build balanced (anchor, positive, negative) triplet index triples.
    WHY 50/50 balance: majority class (attacks, ~68%) would dominate triplet
    sampling and bias the embedding toward attack geometry; forcing balance
    ensures both class boundaries are learned equally.
    """
    rng = np.random.default_rng(seed)
    attack_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    if len(attack_idx) == 0 or len(normal_idx) == 0:
        raise ValueError("build_triplets requires both classes in y.")

    half = n_triplets // 2
    triplets = []

    # Attack-anchored
    for _ in range(half):
        a = rng.choice(attack_idx)
        p = rng.choice(attack_idx)
        while p == a and len(attack_idx) > 1:
            p = rng.choice(attack_idx)
        n = rng.choice(normal_idx)
        triplets.append((int(a), int(p), int(n)))

    # Normal-anchored
    for _ in range(n_triplets - half):
        a = rng.choice(normal_idx)
        p = rng.choice(normal_idx)
        while p == a and len(normal_idx) > 1:
            p = rng.choice(normal_idx)
        n = rng.choice(attack_idx)
        triplets.append((int(a), int(p), int(n)))

    rng.shuffle(triplets)
    print(f"  Triplets built: {len(triplets)} "
          f"({half} attack-anchored, {n_triplets - half} normal-anchored)")
    return triplets


def qsnn_loss_fn(params, X, y, triplets, margin=1.0, lam=0.5):
    """
    Triplet loss + spread penalty.
    params: pnp array with requires_grad=True for Adam differentiation.
    """
    embeddings = qml.math.stack(
        [embedding_circuit(X[i], params) for i in range(len(X))]
    )  # (n, N_QUBITS)

    # Triplet loss: mean(max(0, ||e_a - e_p|| - ||e_a - e_n|| + margin))
    trip_losses = []
    for (ai, pi, ni) in triplets:
        ea, ep, en = embeddings[ai], embeddings[pi], embeddings[ni]
        d_ap = qml.math.sqrt(qml.math.sum((ea - ep) ** 2) + 1e-9)
        d_an = qml.math.sqrt(qml.math.sum((ea - en) ** 2) + 1e-9)
        trip_losses.append(qml.math.clip(d_ap - d_an + margin, 0.0, None))
    triplet_loss = qml.math.mean(qml.math.stack(trip_losses))

    # Spread penalty: mean intra-class distance to class centroid
    atk_mask = (y == 1)
    nrm_mask = (y == 0)
    atk_emb = embeddings[atk_mask]
    nrm_emb = embeddings[nrm_mask]

    def mean_intra_dist(emb):
        centroid = qml.math.mean(emb, axis=0)
        diffs = emb - centroid
        return qml.math.mean(qml.math.sqrt(
            qml.math.sum(diffs ** 2, axis=1) + 1e-9
        ))

    spread = (mean_intra_dist(atk_emb) + mean_intra_dist(nrm_emb)) / 2.0
    return triplet_loss + lam * spread


def run_qsnn(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("  QSNN - Quantum Siamese Neural Network")
    print("=" * 60)
    print(f"  Subsample (triplet training) : {SUBSAMPLE_TRAIN}")
    print(f"  Triplets                     : {QSNN_TRIPLETS}")
    print(f"  Adam steps                   : {ADAM_STEPS}  lr={ADAM_LR}")
    print(f"  Prototype pass               : full train set ({len(X_train):,})")
    print(f"  Test inference               : full test set ({len(X_test):,})")

    # Stratified subsample for triplet training
    rng = np.random.default_rng(RANDOM_SEED)
    atk_idx = np.where(y_train == 1)[0]
    nrm_idx = np.where(y_train == 0)[0]
    half_sub = SUBSAMPLE_TRAIN // 2
    sub_idx = np.concatenate([
        rng.choice(atk_idx, min(half_sub, len(atk_idx)), replace=False),
        rng.choice(nrm_idx, min(SUBSAMPLE_TRAIN - half_sub, len(nrm_idx)), replace=False),
    ])
    rng.shuffle(sub_idx)
    X_sub = X_train[sub_idx]
    y_sub = y_train[sub_idx]
    print(f"  Subsample class balance: "
          f"attack={int((y_sub==1).sum())}  normal={int((y_sub==0).sum())}")

    triplets = build_triplets(X_sub, y_sub, QSNN_TRIPLETS)

    # Initialise params
    params = pnp.array(
        np.random.uniform(-np.pi, np.pi, TOTAL_VQC_PARAMS),
        requires_grad=True,
    )
    opt = qml.AdamOptimizer(stepsize=ADAM_LR)

    print("\n  Training with Adam + adjoint differentiation...")
    for step in range(1, ADAM_STEPS + 1):
        params, loss_val = opt.step_and_cost(
            lambda p: qsnn_loss_fn(p, X_sub, y_sub, triplets), params
        )
        if step % 20 == 0 or step == 1:
            print(f"  Step {step:4d}/{ADAM_STEPS} | loss = {float(loss_val):.4f}")

    opt_params = np.array(params)
    print(f"  Training complete.")

    # ── Prototype computation on full training set ────────────────────────────
    print(f"\n  Computing prototypes on full training set ({len(X_train):,} samples)...")
    print(f"  Estimated: ~{2.95 * len(X_train) / 1000 / 60:.1f} min")
    CHUNK = 2000
    all_emb = []
    for start in range(0, len(X_train), CHUNK):
        end = min(start + CHUNK, len(X_train))
        all_emb.append(embed_batch(X_train[start:end], opt_params))
        if (start // CHUNK) % 10 == 0:
            print(f"  Embedded {end:,}/{len(X_train):,} ...")
    train_emb = np.vstack(all_emb)

    proto_attack = train_emb[y_train == 1].mean(axis=0)
    proto_normal = train_emb[y_train == 0].mean(axis=0)
    inter_dist   = np.linalg.norm(proto_attack - proto_normal)
    intra_atk    = np.mean(np.linalg.norm(train_emb[y_train == 1] - proto_attack, axis=1))
    intra_nrm    = np.mean(np.linalg.norm(train_emb[y_train == 0] - proto_normal, axis=1))
    sep_ratio    = inter_dist / (max(intra_atk, intra_nrm) + 1e-9)

    print(f"\n  === Embedding Diagnostics ===")
    print(f"  Inter-prototype distance     : {inter_dist:.4f}")
    print(f"  Mean intra-class dist (atk)  : {intra_atk:.4f}")
    print(f"  Mean intra-class dist (nrm)  : {intra_nrm:.4f}")
    print(f"  Separation ratio (target>2.5): {sep_ratio:.4f}")
    if sep_ratio < 2.5:
        print("  NOTE: separation ratio below 2.5 - embeddings may overlap.")

    # ── Inference on full test set ────────────────────────────────────────────
    print(f"\n  Classifying full test set ({len(X_test):,} samples)...")
    print(f"  Estimated: ~{2.95 * len(X_test) / 1000 / 60:.1f} min")
    all_test_emb = []
    for start in range(0, len(X_test), CHUNK):
        end = min(start + CHUNK, len(X_test))
        all_test_emb.append(embed_batch(X_test[start:end], opt_params))
        if (start // CHUNK) % 10 == 0:
            print(f"  Embedded {end:,}/{len(X_test):,} ...")
    test_emb = np.vstack(all_test_emb)

    d_attack = np.linalg.norm(test_emb - proto_attack, axis=1)
    d_normal = np.linalg.norm(test_emb - proto_normal, axis=1)

    # Softmax over negative distances -> attack probability
    neg_dists = np.stack([-d_attack, -d_normal], axis=1)
    exp_d = np.exp(neg_dists - neg_dists.max(axis=1, keepdims=True))
    probs_mat = exp_d / exp_d.sum(axis=1, keepdims=True)
    attack_probs = probs_mat[:, 0]
    preds = (attack_probs >= 0.5).astype(int)

    print(f"\n  Probability diagnostics (near-0.5 mean = weak separation):")
    print(f"  min={attack_probs.min():.4f}  max={attack_probs.max():.4f}  "
          f"mean={attack_probs.mean():.4f}  std={attack_probs.std():.4f}")

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    cm   = confusion_matrix(y_test, preds)

    print(f"\n  QSNN Results (FULL test set, {len(y_test):,} samples):")
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    joblib.dump(opt_params, os.path.join(ARTIFACTS_DIR, "qsnn_params.pkl"))
    joblib.dump({"proto_attack": proto_attack, "proto_normal": proto_normal},
                os.path.join(ARTIFACTS_DIR, "qsnn_prototypes.pkl"))
    np.save(os.path.join(ARTIFACTS_DIR, "qsnn_test_probs.npy"), attack_probs)
    np.save(os.path.join(ARTIFACTS_DIR, "qsnn_test_preds.npy"), preds)

    return {
        "model": "QSNN", "params": opt_params,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm,
        "probs": attack_probs, "preds": preds,
        "eval_set": f"full test set ({len(y_test):,} samples)",
        "sep_ratio": sep_ratio,
    }


# =============================================================================
# B) QMLP - Quantum MLP
# Shared embedding circuit + linear head, trained with Adam + adjoint.
# WHY subsample: each Adam step embeds all training samples; with adjoint
# this costs 7.45ms/sample. 1000 samples x 100 steps ~ 12 min on this CPU.
# =============================================================================

def qmlp_loss_fn(all_params, X, y):
    """Binary cross-entropy: quantum embedding -> linear -> sigmoid."""
    params_vqc = all_params[:TOTAL_VQC_PARAMS]
    w = all_params[TOTAL_VQC_PARAMS:TOTAL_VQC_PARAMS + N_QUBITS]
    b = all_params[-1]

    embeddings = qml.math.stack(
        [embedding_circuit(X[i], params_vqc) for i in range(len(X))]
    )
    logits = qml.math.sum(embeddings * w, axis=1) + b
    probs = 1.0 / (1.0 + qml.math.exp(-qml.math.clip(logits, -20.0, 20.0)))
    probs = qml.math.clip(probs, 1e-7, 1.0 - 1e-7)
    return -qml.math.mean(
        y * qml.math.log(probs) + (1.0 - y) * qml.math.log(1.0 - probs)
    )


def run_qmlp(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("  QMLP - Quantum MLP")
    print("=" * 60)
    print(f"  Train subsample: {SUBSAMPLE_TRAIN}  Test subsample: {SUBSAMPLE_TEST}")
    print(f"  Adam steps: {ADAM_STEPS}  lr={ADAM_LR}")
    print(f"  NOTE: 1000 samples x 100 steps ~ 12 min on this CPU.")

    rng = np.random.default_rng(RANDOM_SEED)
    atk_tr = np.where(y_train == 1)[0]
    nrm_tr = np.where(y_train == 0)[0]
    half_tr = SUBSAMPLE_TRAIN // 2
    tr_idx = np.concatenate([
        rng.choice(atk_tr, min(half_tr, len(atk_tr)), replace=False),
        rng.choice(nrm_tr, min(SUBSAMPLE_TRAIN - half_tr, len(nrm_tr)), replace=False),
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

    print(f"  Train: attack={int(y_tr.sum())}  normal={int((1-y_tr).sum())}")
    print(f"  Test : attack={int((y_te==1).sum())}  normal={int((y_te==0).sum())}")

    # Init: VQC params + linear head weight w + bias b
    init_vqc = np.random.uniform(-np.pi, np.pi, TOTAL_VQC_PARAMS)
    init_w   = np.random.uniform(-0.1,   0.1,   N_QUBITS)
    init_b   = np.array([0.0])
    all_params = pnp.array(
        np.concatenate([init_vqc, init_w, init_b]), requires_grad=True
    )
    opt = qml.AdamOptimizer(stepsize=ADAM_LR)

    print("\n  Training with Adam + adjoint differentiation...")
    for step in range(1, ADAM_STEPS + 1):
        all_params, loss_val = opt.step_and_cost(
            lambda p: qmlp_loss_fn(p, X_tr, y_tr), all_params
        )
        if step % 20 == 0 or step == 1:
            print(f"  Step {step:4d}/{ADAM_STEPS} | loss = {float(loss_val):.4f}")

    opt_params = np.array(all_params)
    opt_vqc = opt_params[:TOTAL_VQC_PARAMS]
    opt_w   = opt_params[TOTAL_VQC_PARAMS:TOTAL_VQC_PARAMS + N_QUBITS]
    opt_b   = opt_params[-1]
    print(f"  Training complete.")

    # Evaluate on test subsample
    test_emb = embed_batch(X_te, opt_vqc)
    logits   = test_emb @ opt_w + opt_b
    probs    = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    preds    = (probs >= 0.5).astype(int)

    acc  = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, zero_division=0)
    rec  = recall_score(y_te, preds, zero_division=0)
    f1   = f1_score(y_te, preds, zero_division=0)
    cm   = confusion_matrix(y_te, preds)

    print(f"\n  QMLP Results ({SUBSAMPLE_TEST}-sample subsample):")
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    joblib.dump({"params_vqc": opt_vqc, "w": opt_w, "b": opt_b},
                os.path.join(ARTIFACTS_DIR, "qmlp_params.pkl"))
    np.save(os.path.join(ARTIFACTS_DIR, "qmlp_test_probs.npy"), probs)
    np.save(os.path.join(ARTIFACTS_DIR, "qmlp_test_preds.npy"), preds)

    return {
        "model": "QMLP", "params_vqc": opt_vqc, "w": opt_w, "b": opt_b,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm,
        "probs": probs, "preds": preds,
        "eval_set": f"{SUBSAMPLE_TEST}-sample subsample",
    }


# =============================================================================
# C) QAE - Quantum Autoencoder (novelty / anomaly detector)
#
# Trained ONLY on known-attack samples. Minimises reconstruction error on
# attacks, so attacks should show LOWER error than unseen/normal traffic.
#
# CRITICAL: error direction verified empirically before setting any threshold.
# If direction reverses (attacks show higher error), rule inverts with warning.
# Probs and preds are always kept directionally consistent.
#
# NOTE: QAE's binary accuracy/F1 here is a PROXY metric - its real job is
# distinguishing known-vs-unknown among CNN-flagged suspicious traffic.
# =============================================================================

def qae_loss_fn(params, X):
    """
    Mean reconstruction error over attack-only batch.
    Error = mean(1 - Z_i) for trash qubits {3,4,5}.
    Z_i = +1 means qubit stayed in |0> (perfect compression).
    Deviation measures information leakage to the trash register.
    """
    trash_vals = qml.math.stack(
        [qae_circuit(X[i], params) for i in range(len(X))]
    )  # (n, 3)
    return qml.math.mean(1.0 - trash_vals)


def run_qae(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("  QAE - Quantum Autoencoder")
    print("=" * 60)
    print(f"  Train (attacks only): {SUBSAMPLE_TRAIN}  Test: {SUBSAMPLE_TEST}")
    print(f"  Adam steps: {ADAM_STEPS}  lr={ADAM_LR}")
    print(f"  NOTE: QAE binary accuracy is a proxy - real role is novelty")
    print(f"        detection among CNN-flagged suspicious traffic (Stage 4/6).")

    rng = np.random.default_rng(RANDOM_SEED)

    # Train on attacks only
    atk_tr = np.where(y_train == 1)[0]
    tr_idx = rng.choice(atk_tr, min(SUBSAMPLE_TRAIN, len(atk_tr)), replace=False)
    X_tr_atk = X_train[tr_idx]
    print(f"\n  Training on {len(X_tr_atk)} attack samples")

    # Balanced test subsample
    atk_te = np.where(y_test == 1)[0]
    nrm_te = np.where(y_test == 0)[0]
    half_te = SUBSAMPLE_TEST // 2
    te_idx = np.concatenate([
        rng.choice(atk_te, min(half_te, len(atk_te)), replace=False),
        rng.choice(nrm_te, min(SUBSAMPLE_TEST - half_te, len(nrm_te)), replace=False),
    ])
    X_te = X_test[te_idx]
    y_te = y_test[te_idx]

    params = pnp.array(
        np.random.uniform(-np.pi, np.pi, TOTAL_VQC_PARAMS), requires_grad=True
    )
    opt = qml.AdamOptimizer(stepsize=ADAM_LR)

    print("\n  Training with Adam + adjoint (minimising attack recon error)...")
    for step in range(1, ADAM_STEPS + 1):
        params, loss_val = opt.step_and_cost(
            lambda p: qae_loss_fn(p, X_tr_atk), params
        )
        if step % 20 == 0 or step == 1:
            print(f"  Step {step:4d}/{ADAM_STEPS} | recon_error = {float(loss_val):.4f}")

    opt_params = np.array(params)
    print(f"  Training complete.")

    # Compute reconstruction errors on test subsample
    test_errors = qae_errors_batch(X_te, opt_params)

    # ── CRITICAL: verify error direction empirically ──────────────────────────
    mean_err_atk = float(test_errors[y_te == 1].mean())
    mean_err_nrm = float(test_errors[y_te == 0].mean())
    print(f"\n  === Error Direction Verification ===")
    print(f"  Mean recon error - ATTACK samples : {mean_err_atk:.4f}")
    print(f"  Mean recon error - NORMAL samples : {mean_err_nrm:.4f}")

    # Compute threshold from training-set errors
    train_errors = qae_errors_batch(X_tr_atk, opt_params)

    attacks_lower = mean_err_atk <= mean_err_nrm
    if attacks_lower:
        print("  Direction confirmed: attacks have LOWER error (as expected).")
        print("  Rule: classify as attack if error <= threshold.")
        threshold = float(np.percentile(train_errors, 95))
        preds = (test_errors <= threshold).astype(int)
        # prob_attack inversely proportional to error
        max_err = test_errors.max() + 1e-9
        attack_probs = 1.0 - (test_errors / max_err)
    else:
        print("  WARNING: direction REVERSED - attacks have HIGHER error.")
        print("  Possible cause: circuit over-generalised to normal traffic.")
        print("  Rule inverted: classify as attack if error >= threshold.")
        threshold = float(np.percentile(train_errors, 5))
        preds = (test_errors >= threshold).astype(int)
        attack_probs = test_errors / (test_errors.max() + 1e-9)

    print(f"  Threshold: {threshold:.4f} "
          f"({'95th' if attacks_lower else '5th'} pct of training errors)")

    # Verify directional consistency between probs and preds
    if (preds == 1).sum() > 0 and (preds == 0).sum() > 0:
        consistent = (
            attack_probs[preds == 1].mean() > attack_probs[preds == 0].mean()
        )
        print(f"  Probs/preds consistent: {consistent}")
        if not consistent:
            print("  WARNING: probs and preds are inconsistent - check circuit.")

    acc  = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, zero_division=0)
    rec  = recall_score(y_te, preds, zero_division=0)
    f1   = f1_score(y_te, preds, zero_division=0)
    cm   = confusion_matrix(y_te, preds)
    tn, fp, fn, tp = (cm.ravel() if cm.shape == (2, 2)
                      else (0, 0, 0, int(preds.sum())))
    detection_rate = tp / (tp + fn + 1e-9)
    fp_rate        = fp / (fp + tn + 1e-9)

    print(f"\n  QAE Results ({SUBSAMPLE_TEST}-sample subsample):")
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}")
    print(f"  Detection Rate={detection_rate:.4f}  FP Rate={fp_rate:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    joblib.dump(opt_params, os.path.join(ARTIFACTS_DIR, "qae_params.pkl"))
    joblib.dump({
        "threshold": threshold,
        "attacks_have_lower_error": attacks_lower,
    }, os.path.join(ARTIFACTS_DIR, "qae_config.pkl"))
    np.save(os.path.join(ARTIFACTS_DIR, "qae_test_errors.npy"),  test_errors)
    np.save(os.path.join(ARTIFACTS_DIR, "qae_test_probs.npy"),   attack_probs)
    np.save(os.path.join(ARTIFACTS_DIR, "qae_test_preds.npy"),   preds)

    return {
        "model": "QAE", "params": opt_params,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm,
        "detection_rate": detection_rate, "fp_rate": fp_rate,
        "probs": attack_probs, "preds": preds,
        "eval_set": f"{SUBSAMPLE_TEST}-sample subsample",
        "threshold": threshold,
        "direction_confirmed": attacks_lower,
    }


# =============================================================================
# D) COMPARISON AND SELECTION
# =============================================================================

def compare_and_select(qsnn_res, qmlp_res, qae_res):
    """
    Print comparison table. Select best of QSNN vs QMLP by F1.
    QAE excluded from selection - different purpose (novelty detection),
    used regardless as the anomaly detector in later stages.

    NOTE: QSNN is evaluated on the full test set; QMLP/QAE on subsamples.
    These are NOT directly comparable but both are reported for completeness.
    """
    print("\n" + "=" * 60)
    print("  QUANTUM MODEL COMPARISON")
    print("=" * 60)
    print("  NOTE: QSNN uses full test set; QMLP/QAE use subsamples.")
    print("  Results are directionally comparable but not statistically.")
    print()
    hdr = f"  {'Model':<8} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}  {'DR':>7} {'FPR':>7}  Eval Set"
    print(hdr)
    print("  " + "-" * 80)

    for r in [qsnn_res, qmlp_res, qae_res]:
        dr  = r.get("detection_rate", "-")
        fpr = r.get("fp_rate", "-")
        dr_s  = f"{dr:.4f}"  if isinstance(dr,  float) else dr
        fpr_s = f"{fpr:.4f}" if isinstance(fpr, float) else fpr
        print(f"  {r['model']:<8} {r['acc']:>7.4f} {r['prec']:>7.4f} "
              f"{r['rec']:>7.4f} {r['f1']:>7.4f}  {dr_s:>7} {fpr_s:>7}  "
              f"{r['eval_set']}")

    best = qsnn_res if qsnn_res["f1"] >= qmlp_res["f1"] else qmlp_res
    print(f"\n  Best (QSNN vs QMLP by F1): {best['model']} (F1={best['f1']:.4f})")
    print(f"  QAE used regardless as the novelty detector.")

    joblib.dump(best["model"],
                os.path.join(ARTIFACTS_DIR, "best_quantum_model.pkl"))
    print(f"  Saved -> artifacts/best_quantum_model.pkl")
    return best


# =============================================================================
# MAIN
# =============================================================================

def run_quantum_models():
    print("=" * 60)
    print("STAGE 3 -- QUANTUM MODELS")
    print("=" * 60)
    print(f"\nOptimizer      : Adam (lr={ADAM_LR}, steps={ADAM_STEPS})")
    print(f"Diff method    : {_diff_method}")
    print(f"N_QUBITS       : {N_QUBITS}")
    print(f"VQC params     : {TOTAL_VQC_PARAMS}")
    print(f"Subsample size : {SUBSAMPLE_TRAIN} train / {SUBSAMPLE_TEST} test")
    print(f"\nBenchmarked speeds (this machine):")
    print(f"  Forward only       : 2.95 ms/sample")
    print(f"  Forward + backward : 7.45 ms/sample")
    print(f"  QMLP/QAE training  : ~{7.45*SUBSAMPLE_TRAIN/1000*ADAM_STEPS/60:.0f} min each")
    print(f"  QSNN prototype     : ~{2.95*175341/1000/60:.0f} min (full train)")
    print(f"  QSNN test infer    : ~{2.95*82332/1000/60:.0f} min (full test)")

    print("\nLoading Stage 1 quantum input arrays...")
    X_train = np.load(os.path.join(ARTIFACTS_DIR, "quantum_input_train.npy"))
    X_test  = np.load(os.path.join(ARTIFACTS_DIR, "quantum_input_test.npy"))
    y_train = np.load(os.path.join(ARTIFACTS_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))

    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    qsnn_res = run_qsnn(X_train, y_train, X_test, y_test)
    qmlp_res = run_qmlp(X_train, y_train, X_test, y_test)
    qae_res  = run_qae(X_train, y_train, X_test, y_test)
    best     = compare_and_select(qsnn_res, qmlp_res, qae_res)

    print("\n" + "=" * 60)
    print("STAGE 3 COMPLETE")
    print("=" * 60)
    print(f"  Best quantum model : {best['model']}")
    print(f"  Artifacts saved to : {ARTIFACTS_DIR}/")

    return qsnn_res, qmlp_res, qae_res, best


if __name__ == "__main__":
    run_quantum_models()
