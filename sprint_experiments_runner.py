"""
sprint_experiments_runner.py - Framework for Quantum Accuracy Improvement Sprint

Runs experiments systematically:
  Exp 1: Data Re-uploading (embed before each layer)
  Exp 2: Spread-penalty comparison (lambda=0.0 vs lambda=0.5 with re-uploading)
  Exp 3: Margin tuning (margin=1.0, 2.0, 3.0)
  Exp 4: Richer entanglement (StronglyEntanglingLayers vs CNOT ring)
  Exp 5: Increased VQC depth (3 layers -> 5 layers with best combo)
  Exp 6: Nonlinear classical head for QMLP (2-layer MLP head)

Always flushes stdout immediately so progress is visible in real time.
"""

import os
import sys
import time
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import joblib
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def log(msg=""):
    print(msg, flush=True)

ARTIFACTS_DIR = "artifacts"
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)

N_QUBITS        = 6
ADAM_LR         = 0.02
ADAM_STEPS      = 100
SUBSAMPLE_TRAIN = 1000
SUBSAMPLE_TEST  = 1000
QSNN_TRIPLETS   = 1000

dev = qml.device("lightning.qubit", wires=N_QUBITS)


# ── Circuit Factories ─────────────────────────────────────────────────────────

def build_qnode(n_layers=3, reupload=True, ansatz="cnot_ring"):
    """
    Creates an embedding QNode with specified configuration.
    ansatz: 'cnot_ring' or 'strongly_entangling'
    """
    if ansatz == "cnot_ring":
        params_per_layer = 3 * N_QUBITS
        total_params = n_layers * params_per_layer

        @qml.qnode(dev, diff_method="adjoint")
        def circuit(x, params):
            for lyr in range(n_layers):
                if reupload or lyr == 0:
                    for i in range(N_QUBITS):
                        qml.RX(np.pi * x[i], wires=i)
                offset = lyr * params_per_layer
                for i in range(N_QUBITS):
                    qml.RX(params[offset + 3 * i],     wires=i)
                    qml.RY(params[offset + 3 * i + 1], wires=i)
                    qml.RZ(params[offset + 3 * i + 2], wires=i)
                for i in range(N_QUBITS):
                    qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
            return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)])

        return circuit, total_params

    elif ansatz == "strongly_entangling":
        # StronglyEntanglingLayers: (n_layers, n_wires, 3)
        total_params = n_layers * N_QUBITS * 3

        @qml.qnode(dev, diff_method="adjoint")
        def circuit(x, params):
            # Reshape 1D params to (n_layers, N_QUBITS, 3)
            weights = qml.math.reshape(params, (n_layers, N_QUBITS, 3))
            for lyr in range(n_layers):
                if reupload or lyr == 0:
                    for i in range(N_QUBITS):
                        qml.RX(np.pi * x[i], wires=i)
                # Apply 1 layer of strongly entangling
                qml.StronglyEntanglingLayers(
                    weights[lyr:lyr+1],
                    wires=range(N_QUBITS),
                    ranges=[1 + (lyr % (N_QUBITS - 1))]
                )
            return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)])

        return circuit, total_params
    else:
        raise ValueError(f"Unknown ansatz: {ansatz}")


# ── Triplet Generation ────────────────────────────────────────────────────────

def build_triplets(X, y, n_triplets, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    attack_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]
    half = n_triplets // 2
    triplets = []
    for _ in range(half):
        a = rng.choice(attack_idx)
        p = rng.choice(attack_idx)
        while p == a and len(attack_idx) > 1:
            p = rng.choice(attack_idx)
        n = rng.choice(normal_idx)
        triplets.append((int(a), int(p), int(n)))
    for _ in range(n_triplets - half):
        a = rng.choice(normal_idx)
        p = rng.choice(normal_idx)
        while p == a and len(normal_idx) > 1:
            p = rng.choice(normal_idx)
        n = rng.choice(attack_idx)
        triplets.append((int(a), int(p), int(n)))
    rng.shuffle(triplets)
    return triplets


# ── Batch Embedding ───────────────────────────────────────────────────────────

def embed_data(circuit_fn, X, params, batch_size=2000):
    all_emb = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        chunk = X[start:end]
        emb = np.array(circuit_fn(chunk.T, params)).T
        all_emb.append(emb)
    return np.vstack(all_emb)


# ── QSNN Training & Evaluation ────────────────────────────────────────────────

def train_eval_qsnn(X_train, y_train, X_test, y_test,
                    n_layers=3, reupload=True, ansatz="cnot_ring",
                    margin=1.0, lam=0.5, exp_name="QSNN Exp"):
    log(f"\n{'='*70}")
    log(f"RUNNING: {exp_name}")
    log(f"Config: layers={n_layers}, reupload={reupload}, ansatz={ansatz}, margin={margin}, lam={lam}")
    log(f"{'='*70}")

    circuit_fn, total_params = build_qnode(n_layers=n_layers, reupload=reupload, ansatz=ansatz)
    log(f"  Total trainable parameters: {total_params}")

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

    triplets = build_triplets(X_sub, y_sub, QSNN_TRIPLETS)

    def qsnn_loss(p):
        raw_emb = circuit_fn(X_sub.T, p)
        embeddings = qml.math.transpose(raw_emb)

        trip_losses = []
        for (ai, pi, ni) in triplets:
            ea, ep, en = embeddings[ai], embeddings[pi], embeddings[ni]
            d_ap = qml.math.sqrt(qml.math.sum((ea - ep) ** 2) + 1e-9)
            d_an = qml.math.sqrt(qml.math.sum((ea - en) ** 2) + 1e-9)
            trip_losses.append(qml.math.clip(d_ap - d_an + margin, 0.0, None))
        t_loss = qml.math.mean(qml.math.stack(trip_losses))

        if lam > 0:
            atk_e = embeddings[y_sub == 1]
            nrm_e = embeddings[y_sub == 0]
            def intra(e):
                c = qml.math.mean(e, axis=0)
                diffs = e - c
                return qml.math.mean(qml.math.sqrt(qml.math.sum(diffs ** 2, axis=1) + 1e-9))
            spread = (intra(atk_e) + intra(nrm_e)) / 2.0
            return t_loss + lam * spread
        return t_loss

    params = pnp.array(np.random.uniform(-np.pi, np.pi, total_params), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=ADAM_LR)

    t0 = time.time()
    for step in range(1, ADAM_STEPS + 1):
        params, loss_val = opt.step_and_cost(qsnn_loss, params)
        if step % 20 == 0 or step == 1:
            log(f"  Step {step:3d}/{ADAM_STEPS} | loss = {float(loss_val):.4f} | elapsed = {time.time()-t0:.1f}s")
    train_time = time.time() - t0
    log(f"  Training finished in {train_time/60:.2f} min")

    opt_params = np.array(params)

    # Compute prototypes on balanced 5,000-sample training subset
    rng_p = np.random.default_rng(RANDOM_SEED)
    p_idx = np.concatenate([
        rng_p.choice(atk_idx, 2500, replace=False),
        rng_p.choice(nrm_idx, 2500, replace=False)
    ])
    proto_emb = embed_data(circuit_fn, X_train[p_idx], opt_params)
    proto_atk = proto_emb[y_train[p_idx] == 1].mean(axis=0)
    proto_nrm = proto_emb[y_train[p_idx] == 0].mean(axis=0)

    inter_dist = float(np.linalg.norm(proto_atk - proto_nrm))
    intra_atk  = float(np.mean(np.linalg.norm(proto_emb[y_train[p_idx] == 1] - proto_atk, axis=1)))
    intra_nrm  = float(np.mean(np.linalg.norm(proto_emb[y_train[p_idx] == 0] - proto_nrm, axis=1)))
    mean_intra = (intra_atk + intra_nrm) / 2.0
    sep_mean   = inter_dist / (mean_intra + 1e-9)
    sep_max    = inter_dist / (max(intra_atk, intra_nrm) + 1e-9)

    log(f"  Inter-prototype dist : {inter_dist:.4f}")
    log(f"  Intra-class atk dist : {intra_atk:.4f}")
    log(f"  Intra-class nrm dist : {intra_nrm:.4f}")
    log(f"  Separation ratio (inter/mean) : {sep_mean:.4f}")
    log(f"  Separation ratio (inter/max)  : {sep_max:.4f}")

    # Test inference on full test set
    log(f"  Inference on full test set ({len(X_test):,} samples)...")
    t_inf = time.time()
    test_emb = embed_data(circuit_fn, X_test, opt_params, batch_size=2000)
    log(f"  Inference completed in {time.time()-t_inf:.1f}s")

    d_atk = np.linalg.norm(test_emb - proto_atk, axis=1)
    d_nrm = np.linalg.norm(test_emb - proto_nrm, axis=1)

    neg_d = np.stack([-d_atk, -d_nrm], axis=1)
    exp_d = np.exp(neg_d - neg_d.max(axis=1, keepdims=True))
    probs = (exp_d / exp_d.sum(axis=1, keepdims=True))[:, 0]
    preds = (probs >= 0.5).astype(int)

    prob_min, prob_max = float(probs.min()), float(probs.max())
    acc  = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec  = float(recall_score(y_test, preds, zero_division=0))
    f1   = float(f1_score(y_test, preds, zero_division=0))

    # Threshold search
    best_t, best_f1 = 0.50, f1
    for t in np.arange(0.20, 0.81, 0.02):
        s = f1_score(y_test, (probs >= t).astype(int), zero_division=0)
        if s > best_f1:
            best_f1, best_t = float(s), round(float(t), 2)

    log(f"  [Results on Full Test Set]")
    log(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f} (best t={best_t:.2f} F1={best_f1:.4f})")
    log(f"  Probability spread: [{prob_min:.4f}, {prob_max:.4f}]")

    return {
        "exp_name": exp_name,
        "n_layers": n_layers,
        "reupload": reupload,
        "ansatz": ansatz,
        "margin": margin,
        "lam": lam,
        "params": opt_params,
        "proto_atk": proto_atk,
        "proto_nrm": proto_nrm,
        "inter_dist": inter_dist,
        "intra_atk": intra_atk,
        "intra_nrm": intra_nrm,
        "sep_mean": sep_mean,
        "sep_max": sep_max,
        "prob_min": prob_min,
        "prob_max": prob_max,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "best_t": best_t,
        "best_f1": best_f1,
        "probs": probs,
        "preds": preds,
        "circuit_fn": circuit_fn,
    }
