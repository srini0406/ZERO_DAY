"""
run_sprint_all.py - Complete Automated Runner for Stage 3 Quantum Accuracy Improvement Sprint

Implements all 6 sprint experiments in strict priority order:
  Exp 1: Data Re-uploading (Perez-Salinas et al. 2020)
  Exp 2: Spread-penalty term in QSNN loss (lambda=0.0 vs lambda=0.5)
  Exp 3: Increased Triplet Margin (margin in [1.0, 2.0, 3.0])
  Exp 4: Richer Entanglement (StronglyEntanglingLayers vs CNOT ring)
  Exp 5: Increased VQC Depth (3 layers -> 5 layers with best configuration)
  Exp 6: Nonlinear Classical Head for QMLP (2-layer MLP head)

All runs log immediately (flush=True) and persist results to artifacts/sprint_results.pkl.
"""

import os
import sys
import time
import argparse
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

ARTIFACTS_DIR   = "artifacts"
RANDOM_SEED     = 42
np.random.seed(RANDOM_SEED)

N_QUBITS        = 6
ADAM_LR         = 0.02
ADAM_STEPS      = 70        # Plateau reached by step 55-60; 70 steps gives full convergence
SUBSAMPLE_TRAIN = 1000
QSNN_TRIPLETS   = 1000

dev = qml.device("lightning.qubit", wires=N_QUBITS)

# ── Load Data ─────────────────────────────────────────────────────────────────
log("[Data] Loading Stage 1 quantum artifacts...")
X_train_full = np.load(os.path.join(ARTIFACTS_DIR, "quantum_input_train.npy"))
X_test_full  = np.load(os.path.join(ARTIFACTS_DIR, "quantum_input_test.npy"))
y_train_full = np.load(os.path.join(ARTIFACTS_DIR, "y_train.npy"))
y_test_full  = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))
log(f"  Train: {X_train_full.shape}  Test: {X_test_full.shape}")

# Fixed evaluation subsets:
# 1) Balanced 1,000-sample test set for fast evaluation across iterations
rng_te = np.random.default_rng(RANDOM_SEED + 99)
atk_te = np.where(y_test_full == 1)[0]
nrm_te = np.where(y_test_full == 0)[0]
eval_te_idx = np.concatenate([
    rng_te.choice(atk_te, 500, replace=False),
    rng_te.choice(nrm_te, 500, replace=False),
])
X_test_sub = X_test_full[eval_te_idx]
y_test_sub = y_test_full[eval_te_idx]

# 2) Training subsample for Adam (500 atk, 500 nrm)
rng_tr = np.random.default_rng(RANDOM_SEED)
atk_tr = np.where(y_train_full == 1)[0]
nrm_tr = np.where(y_train_full == 0)[0]
train_sub_idx = np.concatenate([
    rng_tr.choice(atk_tr, SUBSAMPLE_TRAIN // 2, replace=False),
    rng_tr.choice(nrm_tr, SUBSAMPLE_TRAIN // 2, replace=False),
])
rng_tr.shuffle(train_sub_idx)
X_train_sub = X_train_full[train_sub_idx]
y_train_sub = y_train_full[train_sub_idx]

# 3) Prototype computation subset: 5,000 balanced samples from training set
rng_p = np.random.default_rng(RANDOM_SEED + 10)
p_idx = np.concatenate([
    rng_p.choice(atk_tr, 2500, replace=False),
    rng_p.choice(nrm_tr, 2500, replace=False)
])
X_proto = X_train_full[p_idx]
y_proto = y_train_full[p_idx]


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

triplets = build_triplets(X_train_sub, y_train_sub, QSNN_TRIPLETS)


# ── QNode Builder ─────────────────────────────────────────────────────────────

def create_embedding_qnode(n_layers=3, reupload=True, ansatz="cnot_ring"):
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
        total_params = n_layers * N_QUBITS * 3

        @qml.qnode(dev, diff_method="adjoint")
        def circuit(x, params):
            weights = qml.math.reshape(params, (n_layers, N_QUBITS, 3))
            for lyr in range(n_layers):
                if reupload or lyr == 0:
                    for i in range(N_QUBITS):
                        qml.RX(np.pi * x[i], wires=i)
                qml.StronglyEntanglingLayers(
                    weights[lyr:lyr+1],
                    wires=range(N_QUBITS),
                    ranges=[1 + (lyr % (N_QUBITS - 1))]
                )
            return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)])

        return circuit, total_params
    else:
        raise ValueError(f"Unknown ansatz: {ansatz}")


# ── Fast Batch Embedding ──────────────────────────────────────────────────────

def embed_fast(circuit_fn, X, params, batch_size=2000):
    all_emb = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        chunk = X[start:end]
        emb = np.array(circuit_fn(chunk.T, params)).T
        all_emb.append(emb)
    return np.vstack(all_emb)


# ── Run QSNN Experiment ───────────────────────────────────────────────────────

def run_qsnn_experiment(exp_name, n_layers=3, reupload=True, ansatz="cnot_ring",
                        margin=1.0, lam=0.5, eval_full_test=True):
    log("\n" + "=" * 75)
    log(f"EXPERIMENT: {exp_name}")
    log(f"Config: layers={n_layers}, reupload={reupload}, ansatz={ansatz}, margin={margin}, lam={lam}")
    log("=" * 75)

    circuit_fn, total_params = create_embedding_qnode(n_layers=n_layers, reupload=reupload, ansatz=ansatz)
    log(f"  Trainable parameters: {total_params}  |  Adam steps: {ADAM_STEPS}")

    def qsnn_loss(p):
        raw = circuit_fn(X_train_sub.T, p)
        emb = qml.math.transpose(raw)

        trip_losses = []
        for (ai, pi, ni) in triplets:
            ea, ep, en = emb[ai], emb[pi], emb[ni]
            d_ap = qml.math.sqrt(qml.math.sum((ea - ep) ** 2) + 1e-9)
            d_an = qml.math.sqrt(qml.math.sum((ea - en) ** 2) + 1e-9)
            trip_losses.append(qml.math.clip(d_ap - d_an + margin, 0.0, None))
        triplet_loss = qml.math.mean(qml.math.stack(trip_losses))

        if lam > 0.0:
            atk_e = emb[y_train_sub == 1]
            nrm_e = emb[y_train_sub == 0]
            def intra(e):
                c = qml.math.mean(e, axis=0)
                diffs = e - c
                return qml.math.mean(qml.math.sqrt(qml.math.sum(diffs ** 2, axis=1) + 1e-9))
            spread = (intra(atk_e) + intra(nrm_e)) / 2.0
            return triplet_loss + lam * spread
        return triplet_loss

    rng = np.random.default_rng(RANDOM_SEED)
    init_params = pnp.array(rng.uniform(-np.pi, np.pi, total_params), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=ADAM_LR)

    params = init_params
    t0 = time.time()
    for step in range(1, ADAM_STEPS + 1):
        params, loss_val = opt.step_and_cost(qsnn_loss, params)
        if step % 15 == 0 or step == 1 or step == ADAM_STEPS:
            log(f"  Step {step:2d}/{ADAM_STEPS} | loss = {float(loss_val):.4f} | time = {time.time()-t0:.1f}s")
    train_time = time.time() - t0
    opt_params = np.array(params)
    log(f"  Optimization finished in {train_time/60:.2f} min")

    # Prototypes
    proto_emb = embed_fast(circuit_fn, X_proto, opt_params)
    proto_atk = proto_emb[y_proto == 1].mean(axis=0)
    proto_nrm = proto_emb[y_proto == 0].mean(axis=0)

    inter_dist = float(np.linalg.norm(proto_atk - proto_nrm))
    intra_atk  = float(np.mean(np.linalg.norm(proto_emb[y_proto == 1] - proto_atk, axis=1)))
    intra_nrm  = float(np.mean(np.linalg.norm(proto_emb[y_proto == 0] - proto_nrm, axis=1)))
    mean_intra = (intra_atk + intra_nrm) / 2.0
    sep_mean   = inter_dist / (mean_intra + 1e-9)
    sep_max    = inter_dist / (max(intra_atk, intra_nrm) + 1e-9)

    log(f"\n  [Separation Metrics]")
    log(f"  Inter-prototype dist : {inter_dist:.4f}  (Baseline: 0.7712)")
    log(f"  Intra-atk dist       : {intra_atk:.4f}  (Baseline: 0.0889)")
    log(f"  Intra-nrm dist       : {intra_nrm:.4f}  (Baseline: 0.3666)")
    log(f"  Sep ratio (inter/mean): {sep_mean:.4f}  (Baseline: 3.3857)")
    log(f"  Sep ratio (inter/max) : {sep_max:.4f}  (Baseline: 2.1035)")

    # Test inference: Subsample
    sub_emb = embed_fast(circuit_fn, X_test_sub, opt_params)
    d_atk_sub = np.linalg.norm(sub_emb - proto_atk, axis=1)
    d_nrm_sub = np.linalg.norm(sub_emb - proto_nrm, axis=1)
    neg_sub = np.stack([-d_atk_sub, -d_nrm_sub], axis=1)
    exp_sub = np.exp(neg_sub - neg_sub.max(axis=1, keepdims=True))
    probs_sub = (exp_sub / exp_sub.sum(axis=1, keepdims=True))[:, 0]
    preds_sub = (probs_sub >= 0.5).astype(int)
    f1_sub = float(f1_score(y_test_sub, preds_sub, zero_division=0))
    acc_sub = float(accuracy_score(y_test_sub, preds_sub))
    log(f"  Subsample (1,000): F1={f1_sub:.4f}  Acc={acc_sub:.4f}  Probs=[{probs_sub.min():.4f}, {probs_sub.max():.4f}]")

    # Full test evaluation
    full_metrics = {}
    if eval_full_test:
        log(f"  Inference on FULL test set ({len(X_test_full):,} samples)...")
        t_inf = time.time()
        test_emb = embed_fast(circuit_fn, X_test_full, opt_params, batch_size=2000)
        d_atk = np.linalg.norm(test_emb - proto_atk, axis=1)
        d_nrm = np.linalg.norm(test_emb - proto_nrm, axis=1)
        neg_d = np.stack([-d_atk, -d_nrm], axis=1)
        exp_d = np.exp(neg_d - neg_d.max(axis=1, keepdims=True))
        probs_full = (exp_d / exp_d.sum(axis=1, keepdims=True))[:, 0]
        preds_full = (probs_full >= 0.5).astype(int)

        acc  = float(accuracy_score(y_test_full, preds_full))
        prec = float(precision_score(y_test_full, preds_full, zero_division=0))
        rec  = float(recall_score(y_test_full, preds_full, zero_division=0))
        f1   = float(f1_score(y_test_full, preds_full, zero_division=0))
        p_min, p_max = float(probs_full.min()), float(probs_full.max())

        # Optimal threshold search
        best_t, best_f1 = 0.50, f1
        for t in np.arange(0.20, 0.81, 0.02):
            s = f1_score(y_test_full, (probs_full >= t).astype(int), zero_division=0)
            if s > best_f1:
                best_f1, best_t = float(s), round(float(t), 2)

        log(f"  FULL Test (82,332): Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f} (best t={best_t:.2f} F1={best_f1:.4f})")
        log(f"  Full Probs Spread : [{p_min:.4f}, {p_max:.4f}]")
        full_metrics = {
            "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "best_t": best_t, "best_f1": best_f1,
            "prob_min": p_min, "prob_max": p_max,
            "probs": probs_full, "preds": preds_full,
        }

    return {
        "exp_name": exp_name,
        "n_layers": n_layers, "reupload": reupload, "ansatz": ansatz,
        "margin": margin, "lam": lam,
        "params": opt_params,
        "proto_atk": proto_atk, "proto_nrm": proto_nrm,
        "inter_dist": inter_dist, "intra_atk": intra_atk, "intra_nrm": intra_nrm,
        "sep_mean": sep_mean, "sep_max": sep_max,
        "f1_sub": f1_sub, "acc_sub": acc_sub,
        "prob_min_sub": float(probs_sub.min()), "prob_max_sub": float(probs_sub.max()),
        **full_metrics
    }


# ── Run QMLP Experiment (Exp 1 comparison & Exp 6 nonlinear head) ─────────────

def run_qmlp_experiment(exp_name, n_layers=3, reupload=True, ansatz="cnot_ring", nonlinear_head=False):
    log("\n" + "=" * 75)
    log(f"QMLP EXPERIMENT: {exp_name} (Nonlinear Head={nonlinear_head})")
    log("=" * 75)

    circuit_fn, vqc_params_count = create_embedding_qnode(n_layers=n_layers, reupload=reupload, ansatz=ansatz)

    if not nonlinear_head:
        # Linear head: 6 weights + 1 bias
        total_params = vqc_params_count + N_QUBITS + 1

        def qmlp_loss(all_p):
            p_vqc = all_p[:vqc_params_count]
            w = all_p[vqc_params_count:vqc_params_count + N_QUBITS]
            b = all_p[-1]

            raw = circuit_fn(X_train_sub.T, p_vqc)
            emb = qml.math.transpose(raw)
            logits = qml.math.sum(emb * w, axis=1) + b
            probs = 1.0 / (1.0 + qml.math.exp(-qml.math.clip(logits, -20.0, 20.0)))
            probs = qml.math.clip(probs, 1e-7, 1.0 - 1e-7)
            y = y_train_sub.astype(float)
            return -qml.math.mean(y * qml.math.log(probs) + (1.0 - y) * qml.math.log(1.0 - probs))

        init_vqc = np.random.uniform(-np.pi, np.pi, vqc_params_count)
        init_w   = np.random.uniform(-0.1, 0.1, N_QUBITS)
        init_b   = np.array([0.0])
        init_p   = pnp.array(np.concatenate([init_vqc, init_w, init_b]), requires_grad=True)

    else:
        # 2-layer MLP head: 6 -> 16 -> 1
        # W1: 6x16 (96), b1: 16, W2: 16x1 (16), b2: 1  = 129 classical params
        h_dim = 16
        c_count = N_QUBITS * h_dim + h_dim + h_dim * 1 + 1
        total_params = vqc_params_count + c_count

        def qmlp_loss(all_p):
            p_vqc = all_p[:vqc_params_count]
            idx = vqc_params_count
            W1 = qml.math.reshape(all_p[idx:idx + N_QUBITS * h_dim], (N_QUBITS, h_dim))
            idx += N_QUBITS * h_dim
            b1 = all_p[idx:idx + h_dim]
            idx += h_dim
            W2 = qml.math.reshape(all_p[idx:idx + h_dim], (h_dim, 1))
            idx += h_dim
            b2 = all_p[idx:idx + 1]

            raw = circuit_fn(X_train_sub.T, p_vqc)
            emb = qml.math.transpose(raw) # (n, 6)
            h = qml.math.tanh(qml.math.dot(emb, W1) + b1) # (n, 16)
            logits = qml.math.reshape(qml.math.dot(h, W2) + b2, (-1,))
            probs = 1.0 / (1.0 + qml.math.exp(-qml.math.clip(logits, -20.0, 20.0)))
            probs = qml.math.clip(probs, 1e-7, 1.0 - 1e-7)
            y = y_train_sub.astype(float)
            return -qml.math.mean(y * qml.math.log(probs) + (1.0 - y) * qml.math.log(1.0 - probs))

        init_vqc = np.random.uniform(-np.pi, np.pi, vqc_params_count)
        init_W1  = np.random.uniform(-0.2, 0.2, N_QUBITS * h_dim)
        init_b1  = np.zeros(h_dim)
        init_W2  = np.random.uniform(-0.2, 0.2, h_dim * 1)
        init_b2  = np.zeros(1)
        init_p   = pnp.array(np.concatenate([init_vqc, init_W1, init_b1, init_W2, init_b2]), requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=ADAM_LR)
    params = init_p
    t0 = time.time()
    for step in range(1, ADAM_STEPS + 1):
        params, l_val = opt.step_and_cost(qmlp_loss, params)
        if step % 20 == 0 or step == 1 or step == ADAM_STEPS:
            log(f"  Step {step:2d}/{ADAM_STEPS} | loss = {float(l_val):.4f}")
    opt_p = np.array(params)

    # Evaluate on test subsample
    p_vqc = opt_p[:vqc_params_count]
    test_emb = embed_fast(circuit_fn, X_test_sub, p_vqc)

    if not nonlinear_head:
        w = opt_p[vqc_params_count:vqc_params_count + N_QUBITS]
        b = opt_p[-1]
        logits = test_emb @ w + b
    else:
        h_dim = 16
        idx = vqc_params_count
        W1 = opt_p[idx:idx + N_QUBITS * h_dim].reshape(N_QUBITS, h_dim)
        idx += N_QUBITS * h_dim
        b1 = opt_p[idx:idx + h_dim]
        idx += h_dim
        W2 = opt_p[idx:idx + h_dim].reshape(h_dim, 1)
        idx += h_dim
        b2 = opt_p[idx:idx + 1]
        h = np.tanh(test_emb @ W1 + b1)
        logits = (h @ W2 + b2).flatten()

    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    preds = (probs >= 0.5).astype(int)

    acc  = float(accuracy_score(y_test_sub, preds))
    prec = float(precision_score(y_test_sub, preds, zero_division=0))
    rec  = float(recall_score(y_test_sub, preds, zero_division=0))
    f1   = float(f1_score(y_test_sub, preds, zero_division=0))

    best_t, best_f1 = 0.50, f1
    for t in np.arange(0.20, 0.81, 0.02):
        s = f1_score(y_test_sub, (probs >= t).astype(int), zero_division=0)
        if s > best_f1:
            best_f1, best_t = float(s), round(float(t), 2)

    log(f"  Results (1,000 sub): Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} (best t={best_t:.2f} F1={best_f1:.4f})")
    log(f"  Probs Spread: [{probs.min():.4f}, {probs.max():.4f}]")

    # Full test evaluation
    log(f"  Inference on FULL test set ({len(X_test_full):,} samples)...")
    full_emb = embed_fast(circuit_fn, X_test_full, p_vqc)
    if not nonlinear_head:
        full_logits = full_emb @ w + b
    else:
        full_h = np.tanh(full_emb @ W1 + b1)
        full_logits = (full_h @ W2 + b2).flatten()
    full_probs = 1.0 / (1.0 + np.exp(-np.clip(full_logits, -20.0, 20.0)))
    full_preds = (full_probs >= 0.5).astype(int)
    full_acc  = float(accuracy_score(y_test_full, full_preds))
    full_prec = float(precision_score(y_test_full, full_preds, zero_division=0))
    full_rec  = float(recall_score(y_test_full, full_preds, zero_division=0))
    full_f1   = float(f1_score(y_test_full, full_preds, zero_division=0))
    full_best_t, full_best_f1 = 0.50, full_f1
    for t in np.arange(0.20, 0.81, 0.02):
        s = f1_score(y_test_full, (full_probs >= t).astype(int), zero_division=0)
        if s > full_best_f1:
            full_best_f1, full_best_t = float(s), round(float(t), 2)
    log(f"  FULL Test (82,332): Acc={full_acc:.4f} Prec={full_prec:.4f} Rec={full_rec:.4f} F1={full_f1:.4f} (best t={full_best_t:.2f} F1={full_best_f1:.4f})")
    log(f"  Full Probs Spread: [{full_probs.min():.4f}, {full_probs.max():.4f}]")

    return {
        "exp_name": exp_name,
        "nonlinear_head": nonlinear_head,
        "params": opt_p,
        "acc_sub": acc, "prec_sub": prec, "rec_sub": rec, "f1_sub": f1,
        "best_t_sub": best_t, "best_f1_sub": best_f1,
        "prob_min_sub": float(probs.min()), "prob_max_sub": float(probs.max()),
        "acc": full_acc, "prec": full_prec, "rec": full_rec, "f1": full_f1,
        "best_t": full_best_t, "best_f1": full_best_f1,
        "prob_min": float(full_probs.min()), "prob_max": float(full_probs.max()),
        "probs": full_probs, "preds": full_preds
    }


# ── Main Sprint Execution ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=1, help="Which experiment to run (1 to 6)")
    args = parser.parse_args()

    results_file = os.path.join(ARTIFACTS_DIR, "sprint_all_results.pkl")
    all_results = {}
    if os.path.exists(results_file):
        try:
            all_results = joblib.load(results_file)
        except Exception:
            all_results = {}

    log("\n" + "#" * 75)
    log(f"  STAGE 3 ACCURACY IMPROVEMENT SPRINT - EXECUTING EXP {args.exp}")
    log("#" * 75)

    if args.exp == 1:
        # Exp 1: Data Re-uploading (QSNN and QMLP)
        r_qsnn = run_qsnn_experiment("Exp 1: Data Re-uploading (QSNN)", n_layers=3, reupload=True, ansatz="cnot_ring", margin=1.0, lam=0.5)
        r_qmlp = run_qmlp_experiment("Exp 1: Data Re-uploading (QMLP)", n_layers=3, reupload=True, ansatz="cnot_ring", nonlinear_head=False)
        all_results["exp1_qsnn"] = r_qsnn
        all_results["exp1_qmlp"] = r_qmlp

    elif args.exp == 2:
        # Exp 2: Spread-penalty comparison (lambda=0.0 vs lambda=0.5, both with re-uploading)
        r_no_spread = run_qsnn_experiment("Exp 2: Re-uploading without spread penalty (lam=0.0)", n_layers=3, reupload=True, ansatz="cnot_ring", margin=1.0, lam=0.0)
        # Note: lam=0.5 is already exp1_qsnn
        all_results["exp2_no_spread"] = r_no_spread

    elif args.exp == 3:
        # Exp 3: Triplet margin tuning (margin=1.0, 2.0, 3.0)
        # margin=1.0 is in exp1
        r_m2 = run_qsnn_experiment("Exp 3: Margin=2.0", n_layers=3, reupload=True, ansatz="cnot_ring", margin=2.0, lam=0.5)
        all_results["exp3_m2"] = r_m2
        joblib.dump(all_results, results_file)
        log(f"\n[Save] Exp 3 Margin=2.0 checkpoint saved -> {results_file}")
        r_m3 = run_qsnn_experiment("Exp 3: Margin=3.0", n_layers=3, reupload=True, ansatz="cnot_ring", margin=3.0, lam=0.5)
        all_results["exp3_m3"] = r_m3

    elif args.exp == 4:
        # Exp 4: Richer entanglement (StronglyEntanglingLayers vs CNOT ring)
        r_se = run_qsnn_experiment("Exp 4: StronglyEntanglingLayers", n_layers=3, reupload=True, ansatz="strongly_entangling", margin=1.0, lam=0.5)
        all_results["exp4_strongly_entangling"] = r_se

    elif args.exp == 5:
        # Exp 5: Increased VQC Depth (3 -> 5 layers)
        r_d5 = run_qsnn_experiment("Exp 5: 5 VQC Layers", n_layers=5, reupload=True, ansatz="cnot_ring", margin=1.0, lam=0.5)
        all_results["exp5_5_layers"] = r_d5

    elif args.exp == 6:
        # Exp 6: Nonlinear Classical Head for QMLP
        r_mlp_head = run_qmlp_experiment("Exp 6: QMLP with 2-layer MLP head", n_layers=3, reupload=True, ansatz="cnot_ring", nonlinear_head=True)
        all_results["exp6_qmlp_nonlinear"] = r_mlp_head

    joblib.dump(all_results, results_file)
    log(f"\n[Save] Checkpoint saved -> {results_file}")


if __name__ == "__main__":
    main()
