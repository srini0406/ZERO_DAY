"""
sprint_step1_reuploading.py - Experiment 1: Data Re-uploading

Technique:
Re-embed input features before EACH VQC layer:
embed(x) -> layer1 -> embed(x) -> layer2 -> embed(x) -> layer3
Reference: Perez-Salinas et al. 2020 ("Data re-uploading for a universal quantum classifier")

Compares QSNN and QMLP against baseline.
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

ARTIFACTS_DIR = "artifacts"
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)

N_QUBITS         = 6
N_VQC_LAYERS     = 3
PARAMS_PER_LAYER = 3 * N_QUBITS
TOTAL_VQC_PARAMS = N_VQC_LAYERS * PARAMS_PER_LAYER  # 54

ADAM_LR         = 0.02
ADAM_STEPS      = 100
SUBSAMPLE_TRAIN = 1000
SUBSAMPLE_TEST  = 1000
QSNN_TRIPLETS   = 1000

dev = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(dev, diff_method="adjoint")
def embedding_circuit_reupload(x, params):
    """
    Data re-uploading: re-embed x before each VQC layer.
    """
    for lyr in range(N_VQC_LAYERS):
        for i in range(N_QUBITS):
            qml.RX(np.pi * x[i], wires=i)
        offset = lyr * PARAMS_PER_LAYER
        for i in range(N_QUBITS):
            qml.RX(params[offset + 3 * i],     wires=i)
            qml.RY(params[offset + 3 * i + 1], wires=i)
            qml.RZ(params[offset + 3 * i + 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)])


def embed_batch(X, params, batch_size=2000):
    all_emb = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        chunk = X[start:end]
        # PennyLane broadcasting on lightning.qubit for fast embedding
        emb = np.array(embedding_circuit_reupload(chunk.T, params)).T
        all_emb.append(emb)
    return np.vstack(all_emb)


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


def qsnn_loss_fn(params, X, y, triplets, margin=1.0, lam=0.5):
    # Broadcast forward through embedding circuit
    raw_emb = embedding_circuit_reupload(X.T, params) # (N_QUBITS, n)
    embeddings = qml.math.transpose(raw_emb)          # (n, N_QUBITS)

    # Triplet loss
    trip_losses = []
    for (ai, pi, ni) in triplets:
        ea, ep, en = embeddings[ai], embeddings[pi], embeddings[ni]
        d_ap = qml.math.sqrt(qml.math.sum((ea - ep) ** 2) + 1e-9)
        d_an = qml.math.sqrt(qml.math.sum((ea - en) ** 2) + 1e-9)
        trip_losses.append(qml.math.clip(d_ap - d_an + margin, 0.0, None))
    triplet_loss = qml.math.mean(qml.math.stack(trip_losses))

    # Spread penalty
    atk_emb = embeddings[y == 1]
    nrm_emb = embeddings[y == 0]

    def mean_intra_dist(emb):
        centroid = qml.math.mean(emb, axis=0)
        diffs = emb - centroid
        return qml.math.mean(qml.math.sqrt(
            qml.math.sum(diffs ** 2, axis=1) + 1e-9
        ))

    spread = (mean_intra_dist(atk_emb) + mean_intra_dist(nrm_emb)) / 2.0
    return triplet_loss + lam * spread


def qmlp_loss_fn(all_params, X, y):
    params_vqc = all_params[:TOTAL_VQC_PARAMS]
    w = all_params[TOTAL_VQC_PARAMS:TOTAL_VQC_PARAMS + N_QUBITS]
    b = all_params[-1]

    raw_emb = embedding_circuit_reupload(X.T, params_vqc)
    embeddings = qml.math.transpose(raw_emb)
    logits = qml.math.sum(embeddings * w, axis=1) + b
    probs = 1.0 / (1.0 + qml.math.exp(-qml.math.clip(logits, -20.0, 20.0)))
    probs = qml.math.clip(probs, 1e-7, 1.0 - 1e-7)
    return -qml.math.mean(
        y * qml.math.log(probs) + (1.0 - y) * qml.math.log(1.0 - probs)
    )


def run_experiment_1():
    print("=" * 70)
    print("EXPERIMENT 1: DATA RE-UPLOADING")
    print("Architecture: embed(x) -> layer1 -> embed(x) -> layer2 -> embed(x) -> layer3")
    print(f"Device: lightning.qubit + adjoint  |  Parameters: {TOTAL_VQC_PARAMS}")
    print("=" * 70)

    # 1. Load data
    X_train = np.load(os.path.join(ARTIFACTS_DIR, "quantum_input_train.npy"))
    X_test  = np.load(os.path.join(ARTIFACTS_DIR, "quantum_input_test.npy"))
    y_train = np.load(os.path.join(ARTIFACTS_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))

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

    # ─────────────────────────────────────────────────────────────────────────
    # A) Train QSNN with Re-uploading
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Training QSNN with Data Re-uploading ---")
    params = pnp.array(
        np.random.uniform(-np.pi, np.pi, TOTAL_VQC_PARAMS),
        requires_grad=True,
    )
    opt = qml.AdamOptimizer(stepsize=ADAM_LR)

    t0 = time.time()
    for step in range(1, ADAM_STEPS + 1):
        params, loss_val = opt.step_and_cost(
            lambda p: qsnn_loss_fn(p, X_sub, y_sub, triplets, margin=1.0, lam=0.5),
            params
        )
        if step % 10 == 0 or step == 1:
            print(f"  Step {step:3d}/{ADAM_STEPS} | loss = {float(loss_val):.4f} | time = {time.time()-t0:.1f}s")
    qsnn_time = time.time() - t0
    qsnn_opt_params = np.array(params)
    print(f"  QSNN training finished in {qsnn_time/60:.2f} min")

    # Compute prototypes on training subsample + sample for fast verification
    # Using 5000 balanced train samples for representative prototype
    rng_p = np.random.default_rng(RANDOM_SEED)
    p_atk_idx = rng_p.choice(atk_idx, 2500, replace=False)
    p_nrm_idx = rng_p.choice(nrm_idx, 2500, replace=False)
    p_idx = np.concatenate([p_atk_idx, p_nrm_idx])
    X_proto = X_train[p_idx]
    y_proto = y_train[p_idx]

    proto_emb = embed_batch(X_proto, qsnn_opt_params)
    proto_attack = proto_emb[y_proto == 1].mean(axis=0)
    proto_normal = proto_emb[y_proto == 0].mean(axis=0)

    inter_dist = float(np.linalg.norm(proto_attack - proto_normal))
    intra_atk  = float(np.mean(np.linalg.norm(proto_emb[y_proto == 1] - proto_attack, axis=1)))
    intra_nrm  = float(np.mean(np.linalg.norm(proto_emb[y_proto == 0] - proto_normal, axis=1)))
    mean_intra = (intra_atk + intra_nrm) / 2.0
    sep_ratio_mean = inter_dist / (mean_intra + 1e-9)
    sep_ratio_max  = inter_dist / (max(intra_atk, intra_nrm) + 1e-9)

    print(f"\n[QSNN Embedding Diagnostics]")
    print(f"  Inter-prototype distance          : {inter_dist:.4f}  (Baseline: 0.7712)")
    print(f"  Intra-class dist (attack)         : {intra_atk:.4f}  (Baseline: 0.0889)")
    print(f"  Intra-class dist (normal)         : {intra_nrm:.4f}  (Baseline: 0.3666)")
    print(f"  Separation ratio (inter / mean)   : {sep_ratio_mean:.4f}  (Baseline: 3.3857)")
    print(f"  Separation ratio (inter / max)    : {sep_ratio_max:.4f}  (Baseline: 2.1035)")

    # Full Test Set Inference
    print(f"\n  Evaluating QSNN on full test set ({len(X_test):,} samples)...")
    t_inf = time.time()
    test_emb = embed_batch(X_test, qsnn_opt_params, batch_size=2000)
    print(f"  Inference done in {time.time()-t_inf:.1f}s")

    d_attack = np.linalg.norm(test_emb - proto_attack, axis=1)
    d_normal = np.linalg.norm(test_emb - proto_normal, axis=1)

    neg_dists = np.stack([-d_attack, -d_normal], axis=1)
    exp_d = np.exp(neg_dists - neg_dists.max(axis=1, keepdims=True))
    probs_mat = exp_d / exp_d.sum(axis=1, keepdims=True)
    qsnn_probs = probs_mat[:, 0]
    qsnn_preds = (qsnn_probs >= 0.5).astype(int)

    prob_min, prob_max = float(qsnn_probs.min()), float(qsnn_probs.max())
    prob_mean, prob_std = float(qsnn_probs.mean()), float(qsnn_probs.std())

    acc  = accuracy_score(y_test, qsnn_preds)
    prec = precision_score(y_test, qsnn_preds, zero_division=0)
    rec  = recall_score(y_test, qsnn_preds, zero_division=0)
    f1   = f1_score(y_test, qsnn_preds, zero_division=0)
    cm   = confusion_matrix(y_test, qsnn_preds)

    print(f"\n[QSNN Full Test Results (t=0.50)]")
    print(f"  Accuracy  : {acc:.4f}  (Baseline: 0.7659)")
    print(f"  Precision : {prec:.4f}  (Baseline: 0.7023)")
    print(f"  Recall    : {rec:.4f}  (Baseline: 0.9976)")
    print(f"  F1 Score  : {f1:.4f}  (Baseline: 0.8243)")
    print(f"  Prob min / max : [{prob_min:.4f}, {prob_max:.4f}]  (Baseline: [0.3163, 0.6838])")
    print(f"  Confusion Matrix:\n{cm}")

    # Threshold search
    best_t, best_f1 = 0.50, f1
    for t in np.arange(0.20, 0.81, 0.02):
        score = f1_score(y_test, (qsnn_probs >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, round(float(t), 2)
    print(f"  Best threshold: t={best_t:.2f} -> F1={best_f1:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # B) Train QMLP with Re-uploading for comparison
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Training QMLP with Data Re-uploading (Comparison) ---")
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
    init_w   = np.random.uniform(-0.1,   0.1,   N_QUBITS)
    init_b   = np.array([0.0])
    qmlp_params = pnp.array(
        np.concatenate([init_vqc, init_w, init_b]), requires_grad=True
    )
    opt_qmlp = qml.AdamOptimizer(stepsize=ADAM_LR)

    t0 = time.time()
    for step in range(1, ADAM_STEPS + 1):
        qmlp_params, l_val = opt_qmlp.step_and_cost(
            lambda p: qmlp_loss_fn(p, X_sub, y_sub.astype(float)), qmlp_params
        )
        if step % 20 == 0 or step == 1:
            print(f"  Step {step:3d}/{ADAM_STEPS} | loss = {float(l_val):.4f}")
    print(f"  QMLP training finished in {(time.time()-t0)/60:.2f} min")

    qmlp_opt = np.array(qmlp_params)
    opt_vqc = qmlp_opt[:TOTAL_VQC_PARAMS]
    opt_w   = qmlp_opt[TOTAL_VQC_PARAMS:TOTAL_VQC_PARAMS + N_QUBITS]
    opt_b   = qmlp_opt[-1]

    qmlp_test_emb = embed_batch(X_te, opt_vqc)
    logits = qmlp_test_emb @ opt_w + opt_b
    qmlp_probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    qmlp_preds = (qmlp_probs >= 0.5).astype(int)

    qmlp_acc  = accuracy_score(y_te, qmlp_preds)
    qmlp_prec = precision_score(y_te, qmlp_preds, zero_division=0)
    qmlp_rec  = recall_score(y_te, qmlp_preds, zero_division=0)
    qmlp_f1   = f1_score(y_te, qmlp_preds, zero_division=0)

    best_qmlp_t, best_qmlp_f1 = 0.50, qmlp_f1
    for t in np.arange(0.20, 0.81, 0.02):
        score = f1_score(y_te, (qmlp_probs >= t).astype(int), zero_division=0)
        if score > best_qmlp_f1:
            best_qmlp_f1, best_qmlp_t = score, round(float(t), 2)

    print(f"\n[QMLP Subsample Results (1000 samples)]")
    print(f"  t=0.50 : Acc={qmlp_acc:.4f}  Prec={qmlp_prec:.4f}  Rec={qmlp_rec:.4f}  F1={qmlp_f1:.4f}  (Baseline: 0.7984)")
    print(f"  Best t={best_qmlp_t:.2f} : F1={best_qmlp_f1:.4f}  (Baseline tuned: 0.8037)")
    print(f"  Prob min / max: [{qmlp_probs.min():.4f}, {qmlp_probs.max():.4f}]")

    # Save checkpoint results
    results = {
        "exp": "Exp 1: Data Re-uploading",
        "qsnn": {
            "params": qsnn_opt_params,
            "proto_attack": proto_attack,
            "proto_normal": proto_normal,
            "inter_dist": inter_dist,
            "intra_atk": intra_atk,
            "intra_nrm": intra_nrm,
            "sep_ratio_mean": sep_ratio_mean,
            "sep_ratio_max": sep_ratio_max,
            "prob_min": prob_min,
            "prob_max": prob_max,
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "best_t": best_t,
            "best_f1": best_f1,
            "probs": qsnn_probs,
            "preds": qsnn_preds,
        },
        "qmlp": {
            "acc": qmlp_acc,
            "prec": qmlp_prec,
            "rec": qmlp_rec,
            "f1": qmlp_f1,
            "best_t": best_qmlp_t,
            "best_f1": best_qmlp_f1,
            "prob_min": float(qmlp_probs.min()),
            "prob_max": float(qmlp_probs.max()),
        }
    }
    joblib.dump(results, "artifacts/sprint_exp1_results.pkl")
    print("\nSaved checkpoint -> artifacts/sprint_exp1_results.pkl")
    return results


if __name__ == "__main__":
    run_experiment_1()
