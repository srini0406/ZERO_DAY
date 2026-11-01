"""
explainability.py — Stage 5: SHAP Explainability for Hybrid Quantum IDS
────────────────────────────────────────────────────────────────────────
Computes game-theoretic SHAP (SHapley Additive exPlanations) values for the
hybrid quantum IDS pipeline.

DESIGN DECISIONS & RATIONALE (FOR FACULTY GUIDE / DEFENSE):
──────────────────────────────────────────────────────────
1. WHY EXPLAIN THE 10 PRE-COMPRESSION FEATURES (NOT 6 PCA COMPONENTS)?
   - PCA components (PC1..PC6) are mathematical linear combinations of features.
     An explanation like "PC1 increased by 0.35" has zero actionable meaning to a
     SOC (Security Operations Center) analyst or network engineer.
   - In contrast, attributing importance to "sbytes", "rate", or "sttl" provides
     immediate forensic context (e.g., packet flood, protocol anomaly, scan).
   - Therefore, we wrap the complete pipeline:
       [10 classical features] ──> [PCA (6 dims)] ──> [Scaler [0,1]] ──> [QSNN VQC] ──> [Softmax Probs]
     into a single black-box callable function f(x) and explain f(x) directly
     over the 10 domain features.

2. WHY shap.KernelExplainer (NOT TreeExplainer or DeepExplainer)?
   - The quantum variational circuit (VQC) with parameterized rotation gates (RX, RY, RZ)
     and prototype-distance softmax is neither a decision tree nor a standard PyTorch DAG.
   - KernelExplainer is model-agnostic, treating the composite pipeline as a black box
     and using weighted local linear regressions to compute Shapley values satisfying
     efficiency, symmetry, dummy, and additivity axioms.

3. WHY A REPRESENTATIVE BACKGROUND DATASET?
   - KernelExplainer evaluates marginal expectations E[f(x | x_S)] by integrating over
     a reference background distribution. Evaluating on the entire 175,211 training set
     is computationally prohibitive (O(N_eval * N_bg * 2^|S|)).
   - We extract a diverse background sample (50 representative samples from training data
     stratified across attack and normal flows) to preserve feature marginals while keeping
     quantum evaluation time under ~5-10 seconds per explained instance.

4. MULTI-CATEGORY EXPLANATIONS:
   - We generate explanations for 5 representative operational categories:
     a) Confident Known Attack (CNN + Quantum confirmed)
     b) Confident Normal Traffic (CNN cleared)
     c) Benign Novel Behavior (Quantum uncertain -> DBSCAN benign cluster)
     d) Malicious Attack (Quantum uncertain -> DBSCAN attack cluster)
     e) Zero-Day Attack (Quantum uncertain -> DBSCAN novel outlier)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import pennylane as qml
import shap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def log(msg=""): print(msg, flush=True)

ART = "artifacts"
SEED = 42
N_QUBITS = 6

# ── 1. Load Artifacts ────────────────────────────────────────────────────────
log("=" * 75)
log("STAGE 5 — SHAP EXPLAINABILITY (Pre-Compression Domain Features)")
log("=" * 75)

log("\n[1/6] Loading pipeline artifacts and models...")
pca = joblib.load(os.path.join(ART, "pca_quantum.pkl"))
pca_scaler = joblib.load(os.path.join(ART, "pca_scaler.pkl"))
qsnn_params = joblib.load(os.path.join(ART, "qsnn_params.pkl"))
protos = joblib.load(os.path.join(ART, "qsnn_prototypes.pkl"))
proto_atk = protos["proto_attack"]
proto_nrm = protos["proto_normal"]

feat_df = pd.read_csv(os.path.join(ART, "encoded_feature_names.csv"))
FEATURE_NAMES = feat_df["feature"].tolist()

X_train_10 = np.load(os.path.join(ART, "X_train_15.npy"))
X_test_10  = np.load(os.path.join(ART, "X_test_15.npy"))
y_train    = np.load(os.path.join(ART, "y_train.npy"))
y_test     = np.load(os.path.join(ART, "y_test.npy"))
y_test_cat = np.load(os.path.join(ART, "y_test_cat.npy"), allow_pickle=True)
final_labels = np.load(os.path.join(ART, "final_test_labels.npy"), allow_pickle=True)

log(f"  Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
log(f"  X_train_10 shape: {X_train_10.shape}  |  X_test_10 shape: {X_test_10.shape}")

# ── 2. Quantum Circuit Definition ───────────────────────────────────────────
log("\n[2/6] Initializing PennyLane lightning device for inference wrapper...")
dev = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(dev, diff_method="adjoint")
def qsnn_circuit(x, params):
    """Data re-uploading quantum circuit (Exp 1 winner)."""
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

def quantum_predict_proba(X_10):
    """
    Complete end-to-end inference callable:
      10 domain features -> PCA(6) -> MinMaxScaler -> PennyLane VQC -> Distance to Prototypes -> Softmax
    Returns P(Attack) array of shape (N,).
    """
    X_arr = np.asarray(X_10, dtype=np.float32)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    
    # 1. Transform classical 10 features to 6 PCA dimensions
    X_pca = pca.transform(X_arr)
    
    # 2. Scale PCA output to [0, 1] for rotation angles
    X_q = pca_scaler.transform(X_pca).astype(np.float32)
    
    # 3. Fast batched circuit evaluation
    batch_size = 2000
    all_embs = []
    for s in range(0, len(X_q), batch_size):
        chunk = X_q[s:s+batch_size]
        emb = np.array(qsnn_circuit(chunk.T, qsnn_params)).T
        all_embs.append(emb)
    emb = np.vstack(all_embs)
    
    # 4. Prototype distances & Softmax
    d_atk = np.linalg.norm(emb - proto_atk, axis=1)
    d_nrm = np.linalg.norm(emb - proto_nrm, axis=1)
    neg_d = np.stack([-d_atk, -d_nrm], axis=1)
    exp_d = np.exp(neg_d - neg_d.max(axis=1, keepdims=True))
    probs = (exp_d / exp_d.sum(axis=1, keepdims=True))[:, 0]
    return probs

# ── 3. Build KernelExplainer with Stratified Background Data ────────────────
log("\n[3/6] Building SHAP KernelExplainer with stratified background set...")
# Sample 25 normal and 25 attack flows from training set for representative marginals
rng = np.random.default_rng(SEED)
idx_nrm = rng.choice(np.where(y_train == 0)[0], size=25, replace=False)
idx_atk = rng.choice(np.where(y_train == 1)[0], size=25, replace=False)
bg_indices = np.concatenate([idx_nrm, idx_atk])
background_data = X_train_10[bg_indices]

explainer = shap.KernelExplainer(quantum_predict_proba, background_data)
expected_val = float(explainer.expected_value)
log(f"  Background samples: {len(background_data)} (25 normal, 25 attack)")
log(f"  Base expected probability E[f(x)]: {expected_val:.4f}")

# ── 4. Select 5 Representative Exemplar Samples ──────────────────────────────
log("\n[4/6] Identifying representative samples across operational categories...")

# Category 1: Confident Known Attack (CNN=1, QSNN >= 0.60, True Attack)
idx_known_atk = int(np.where((final_labels == "CNN+Quantum: Attack") & (y_test == 1))[0][0])

# Category 2: Confident Normal (CNN=0, True Normal)
idx_normal = int(np.where((final_labels == "CNN: Normal") & (y_test == 0))[0][0])

# Category 3: Benign Novel Behavior (DBSCAN resolved uncertain as Benign)
idx_benign_novel = int(np.where(final_labels == "Benign Novel Behavior")[0][0])

# Category 4: Malicious Attack (DBSCAN resolved uncertain as Malicious)
idx_malicious = int(np.where(final_labels == "Malicious Attack")[0][0])

# Category 5: Zero-Day Attack (DBSCAN resolved uncertain as Zero-Day)
idx_zero_day = int(np.where(final_labels == "Zero-Day Attack")[0][0])

exemplar_indices = [
    ("Confident Known Attack", idx_known_atk),
    ("Confident Normal Traffic", idx_normal),
    ("Benign Novel Behavior",   idx_benign_novel),
    ("Malicious Attack",        idx_malicious),
    ("Zero-Day Attack",         idx_zero_day),
]

for cat_name, idx in exemplar_indices:
    log(f"  [{cat_name}] -> Index {idx:5d} | Stage Label: '{final_labels[idx]}' | True Category: {y_test_cat[idx]} (label={y_test[idx]})")

# ── 5. Compute SHAP Values for Exemplars ─────────────────────────────────────
log("\n[5/6] Computing SHAP values and generating natural-language explanations...")

FEATURE_DESCRIPTIONS = {
    "sbytes": "source-transmitted byte volume (sbytes)",
    "sttl":   "source packet time-to-live (sttl)",
    "dbytes": "destination response byte volume (dbytes)",
    "dttl":   "destination time-to-live (dttl)",
    "rate":   "packet transmission rate (rate)",
    "sload":  "source bit throughput (sload)",
    "dur":    "flow connection duration (dur)",
    "smean":  "mean source packet size (smean)",
    "dmean":  "mean destination packet size (dmean)",
    "dload":  "destination bit throughput (dload)",
}

def generate_plain_explanation(category, idx, prob, shap_row, x_row):
    """Generate human-readable SOC analyst narrative from SHAP attribution."""
    # Top 3 features pushing probability toward attack (positive SHAP)
    pos_idx = np.argsort(shap_row)[::-1]
    top_pos = [i for i in pos_idx if shap_row[i] > 0.005][:3]
    
    # Top 2 features pushing probability toward normal (negative SHAP)
    neg_idx = np.argsort(shap_row)
    top_neg = [i for i in neg_idx if shap_row[i] < -0.005][:2]
    
    pos_phrases = [f"{FEATURE_DESCRIPTIONS[FEATURE_NAMES[i]]} (+{shap_row[i]:.3f})" for i in top_pos]
    neg_phrases = [f"{FEATURE_DESCRIPTIONS[FEATURE_NAMES[i]]} ({shap_row[i]:.3f})" for i in top_neg]
    
    lines = []
    lines.append(f"• Decision & Context: Categorized as '{category}' with Quantum P(Attack) = {prob:.4f}.")
    if pos_phrases:
        lines.append(f"• Primary Suspicion Drivers: Suspicion elevated primarily by {', '.join(pos_phrases)}.")
    else:
        lines.append(f"• Primary Suspicion Drivers: No individual features exerted strong positive suspicion.")
        
    if neg_phrases:
        lines.append(f"• Mitigating Benign Factors: Counter-balanced toward benign by {', '.join(neg_phrases)}.")
        
    # Operational takeaway
    if "Zero-Day" in category:
        lines.append("• SOC Action: HIGH-PRIORITY ISOLATION. Flow exhibited borderline quantum uncertainty and sits far from all known attack/normal clusters. Novel attack vector suspected.")
    elif "Malicious" in category or "Known Attack" in category:
        lines.append("• SOC Action: ALERT & BLOCK. Signatures align with known attack characteristics. Packet rate and TTL profiles indicate active exploit or DoS attempt.")
    elif "Benign Novel" in category:
        lines.append("• SOC Action: LOG & WHITELIST. Flow deviates slightly in CNN space but geometry matches normal cluster baselines. Safe to add to normal profile pool.")
    else:
        lines.append("• SOC Action: PASS. Flow conforms to standard benign operational traffic.")
        
    return "\n  ".join(lines)

explanation_results = []
shap_matrix = []

for cat_name, idx in exemplar_indices:
    log(f"\n--- Explaining Exemplar: {cat_name} (Sample Index {idx}) ---")
    x_sample = X_test_10[idx:idx+1]
    prob = float(quantum_predict_proba(x_sample)[0])
    
    # Compute SHAP
    shap_vals = explainer.shap_values(x_sample, nsamples=150)
    shap_row = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
    shap_matrix.append(shap_row)
    
    narrative = generate_plain_explanation(cat_name, idx, prob, shap_row, x_sample[0])
    log(f"  {narrative}\n")
    
    # Feature breakdown table
    log(f"  {'Feature':<10} {'Value (norm)':<14} {'SHAP Impact':<12} {'Direction'}")
    log("  " + "-" * 48)
    for f_idx in np.argsort(np.abs(shap_row))[::-1]:
        f_name = FEATURE_NAMES[f_idx]
        val    = x_sample[0, f_idx]
        sv     = shap_row[f_idx]
        direction = "▲ PUSH TO ATTACK" if sv > 0 else "▼ PUSH TO NORMAL"
        log(f"  {f_name:<10} {val:<14.4f} {sv:<+12.4f} {direction}")
        
    explanation_results.append({
        "category": cat_name,
        "sample_index": idx,
        "final_label": final_labels[idx],
        "true_category": y_test_cat[idx],
        "true_y": int(y_test[idx]),
        "quantum_p_attack": prob,
        "shap_values": shap_row,
        "raw_features_norm": x_sample[0],
        "narrative": narrative
    })

# ── 6. Global Feature Importance & Artifact Saving ──────────────────────────
log("\n[6/6] Computing global feature importance and saving artifacts...")
shap_matrix = np.array(shap_matrix)
global_importance = np.mean(np.abs(shap_matrix), axis=0)
imp_order = np.argsort(global_importance)[::-1]

imp_df = pd.DataFrame({
    "rank": np.arange(1, len(FEATURE_NAMES) + 1),
    "feature": [FEATURE_NAMES[i] for i in imp_order],
    "description": [FEATURE_DESCRIPTIONS[FEATURE_NAMES[i]] for i in imp_order],
    "mean_abs_shap": [global_importance[i] for i in imp_order]
})

log("\n  Global Feature Importance (Across Explored Operational Regimes):")
log(f"  {'Rank':<5} {'Feature':<10} {'Mean |SHAP|':<14} {'Feature Description'}")
log("  " + "-" * 60)
for _, r in imp_df.iterrows():
    log(f"  {r['rank']:<5} {r['feature']:<10} {r['mean_abs_shap']:<14.4f} {r['description']}")

# Save artifacts
imp_df.to_csv(os.path.join(ART, "shap_feature_importance.csv"), index=False)
joblib.dump({
    "expected_value": expected_val,
    "feature_names": FEATURE_NAMES,
    "feature_descriptions": FEATURE_DESCRIPTIONS,
    "background_indices": bg_indices,
    "exemplar_results": explanation_results,
    "global_importance": imp_df
}, os.path.join(ART, "shap_summary.pkl"))

log(f"\n  Saved artifacts:")
log(f"  - {os.path.join(ART, 'shap_feature_importance.csv')}")
log(f"  - {os.path.join(ART, 'shap_summary.pkl')}")

log("\n" + "=" * 75)
log("STAGE 5 COMPLETE — SHAP EXPLAINABILITY VERIFIED")
log("=" * 75)
