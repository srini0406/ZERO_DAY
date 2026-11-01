"""
synthetic_routing_test.py
SYNTHETIC boundary stress test - tests DBSCAN routing only; not real zero-day detection.
──────────────────────────────────────────────────────────────────────────────────────────
This script implements a rigorous synthetic boundary stress test for the UNSW-NB15
Hybrid DL + QML + DBSCAN architecture. It specifically tests the DBSCAN routing mechanism
on flows designed to land in the quantum uncertain zone [0.45, 0.60).

FROZEN ARTIFACTS ONLY:
- Deployed CNN threshold loaded from artifacts/cnn_threshold.pkl
- CNN weights loaded from artifacts/cnn_model.pt
- Preprocessing scalers/encoders loaded from artifacts/scaler_full.pkl, ohe_full.pkl, scaler_15.pkl
- Quantum PCA and scaler loaded from artifacts/pca_quantum.pkl, pca_scaler.pkl
- QSNN parameters and prototypes loaded from artifacts/qsnn_params.pkl, qsnn_prototypes.pkl
- DBSCAN scaler, PCA, centroids, and config loaded from artifacts/
NO THRESHOLDS ARE RECOMPUTED OR TUNED ON SYNTHETIC FLOWS.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
import pennylane as qml
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure reproducible stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ART = "artifacts"
MANDATORY_DISCLAIMER = "SYNTHETIC boundary stress test - tests DBSCAN routing only; not real zero-day detection."

print("=" * 88)
print(f"HYBRID IDS — {MANDATORY_DISCLAIMER}")
print("=" * 88)

# ── 1. Load Frozen Artifacts ──────────────────────────────────────────────────
print("\n[1/6] Loading Frozen Model Artifacts & Thresholds...")

cnn_threshold = float(joblib.load(os.path.join(ART, "cnn_threshold.pkl")))
print(f"  [FROZEN ARTIFACT] Loaded Deployed CNN Threshold: {cnn_threshold:.4f}")

db_cfg = joblib.load(os.path.join(ART, "dbscan_config.pkl"))
benign_thresh = float(db_cfg["benign_thresh"])
attack_thresh = float(db_cfg["attack_thresh"])
print(f"  [FROZEN ARTIFACT] Loaded DBSCAN Benign Distance Threshold: {benign_thresh:.4f}")
print(f"  [FROZEN ARTIFACT] Loaded DBSCAN Attack Distance Threshold: {attack_thresh:.4f}")

# Feature names and transformers
full_feature_names = pd.read_csv(os.path.join(ART, "full_feature_names.csv"))["feature"].tolist()
NUMERIC_FEATURES = full_feature_names[:39]
CATEGORICAL_FEATURES = ["proto", "service", "state"]

scaler_full = joblib.load(os.path.join(ART, "scaler_full.pkl"))
ohe_full    = joblib.load(os.path.join(ART, "ohe_full.pkl"))

scaler_15  = joblib.load(os.path.join(ART, "scaler_15.pkl"))
enc_10_names = pd.read_csv(os.path.join(ART, "encoded_feature_names.csv"))["feature"].tolist()
pca_q      = joblib.load(os.path.join(ART, "pca_quantum.pkl"))
pca_scaler = joblib.load(os.path.join(ART, "pca_scaler.pkl"))

# CNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from cnn_model import CNN1D
cnn_model = CNN1D(n_features=57).to(device)
cnn_model.load_state_dict(torch.load(os.path.join(ART, "cnn_model.pt"), map_location=device))
cnn_model.eval()
print(f"  [FROZEN ARTIFACT] 1D CNN loaded on device: {device}")

# Quantum circuit setup
N_QUBITS = 6
qsnn_params = joblib.load(os.path.join(ART, "qsnn_params.pkl"))
protos = joblib.load(os.path.join(ART, "qsnn_prototypes.pkl"))
proto_atk = protos["proto_attack"]
proto_nrm = protos["proto_normal"]

dev_q = qml.device("lightning.qubit", wires=N_QUBITS)
@qml.qnode(dev_q, diff_method="adjoint")
def qsnn_circuit(x, params):
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

# DBSCAN Artifacts
db_scaler = joblib.load(os.path.join(ART, "dbscan_scaler.pkl"))
db_pca    = joblib.load(os.path.join(ART, "dbscan_pca.pkl"))
db_nrm_centroids = joblib.load(os.path.join(ART, "dbscan_normal_centroids.pkl"))
db_atk_centroids = joblib.load(os.path.join(ART, "dbscan_attack_centroids.pkl"))
print(f"  [FROZEN ARTIFACT] DBSCAN Centroids: {len(db_nrm_centroids)} normal, {len(db_atk_centroids)} attack")

# ── 2. Load Raw Source & Partner Data ─────────────────────────────────────────
print("\n[2/6] Loading Datasets for Source Cohorts and Normal Partners...")

train_df = pd.read_csv("UNSW_NB15_training-set.csv")
train_df["attack_cat"] = train_df["attack_cat"].str.strip().str.title()
train_normals = train_df[train_df["label"] == 0].reset_index(drop=True)
print(f"  Available Training Normal Partners: {len(train_normals):,}")

test_df = pd.read_csv("UNSW_NB15_testing-set.csv")
test_df["attack_cat"] = test_df["attack_cat"].str.strip().str.title()

# Worms cohort: 130 held-out + 44 test Worms = 174 flows
heldout_worms = pd.read_csv(os.path.join(ART, "zero_day_worms.csv"))
heldout_worms["attack_cat"] = "Worms"
test_worms = test_df[test_df["attack_cat"] == "Worms"].copy()
worms_cohort = pd.concat([heldout_worms, test_worms], ignore_index=True)
print(f"  Worms Source Flows: {len(worms_cohort):,} (130 held-out + 44 test)")

# Seen attack cohorts from test set
exploits_cohort = test_df[test_df["attack_cat"] == "Exploits"].reset_index(drop=True)
dos_cohort      = test_df[test_df["attack_cat"] == "Dos"].reset_index(drop=True)
generic_cohort  = test_df[test_df["attack_cat"] == "Generic"].reset_index(drop=True)
print(f"  Exploits Source Flows : {len(exploits_cohort):,}")
print(f"  DoS Source Flows      : {len(dos_cohort):,}")
print(f"  Generic Source Flows  : {len(generic_cohort):,}")

# Reference Group: 38 real flows routed to DBSCAN
unc_indices = np.load(os.path.join(ART, "dbscan_uncertain_indices.npy"))
db_cats_38  = np.load(os.path.join(ART, "dbscan_categories.npy"), allow_pickle=True)
d_n_38      = np.load(os.path.join(ART, "dbscan_d_normals.npy"))
d_a_38      = np.load(os.path.join(ART, "dbscan_d_attacks.npy"))
cnn_probs_full = np.load(os.path.join(ART, "cnn_test_probs.npy"))
qsnn_probs_full = np.load(os.path.join(ART, "qsnn_test_probs.npy"))

ref_p_cnn = cnn_probs_full[unc_indices]
ref_p_q   = qsnn_probs_full[unc_indices]
print(f"  Reference Group: {len(unc_indices)} Real Flows in DBSCAN Stage")

# Compute training min and max for sanity checks
train_mins = train_df[NUMERIC_FEATURES].min().values
train_maxs = train_df[NUMERIC_FEATURES].max().values

# ── 3. Helper Functions: Pipeline Transformers & Metrics ───────────────────────

def transform_cnn(syn_numeric, src_categoricals):
    """Transform synthetic numeric + copied categoricals to 57-dim CNN input."""
    syn_scaled = scaler_full.transform(syn_numeric)
    cat_parts = []
    n_samples = len(syn_numeric)
    for col in CATEGORICAL_FEATURES:
        series = pd.Series([src_categoricals[col]] * n_samples).astype(str)
        keep = ohe_full[col]["keep_cats"]
        series = series.where(series.isin(keep), other="other")
        for cat in ohe_full[col]["all_cats"]:
            cat_parts.append((series == cat).astype(float).values)
    cat_mat = np.column_stack(cat_parts)
    return np.hstack([syn_scaled, cat_mat]).astype(np.float32)

enc_10_indices = [NUMERIC_FEATURES.index(f) for f in enc_10_names]

def transform_qsnn(syn_numeric):
    """Transform synthetic numeric to 6-dim quantum input."""
    raw_10 = syn_numeric[:, enc_10_indices]
    scaled_10 = scaler_15.transform(raw_10)
    pca_comp = pca_q.transform(scaled_10)
    return pca_scaler.transform(pca_comp).astype(np.float32)

def evaluate_qsnn_probs(q_inputs):
    """Compute P_q(Attack) from QSNN quantum circuit embeddings."""
    embs = np.array(qsnn_circuit(q_inputs.T, qsnn_params)).T
    d_atk = np.linalg.norm(embs - proto_atk, axis=1)
    d_nrm = np.linalg.norm(embs - proto_nrm, axis=1)
    neg_d = np.stack([-d_atk, -d_nrm], axis=1)
    exp_d = np.exp(neg_d - neg_d.max(axis=1, keepdims=True))
    probs_mat = exp_d / exp_d.sum(axis=1, keepdims=True)
    return probs_mat[:, 0]

def classify_dbscan_points(cnn_embeddings):
    """Classify 64-dim CNN embeddings using frozen DBSCAN PCA, centroids, and thresholds."""
    emb_sc = db_scaler.transform(cnn_embeddings)
    emb_pca = db_pca.transform(emb_sc)
    
    d_normals = np.array([float(min(np.linalg.norm(x - c) for c in db_nrm_centroids)) for x in emb_pca])
    d_attacks = np.array([float(min(np.linalg.norm(x - c) for c in db_atk_centroids)) for x in emb_pca])
    
    labels = []
    for dn, da in zip(d_normals, d_attacks):
        close_n = (dn <= benign_thresh)
        close_a = (da <= attack_thresh)
        if close_n and not close_a:
            labels.append("Benign Novel Behavior")
        elif close_a and not close_n:
            labels.append("Malicious Attack")
        elif close_n and close_a:
            labels.append("Benign Novel Behavior" if dn < da else "Malicious Attack")
        else:
            labels.append("Zero-Day Attack")
    return np.array(labels), d_normals, d_attacks

def wilson_ci(k, n, z=1.95996):
    """Wilson score confidence interval for proportions."""
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p)) / n + (z**2) / (4 * (n**2)))) / denom
    return (max(0.0, center - margin) * 100, min(1.0, center + margin) * 100)

def prop_diff_ci(p1, n1, p2, n2, z=1.95996):
    """Confidence interval for the difference between two independent proportions."""
    diff = p1 - p2
    se = np.sqrt((p1 * (1.0 - p1) / n1) + (p2 * (1.0 - p2) / n2))
    return (diff * 100, (diff - z * se) * 100, (diff + z * se) * 100)

# ── 4. Execution Loop for a Given Seed ────────────────────────────────────────

ALPHAS = np.arange(0.02, 0.99, 0.02)  # 49 grid steps: 0.02 to 0.98
TARGET_N = 200
MAX_ATTEMPTS = 20000

cohorts_map = {
    "Worms": worms_cohort,
    "Exploits": exploits_cohort,
    "DoS": dos_cohort,
    "Generic": generic_cohort
}

def run_stress_test_for_seed(seed):
    print(f"\n{'='*75}")
    print(f"RUNNING SYNTHETIC STRESS TEST WITH SEED = {seed}")
    print(f"{'='*75}")
    rng = np.random.RandomState(seed)
    
    accepted_records = []
    group_stats = {}
    sanity_out_of_bounds = []
    
    for group_name, cohort in cohorts_map.items():
        t0 = time.time()
        accepted_group = []
        attempts = 0
        all_alpha_pq = []  # For alpha vs pq plotting
        
        while len(accepted_group) < TARGET_N and attempts < MAX_ATTEMPTS:
            # Sample random source flow and random training normal partner
            s_idx = rng.randint(len(cohort))
            n_idx = rng.randint(len(train_normals))
            src_row = cohort.iloc[s_idx]
            nrm_row = train_normals.iloc[n_idx]
            
            src_num = src_row[NUMERIC_FEATURES].values.astype(float)
            nrm_num = nrm_row[NUMERIC_FEATURES].values.astype(float)
            src_cats = {c: src_row[c] for c in CATEGORICAL_FEATURES}
            
            # Linear interpolation across all 49 alpha values
            syn_nums = np.outer(ALPHAS, src_num) + np.outer(1.0 - ALPHAS, nrm_num)
            
            # Sanity check: min/max check against training set
            oob = ((syn_nums < train_mins - 1e-6) | (syn_nums > train_maxs + 1e-6)).any(axis=1)
            sanity_out_of_bounds.extend(oob.tolist())
            
            # 1. 57-dim CNN forward pass
            x_cnn = transform_cnn(syn_nums, src_cats)
            with torch.no_grad():
                t_cnn = torch.tensor(x_cnn, device=device)
                p_cnn_batch = cnn_model(t_cnn)[:, 1].cpu().numpy()
            
            # Filter condition 1: P_cnn >= deployed threshold
            cnn_mask = (p_cnn_batch >= cnn_threshold)
            attempts += len(ALPHAS)
            
            if cnn_mask.any():
                idx_pass = np.where(cnn_mask)[0]
                q_in = transform_qsnn(syn_nums[idx_pass])
                p_q_pass = evaluate_qsnn_probs(q_in)
                
                for a_val, pq_val in zip(ALPHAS[idx_pass], p_q_pass):
                    all_alpha_pq.append((a_val, pq_val))
                
                # Filter condition 2: 0.45 <= P_q < 0.60
                q_mask = (p_q_pass >= 0.45) & (p_q_pass < 0.60)
                if q_mask.any():
                    idx_acc = idx_pass[q_mask]
                    acc_nums = syn_nums[idx_acc]
                    acc_p_cnn = p_cnn_batch[idx_acc]
                    acc_p_q = p_q_pass[q_mask]
                    acc_alphas = ALPHAS[idx_acc]
                    
                    # Extract CNN embeddings for accepted flows
                    x_acc_cnn = transform_cnn(acc_nums, src_cats)
                    with torch.no_grad():
                        emb_batch = cnn_model.get_embedding(torch.tensor(x_acc_cnn, device=device)).cpu().numpy()
                    
                    labels, dn_vals, da_vals = classify_dbscan_points(emb_batch)
                    
                    for i in range(len(idx_acc)):
                        if len(accepted_group) < TARGET_N:
                            rec = {
                                "seed": seed,
                                "group": group_name,
                                "source_id": src_row.get("id", s_idx),
                                "partner_id": nrm_row.get("id", n_idx),
                                "alpha": float(acc_alphas[i]),
                                "p_cnn": float(acc_p_cnn[i]),
                                "p_q": float(acc_p_q[i]),
                                "d_n": float(dn_vals[i]),
                                "d_a": float(da_vals[i]),
                                "label": labels[i]
                            }
                            accepted_group.append(rec)
                            accepted_records.append(rec)
        
        acc_rate = (len(accepted_group) / attempts) * 100 if attempts > 0 else 0.0
        df_grp = pd.DataFrame(accepted_group)
        
        group_stats[group_name] = {
            "n_accepted": len(df_grp),
            "attempts": attempts,
            "acc_rate": acc_rate,
            "mean_alpha": df_grp["alpha"].mean() if len(df_grp) else 0.0,
            "mean_p_cnn": df_grp["p_cnn"].mean() if len(df_grp) else 0.0,
            "mean_p_q": df_grp["p_q"].mean() if len(df_grp) else 0.0,
            "median_d_n": df_grp["d_n"].median() if len(df_grp) else 0.0,
            "median_d_a": df_grp["d_a"].median() if len(df_grp) else 0.0,
            "n_benign": int((df_grp["label"] == "Benign Novel Behavior").sum()),
            "n_malicious": int((df_grp["label"] == "Malicious Attack").sum()),
            "n_zeroday": int((df_grp["label"] == "Zero-Day Attack").sum()),
            "df": df_grp,
            "all_alpha_pq": all_alpha_pq
        }
        print(f"  [{group_name}] Accepted: {len(df_grp)}/{TARGET_N} in {attempts:,} attempts "
              f"({acc_rate:.2f}% rate, {time.time()-t0:.1f}s) — "
              f"ZD: {group_stats[group_name]['n_zeroday']}/{len(df_grp)} "
              f"({group_stats[group_name]['n_zeroday']/len(df_grp)*100:.1f}%)")
        
    return accepted_records, group_stats, sanity_out_of_bounds

# ── 5. Run Across Seeds 42, 43, 44 ─────────────────────────────────────────────
all_seeds_records = []
all_seeds_stats = {}
all_sanity_oob = []

for s in [42, 43, 44]:
    recs, stats_s, oob_s = run_stress_test_for_seed(s)
    all_seeds_records.extend(recs)
    all_seeds_stats[s] = stats_s
    all_sanity_oob.extend(oob_s)

# Save Master synthetic_results.csv
results_df = pd.DataFrame(all_seeds_records)
results_df.to_csv("synthetic_results.csv", index=False)
print(f"\n[Artifact Saved] synthetic_results.csv ({len(results_df):,} total accepted flows)")

# ── 6. Sanity Checks ──────────────────────────────────────────────────────────
fraction_oob = (sum(all_sanity_oob) / len(all_sanity_oob)) * 100 if len(all_sanity_oob) else 0.0
print(f"\n[Sanity Check] Fraction of synthetic flows outside training min/max: {fraction_oob:.4f}% (n={len(all_sanity_oob):,})")

# ── 7. Plotting: alpha vs p_q for each group ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
fig.suptitle(f"{MANDATORY_DISCLAIMER}\nα vs P_q(Attack) Interpolation Profile Across Source Groups", 
             fontsize=13, fontweight='bold', y=0.98)

colors = {"Zero-Day Attack": "#d90429", "Malicious Attack": "#f77f00", "Benign Novel Behavior": "#0077b6"}
plot_data_seed42 = all_seeds_stats[42]

for ax, grp in zip(axes.flatten(), ["Worms", "Exploits", "DoS", "Generic"]):
    df_grp = plot_data_seed42[grp]["df"]
    
    # Highlight accepted zone [0.45, 0.60)
    ax.axhspan(0.45, 0.60, color="#2ec4b6", alpha=0.18, label="DBSCAN Uncertainty Zone [0.45, 0.60)")
    ax.axhline(0.45, color="#0f4c5c", linestyle="--", linewidth=1.2)
    ax.axhline(0.60, color="#0f4c5c", linestyle="--", linewidth=1.2)
    
    # Plot accepted flows by label
    for lbl, col in colors.items():
        subset = df_grp[df_grp["label"] == lbl]
        if len(subset) > 0:
            ax.scatter(subset["alpha"], subset["p_q"], color=col, alpha=0.75, s=36, edgecolors='none', label=f"{lbl} (n={len(subset)})")
            
    ax.set_title(f"Source Group: {grp} (Accepted n={len(df_grp)})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Interpolation Weight α (0 = Normal, 1 = Attack)", fontsize=10)
    ax.set_ylabel("Quantum Attack Probability P_q", fontsize=10)
    ax.set_ylim(0.42, 0.63)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = "synthetic_alpha_vs_pq.png"
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"[Plot Saved] {plot_path}")

# ── 8. Statistical Analysis & Key Comparison ──────────────────────────────────
def compute_comparisons(df_pool):
    worms_df = df_pool[df_pool["group"] == "Worms"]
    seen_df  = df_pool[df_pool["group"].isin(["Exploits", "DoS", "Generic"])]
    
    n_w = len(worms_df)
    zd_w = int((worms_df["label"] == "Zero-Day Attack").sum())
    p_w = zd_w / n_w if n_w else 0.0
    
    n_s = len(seen_df)
    zd_s = int((seen_df["label"] == "Zero-Day Attack").sum())
    p_s = zd_s / n_s if n_s else 0.0
    
    diff, diff_low, diff_high = prop_diff_ci(p_w, n_w, p_s, n_s)
    
    # Fisher exact test
    cont_table = [[zd_w, n_w - zd_w], [zd_s, n_s - zd_s]]
    odds_ratio, p_val = stats.fisher_exact(cont_table)
    
    # Stratified by p_q bins: [0.45, 0.50), [0.50, 0.55), [0.55, 0.60)
    bins = [(0.45, 0.50), (0.50, 0.55), (0.55, 0.60)]
    strat_results = []
    for b_lo, b_hi in bins:
        w_bin = worms_df[(worms_df["p_q"] >= b_lo) & (worms_df["p_q"] < b_hi)]
        s_bin = seen_df[(seen_df["p_q"] >= b_lo) & (seen_df["p_q"] < b_hi)]
        
        n_wb = len(w_bin)
        zd_wb = int((w_bin["label"] == "Zero-Day Attack").sum())
        p_wb = zd_wb / n_wb if n_wb else 0.0
        
        n_sb = len(s_bin)
        zd_sb = int((s_bin["label"] == "Zero-Day Attack").sum())
        p_sb = zd_sb / n_sb if n_sb else 0.0
        
        d_b, dl_b, dh_b = prop_diff_ci(p_wb, n_wb, p_sb, n_sb)
        res_f = stats.fisher_exact([[zd_wb, n_wb - zd_wb], [zd_sb, n_sb - zd_sb]]) if (n_wb > 0 and n_sb > 0) else (1.0, 1.0)
        
        strat_results.append({
            "bin": f"[{b_lo:.2f}, {b_hi:.2f})",
            "n_w": n_wb, "zd_w": zd_wb, "pct_w": p_wb * 100,
            "n_s": n_sb, "zd_s": zd_sb, "pct_s": p_sb * 100,
            "diff": d_b, "ci_low": dl_b, "ci_high": dh_b,
            "p_val": res_f[1], "or": res_f[0]
        })
        
    return {
        "n_w": n_w, "zd_w": zd_w, "pct_w": p_w * 100,
        "n_s": n_s, "zd_s": zd_s, "pct_s": p_s * 100,
        "diff": diff, "diff_low": diff_low, "diff_high": diff_high,
        "odds_ratio": odds_ratio, "p_val": p_val,
        "strat": strat_results
    }

# Compute comparisons for each seed and pooled
comp_by_seed = {}
for s in [42, 43, 44]:
    comp_by_seed[s] = compute_comparisons(results_df[results_df["seed"] == s])
comp_pooled = compute_comparisons(results_df)

# Print Summary Table
print("\n" + "=" * 90)
print(f"SUMMARY TABLE: {MANDATORY_DISCLAIMER}")
print("=" * 90)

header_fmt = "{:<12} {:>6} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>24} {:>24} {:>24}"
print(header_fmt.format("Group", "N", "Acc Rate", "Mean α", "Mean Pcnn", "Mean Pq", "Med dn", "Med da", 
                        "Benign Novel (95% CI)", "Malicious (95% CI)", "Zero-Day (95% CI)"))
print("-" * 155)

for grp in ["Worms", "Exploits", "DoS", "Generic"]:
    sub = results_df[results_df["group"] == grp]
    n = len(sub)
    # Average across the 3 seeds
    attempts_tot = sum(all_seeds_stats[s][grp]["attempts"] for s in [42, 43, 44])
    acc_rate = (n / attempts_tot) * 100
    mean_a = sub["alpha"].mean()
    mean_pcnn = sub["p_cnn"].mean()
    mean_pq = sub["p_q"].mean()
    med_dn = sub["d_n"].median()
    med_da = sub["d_a"].median()
    
    n_b = (sub["label"] == "Benign Novel Behavior").sum()
    n_m = (sub["label"] == "Malicious Attack").sum()
    n_z = (sub["label"] == "Zero-Day Attack").sum()
    
    ci_b = wilson_ci(n_b, n)
    ci_m = wilson_ci(n_m, n)
    ci_z = wilson_ci(n_z, n)
    
    str_b = f"{n_b/n*100:5.1f}% [{ci_b[0]:4.1f}, {ci_b[1]:4.1f}]"
    str_m = f"{n_m/n*100:5.1f}% [{ci_m[0]:4.1f}, {ci_m[1]:4.1f}]"
    str_z = f"{n_z/n*100:5.1f}% [{ci_z[0]:4.1f}, {ci_z[1]:4.1f}]"
    
    print(header_fmt.format(f"{grp} (All)", n, f"{acc_rate:6.2f}%", f"{mean_a:6.3f}", f"{mean_pcnn:6.3f}", 
                            f"{mean_pq:6.3f}", f"{med_dn:6.2f}", f"{med_da:6.2f}", str_b, str_m, str_z))

# Reference Group (Real 38 Flows)
n_ref = len(unc_indices)
n_b_ref = (db_cats_38 == "Benign Novel Behavior").sum()
n_m_ref = (db_cats_38 == "Malicious Attack").sum()
n_z_ref = (db_cats_38 == "Zero-Day Attack").sum()
ci_b_ref = wilson_ci(n_b_ref, n_ref)
ci_m_ref = wilson_ci(n_m_ref, n_ref)
ci_z_ref = wilson_ci(n_z_ref, n_ref)
str_b_ref = f"{n_b_ref/n_ref*100:5.1f}% [{ci_b_ref[0]:4.1f}, {ci_b_ref[1]:4.1f}]"
str_m_ref = f"{n_m_ref/n_ref*100:5.1f}% [{ci_m_ref[0]:4.1f}, {ci_m_ref[1]:4.1f}]"
str_z_ref = f"{n_z_ref/n_ref*100:5.1f}% [{ci_z_ref[0]:4.1f}, {ci_z_ref[1]:4.1f}]"

print("-" * 155)
print(header_fmt.format("Reference(38)", n_ref, "100.00%", "   N/A", f"{ref_p_cnn.mean():6.3f}", 
                        f"{ref_p_q.mean():6.3f}", f"{np.median(d_n_38):6.2f}", f"{np.median(d_a_38):6.2f}", 
                        str_b_ref, str_m_ref, str_z_ref))
print("=" * 155)

print("\n" + "=" * 80)
print(f"KEY COMPARISON: Worms-Derived vs. Seen-Family-Derived (Pooled & Stratified)")
print("=" * 80)
for s in [42, 43, 44]:
    cs = comp_by_seed[s]
    print(f"Seed {s}:")
    print(f"  Worms Zero-Day Rate: {cs['zd_w']}/{cs['n_w']} ({cs['pct_w']:.2f}%)")
    print(f"  Seen  Zero-Day Rate: {cs['zd_s']}/{cs['n_s']} ({cs['pct_s']:.2f}%)")
    print(f"  Difference (W - S) : {cs['diff']:+.2f}% [95% CI: {cs['diff_low']:+.2f}%, {cs['diff_high']:+.2f}%]")
    print(f"  Fisher Exact Test  : Odds Ratio = {cs['odds_ratio']:.3f}, p-value = {cs['p_val']:.4e}")

print("\nPOOLED (All 3 Seeds, N=600 Worms vs N=1800 Seen):")
cp = comp_pooled
print(f"  Worms Zero-Day Rate: {cp['zd_w']}/{cp['n_w']} ({cp['pct_w']:.2f}%)")
print(f"  Seen  Zero-Day Rate: {cp['zd_s']}/{cp['n_s']} ({cp['pct_s']:.2f}%)")
print(f"  Difference (W - S) : {cp['diff']:+.2f}% [95% CI: {cp['diff_low']:+.2f}%, {cp['diff_high']:+.2f}%]")
print(f"  Fisher Exact Test  : Odds Ratio = {cp['odds_ratio']:.3f}, p-value = {cp['p_val']:.4e}")

print("\nSTRATIFIED COMPARISON BY P_q BINS (Pooled Across Seeds):")
print(f"{'P_q Bin':<15} {'Worms ZD Rate':<18} {'Seen ZD Rate':<18} {'Difference (95% CI)':<26} {'Fisher p-val':<12}")
print("-" * 90)
for row in cp["strat"]:
    w_str = f"{row['zd_w']}/{row['n_w']} ({row['pct_w']:.1f}%)"
    s_str = f"{row['zd_s']}/{row['n_s']} ({row['pct_s']:.1f}%)"
    d_str = f"{row['diff']:+.1f}% [{row['ci_low']:+.1f}%, {row['ci_high']:+.1f}%]"
    print(f"{row['bin']:<15} {w_str:<18} {s_str:<18} {d_str:<26} {row['p_val']:<12.4e}")
print("=" * 90)

print("\n[Done] synthetic_routing_test.py completed successfully.")
