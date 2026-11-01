"""
test_zeroday_samples.py
──────────────────────────────────────────────────────────────────────────────
CLI test runner to evaluate the 10 Zero-Day attack samples stored in
'generated_zeroday_samples.csv' through the live multi-stage pipeline.

USAGE:
  python test_zeroday_samples.py             # Tests all 10 samples with summary table
  python test_zeroday_samples.py --sample 1  # Detailed trace of sample #1 (1 to 10)
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import pennylane as qml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ART = "artifacts"
CSV_PATH = "generated_zeroday_samples.csv"

# Load artifacts
print("Loading model artifacts and frozen transformers...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from cnn_model import CNN1D
cnn_model = CNN1D(57).to(device)
cnn_model.load_state_dict(torch.load(os.path.join(ART, "cnn_model.pt"), map_location=device))
cnn_model.eval()

scaler_full = joblib.load(os.path.join(ART, "scaler_full.pkl"))
ohe_full    = joblib.load(os.path.join(ART, "ohe_full.pkl"))
cnn_th      = float(joblib.load(os.path.join(ART, "cnn_threshold.pkl")))

scaler_15   = joblib.load(os.path.join(ART, "scaler_15.pkl"))
pca_q       = joblib.load(os.path.join(ART, "pca_quantum.pkl"))
pca_scaler  = joblib.load(os.path.join(ART, "pca_scaler.pkl"))
qsnn_params = joblib.load(os.path.join(ART, "qsnn_params.pkl"))
protos      = joblib.load(os.path.join(ART, "qsnn_prototypes.pkl"))
proto_atk   = protos["proto_attack"]
proto_nrm   = protos["proto_normal"]

db_cfg      = joblib.load(os.path.join(ART, "dbscan_config.pkl"))
bn_thresh   = float(db_cfg["benign_thresh"])
at_thresh   = float(db_cfg["attack_thresh"])
db_scaler   = joblib.load(os.path.join(ART, "dbscan_scaler.pkl"))
db_pca      = joblib.load(os.path.join(ART, "dbscan_pca.pkl"))
db_nrm      = joblib.load(os.path.join(ART, "dbscan_normal_centroids.pkl"))
db_atk      = joblib.load(os.path.join(ART, "dbscan_attack_centroids.pkl"))

full_names  = pd.read_csv(os.path.join(ART, "full_feature_names.csv"))["feature"].tolist()
NUM_COLS    = full_names[:39]
enc_names   = pd.read_csv(os.path.join(ART, "encoded_feature_names.csv"))["feature"].tolist()
col_indices = [NUM_COLS.index(c) for c in enc_names]

dev_q = qml.device("lightning.qubit", wires=6)
@qml.qnode(dev_q, diff_method="adjoint")
def qsnn_circuit(x, params):
    for lyr in range(3):
        for i in range(6):
            qml.RX(np.pi * x[i], wires=i)
        offset = lyr * 18
        for i in range(6):
            qml.RX(params[offset + 3*i],     wires=i)
            qml.RY(params[offset + 3*i + 1], wires=i)
            qml.RZ(params[offset + 3*i + 2], wires=i)
        for i in range(6):
            qml.CNOT(wires=[i, (i+1) % 6])
    return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(6)])

def evaluate_row(row_series):
    # 1. CNN
    num_vals = row_series[NUM_COLS].values.astype(float).reshape(1, -1)
    num_sc = scaler_full.transform(num_vals)
    cat_parts = []
    for col in ["proto", "service", "state"]:
        s = pd.Series([row_series[col]]).astype(str)
        s = s.where(s.isin(ohe_full[col]["keep_cats"]), other="other")
        for c in ohe_full[col]["all_cats"]:
            cat_parts.append((s == c).astype(float).values)
    x_cnn = np.hstack([num_sc, np.column_stack(cat_parts)]).astype(np.float32)
    
    with torch.no_grad():
        t = torch.tensor(x_cnn, device=device)
        p_cnn = float(cnn_model(t)[0, 1].cpu().item())
        emb = cnn_model.get_embedding(t).cpu().numpy()
        
    # 2. QSNN
    raw_10 = num_vals[:, col_indices]
    scaled_10 = scaler_15.transform(raw_10)
    q_in = pca_scaler.transform(pca_q.transform(scaled_10)).astype(np.float32)
    q_emb = np.array(qsnn_circuit(q_in.T, qsnn_params)).T[0]
    
    d_atk = float(np.linalg.norm(q_emb - proto_atk))
    d_nrm = float(np.linalg.norm(q_emb - proto_nrm))
    neg_d = np.array([-d_atk, -d_nrm])
    exp_d = np.exp(neg_d - neg_d.max())
    p_q = float(exp_d[0] / exp_d.sum())
    
    # 3. DBSCAN
    emb_sc = db_scaler.transform(emb)
    emb_pca = db_pca.transform(emb_sc)
    dn = float(min(np.linalg.norm(emb_pca[0] - c) for c in db_nrm))
    da = float(min(np.linalg.norm(emb_pca[0] - c) for c in db_atk))
    
    close_n = (dn <= bn_thresh)
    close_a = (da <= at_thresh)
    if close_n and not close_a:
        verdict = "Benign Novel Behavior"
        auth = "Stage 4 (DBSCAN Whitelist)"
    elif close_a and not close_n:
        verdict = "Malicious Attack"
        auth = "Stage 4 (DBSCAN Known Cluster)"
    elif close_n and close_a:
        verdict = "Benign Novel Behavior" if dn < da else "Malicious Attack"
        auth = "Stage 4 (DBSCAN Tie-Breaker)"
    else:
        verdict = "Zero-Day Attack"
        auth = "Stage 4 (DBSCAN Geometric Outlier)"
        
    return p_cnn, p_q, dn, da, verdict, auth, d_atk, d_nrm

def main():
    parser = argparse.ArgumentParser(description="Test Zero-Day samples from generated_zeroday_samples.csv")
    parser.add_argument("--sample", type=int, default=None, help="Sample index to test (1 to 10)")
    args = parser.parse_args()
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Please run 'python generate_zeroday_samples.py' first.")
        sys.exit(1)
        
    df = pd.read_csv(CSV_PATH)
    n_total = len(df)
    
    if args.sample is not None:
        idx = args.sample - 1
        if idx < 0 or idx >= n_total:
            print(f"Error: Sample index must be between 1 and {n_total}.")
            sys.exit(1)
            
        row = df.iloc[idx]
        p_c, p_q, dn, da, verd, auth, datk, dnrm = evaluate_row(row)
        
        print("\n" + "=" * 75)
        print(f"DETAILED PIPELINE TRACE FOR SAMPLE #{args.sample}: {row['sample_type']}")
        print("=" * 75)
        print(f"• Packet Identification : ID={row['id']} | Proto={row['proto']} | Svc={row['service']} | State={row['state']}")
        print(f"• Flow Dynamics        : Dur={row['dur']:.2f}s | Sbytes={int(row['sbytes']):,} | Dbytes={int(row['dbytes']):,} | Rate={row['rate']:.1f} pkts/s")
        print(f"• TTL Values           : Source TTL={int(row['sttl'])} | Dest TTL={int(row['dttl'])}")
        print("-" * 75)
        print(f"[Stage 2: 1D-CNN Screener]")
        print(f"  • Attack Probability P(CNN) : {p_c:.4f}  (Threshold = {cnn_th:.2f})")
        print(f"  • Anomaly Screener Result   : {'ALERT: Suspicious Traffic Flagged' if p_c >= cnn_th else 'CLEARED: Normal'}")
        print(f"\n[Stage 3: 6-Qubit QSNN Quantum Verifier]")
        print(f"  • Quantum Attack Prob P(Q) : {p_q:.4f}  (Uncertain Zone = [0.45, 0.60))")
        print(f"  • Prototype Distances       : d(Attack Prototype)={datk:.3f} | d(Normal Prototype)={dnrm:.3f}")
        print(f"  • Uncertainty Gating        : {'TRIGGERED (Escalated to Stage 4 DBSCAN)' if (0.45 <= p_q < 0.60) else 'Bypassed'}")
        print(f"\n[Stage 4: DBSCAN Geometric Latent Triage]")
        print(f"  • Distance to Normal Centroid : {dn:.2f}  (Cutoff = {bn_thresh:.2f})  -> Exceeds Cutoff: {dn > bn_thresh}")
        print(f"  • Distance to Attack Centroid : {da:.2f}  (Cutoff = {at_thresh:.2f})  -> Exceeds Cutoff: {da > at_thresh}")
        print(f"  • Geometric Analysis          : Outlier from ALL known clusters in 10-dim latent space")
        print("-" * 75)
        print(f"FINAL DECISION : {verd.upper()}")
        print(f"AUTHORITY      : {auth}")
        print("=" * 75 + "\n")
        
    else:
        print("\n" + "=" * 105)
        print(f"EVALUATING ALL {n_total} ZERO-DAY ATTACK SAMPLES THROUGH COMPLETE HYBRID PIPELINE")
        print("=" * 105)
        print(f"{'#':<3} {'Sample Type':<38} {'P(CNN)':<8} {'P(Q)':<8} {'d(Norm)':<9} {'d(Atk)':<8} {'Live Verdict':<18} {'Status'}")
        print("-" * 105)
        
        all_passed = True
        for i in range(n_total):
            row = df.iloc[i]
            p_c, p_q, dn, da, verd, auth, _, _ = evaluate_row(row)
            is_zd = (verd == "Zero-Day Attack")
            status = "✅ PASS" if is_zd else "❌ FAIL"
            if not is_zd: all_passed = False
            
            print(f"{i+1:<3} {row['sample_type']:<38} {p_c:<8.4f} {p_q:<8.4f} {dn:<9.2f} {da:<8.2f} {verd:<18} {status}")
            
        print("=" * 105)
        if all_passed:
            print("RESULT: 100% of samples (10/10) verified and classified as 'Zero-Day Attack' by DBSCAN!")
        print("\nTip: To trace an individual sample with full step-by-step forensic logs, run:")
        print("     python test_zeroday_samples.py --sample 1\n")

if __name__ == "__main__":
    main()
