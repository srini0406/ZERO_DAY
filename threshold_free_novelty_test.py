"""
threshold_free_novelty_test.py
───────────────────────────────
Threshold-free novelty detection evaluation:
Computes AUROC and AUPRC for separating genuinely held-out Zero-Day Worms (positives)
from known attacks in the test set (negatives) across five distinct novelty/anomaly scores.
"""
import os, sys, torch, joblib, numpy as np, pandas as pd, pennylane as qml
from sklearn.metrics import roc_auc_score, average_precision_score
from cnn_model import CNN1D
from preprocessing import ALL_NUMERIC_FEATURES, ALL_CATEGORICAL_FEATURES

ART = 'artifacts'
N_QUBITS = 6

scaler_full = joblib.load(f'{ART}/scaler_full.pkl')
ohe_full = joblib.load(f'{ART}/ohe_full.pkl')
scaler_15 = joblib.load(f'{ART}/scaler_15.pkl')
feat_df = pd.read_csv(f'{ART}/encoded_feature_names.csv')
FEATURE_NAMES = feat_df['feature'].tolist()

pca_q = joblib.load(f'{ART}/pca_quantum.pkl')
pca_q_sc = joblib.load(f'{ART}/pca_scaler.pkl')
qsnn_params = joblib.load(f'{ART}/qsnn_params.pkl')
protos = joblib.load(f'{ART}/qsnn_prototypes.pkl')
proto_atk = protos['proto_attack']
proto_nrm = protos['proto_normal']

qae_params = joblib.load(f'{ART}/qae_params.pkl')

db_scaler = joblib.load(f'{ART}/dbscan_scaler.pkl')
db_pca = joblib.load(f'{ART}/dbscan_pca.pkl')
db_nrm = joblib.load(f'{ART}/dbscan_normal_centroids.pkl')
db_atk = joblib.load(f'{ART}/dbscan_attack_centroids.pkl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = CNN1D(n_features=57).to(device)
cnn_model.load_state_dict(torch.load(f'{ART}/cnn_model.pt', map_location=device))
cnn_model.eval()

dev = qml.device('lightning.qubit', wires=N_QUBITS)

@qml.qnode(dev, diff_method='adjoint')
def qae_circuit(x, params):
    for i in range(N_QUBITS):
        qml.RX(np.pi * x[i], wires=i)
    for lyr in range(3):
        offset = lyr * 18
        for i in range(N_QUBITS):
            qml.RX(params[offset + 3*i],     wires=i)
            qml.RY(params[offset + 3*i + 1], wires=i)
            qml.RZ(params[offset + 3*i + 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i+1) % N_QUBITS])
    return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in [3, 4, 5]])

@qml.qnode(dev, diff_method='adjoint')
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

def get_features(df):
    X_num = scaler_full.transform(df[ALL_NUMERIC_FEATURES].values.astype(float))
    cat_parts = []
    for col in ALL_CATEGORICAL_FEATURES:
        s = df[col].astype(str)
        info = ohe_full[col]
        s = s.where(s.isin(info['keep_cats']), other='other')
        for cat in info['all_cats']:
            cat_parts.append((s == cat).astype(float).values)
    X_cat = np.column_stack(cat_parts)
    X_full = np.hstack([X_num, X_cat]).astype(np.float32)
    X_10 = scaler_15.transform(df[FEATURE_NAMES].values.astype(float)).astype(np.float32)
    return X_full, X_10

# 1. Test Set
y_test = np.load(f'{ART}/y_test.npy')
y_test_cat = np.load(f'{ART}/y_test_cat.npy', allow_pickle=True)
is_test_worm = (pd.Series(y_test_cat).astype(str).str.strip().str.lower() == 'worms').values
is_test_attack = (y_test == 1)
is_test_known_attack = is_test_attack & (~is_test_worm)

print(f"Test known attacks (negatives): {is_test_known_attack.sum():,}")
print(f"Test Worms (positives): {is_test_worm.sum():,}")

# 2. Held-out Worms
df_heldout = pd.read_csv(f'{ART}/zero_day_worms.csv')
X_full_h, X_10_h = get_features(df_heldout)
print(f"Held-out Worms (positives): {len(df_heldout):,}")

X_full_test = np.load(f'{ART}/X_test_full.npy')
X_10_test = np.load(f'{ART}/X_test_15.npy')

with torch.no_grad():
    x_t = torch.tensor(X_full_test, dtype=torch.float32).to(device)
    cnn_probs_test = cnn_model(x_t)[:, 1].cpu().numpy()
    cnn_embs_test = cnn_model.get_embedding(x_t).cpu().numpy()

emb_red_test = db_pca.transform(db_scaler.transform(cnn_embs_test))
d_atk_db_test = np.array([min(np.linalg.norm(x - c) for c in db_atk) for x in emb_red_test])
d_nrm_db_test = np.array([min(np.linalg.norm(x - c) for c in db_nrm) for x in emb_red_test])
d_any_db_test = np.minimum(d_atk_db_test, d_nrm_db_test)

X_pca_test = pca_q.transform(X_10_test)
X_q_test = pca_q_sc.transform(X_pca_test).astype(np.float32)
qae_errors_test = np.load(f'{ART}/qae_test_errors.npy')

qsnn_embs_test = []
for s in range(0, len(X_q_test), 2000):
    chunk = X_q_test[s:s+2000]
    qsnn_embs_test.append(np.array(qsnn_circuit(chunk.T, qsnn_params)).T)
qsnn_embs_test = np.vstack(qsnn_embs_test)
qsnn_d_atk_test = np.linalg.norm(qsnn_embs_test - proto_atk, axis=1)

# Held-out Worms
with torch.no_grad():
    x_h = torch.tensor(X_full_h, dtype=torch.float32).to(device)
    cnn_probs_h = cnn_model(x_h)[:, 1].cpu().numpy()
    cnn_embs_h = cnn_model.get_embedding(x_h).cpu().numpy()

emb_red_h = db_pca.transform(db_scaler.transform(cnn_embs_h))
d_atk_db_h = np.array([min(np.linalg.norm(x - c) for c in db_atk) for x in emb_red_h])
d_nrm_db_h = np.array([min(np.linalg.norm(x - c) for c in db_nrm) for x in emb_red_h])
d_any_db_h = np.minimum(d_atk_db_h, d_nrm_db_h)

X_pca_h = pca_q.transform(X_10_h)
X_q_h = pca_q_sc.transform(X_pca_h).astype(np.float32)

raw_qae_h = np.array(qae_circuit(X_q_h.T, qae_params))
qae_errors_h = np.mean(1.0 - raw_qae_h.T, axis=1)

qsnn_embs_h = np.array(qsnn_circuit(X_q_h.T, qsnn_params)).T
qsnn_d_atk_h = np.linalg.norm(qsnn_embs_h - proto_atk, axis=1)

def eval_scores(pos_dict, neg_dict, title):
    print(f"\n=== {title} ===")
    n_pos = len(next(iter(pos_dict.values())))
    n_neg = len(next(iter(neg_dict.values())))
    prev = n_pos / (n_pos + n_neg)
    print(f"Positives (Worms): {n_pos:,} | Negatives (Known Attacks): {n_neg:,} | Positive Prevalence: {prev*100:.3f}%")
    header = f"{'Score Metric':<45} {'AUROC':>10} {'AUPRC':>10} {'Random Baseline':>18}"
    print(header)
    print("-" * len(header))
    results = []
    for name in pos_dict.keys():
        pos_v = pos_dict[name]
        neg_v = neg_dict[name]
        y_true = np.concatenate([np.ones(len(pos_v)), np.zeros(len(neg_v))])
        y_score = np.concatenate([pos_v, neg_v])
        
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        print(f"{name:<45} {auc:>10.4f} {ap:>10.4f} {prev:>18.4f}")
        results.append({"metric": name, "auroc": auc, "auprc": ap, "random_auprc": prev})
    return pd.DataFrame(results)

negatives = {
    "(a) Min dist to Attack Centroids (DBSCAN)": d_atk_db_test[is_test_known_attack],
    "(b) Min dist to ANY Centroid (DBSCAN)":    d_any_db_test[is_test_known_attack],
    "(c) QAE Reconstruction Error":              qae_errors_test[is_test_known_attack],
    "(d) QSNN Dist to Attack Prototype":         qsnn_d_atk_test[is_test_known_attack],
    "(e) CNN Anomaly (1 - P_attack)":            1.0 - cnn_probs_test[is_test_known_attack]
}

pos_test = {
    "(a) Min dist to Attack Centroids (DBSCAN)": d_atk_db_test[is_test_worm],
    "(b) Min dist to ANY Centroid (DBSCAN)":    d_any_db_test[is_test_worm],
    "(c) QAE Reconstruction Error":              qae_errors_test[is_test_worm],
    "(d) QSNN Dist to Attack Prototype":         qsnn_d_atk_test[is_test_worm],
    "(e) CNN Anomaly (1 - P_attack)":            1.0 - cnn_probs_test[is_test_worm]
}

pos_heldout = {
    "(a) Min dist to Attack Centroids (DBSCAN)": d_atk_db_h,
    "(b) Min dist to ANY Centroid (DBSCAN)":    d_any_db_h,
    "(c) QAE Reconstruction Error":              qae_errors_h,
    "(d) QSNN Dist to Attack Prototype":         qsnn_d_atk_h,
    "(e) CNN Anomaly (1 - P_attack)":            1.0 - cnn_probs_h
}

pos_combined = {
    k: np.concatenate([pos_heldout[k], pos_test[k]]) for k in pos_test.keys()
}

df_res_test    = eval_scores(pos_test, negatives, "TEST SET WORMS (n=44) vs KNOWN ATTACKS (n=45,288)")
df_res_heldout = eval_scores(pos_heldout, negatives, "HELD-OUT WORMS (n=130) vs KNOWN ATTACKS (n=45,288)")
df_res_all     = eval_scores(pos_combined, negatives, "ALL WORMS COMBINED (n=174) vs KNOWN ATTACKS (n=45,288)")

joblib.dump({
    "test_worms": df_res_test,
    "heldout_worms": df_res_heldout,
    "combined_worms": df_res_all
}, os.path.join(ART, "threshold_free_novelty_results.pkl"))
print("\nSaved artifacts/threshold_free_novelty_results.pkl")
