"""
trace_and_investigate_worms.py
──────────────────────────────
Trace all 130 held-out Worms and 44 test Worms through the complete pipeline.
Investigate uncertain-zone calibration and DBSCAN zero-day assignment.
"""
import os, sys, torch, joblib, numpy as np, pandas as pd, pennylane as qml
from cnn_model import CNN1D
from preprocessing import ALL_NUMERIC_FEATURES, ALL_CATEGORICAL_FEATURES

ART = 'artifacts'
N_QUBITS = 6

scaler_full = joblib.load(f'{ART}/scaler_full.pkl')
ohe_full = joblib.load(f'{ART}/ohe_full.pkl')
cnn_thresh = float(joblib.load(f'{ART}/cnn_threshold.pkl'))

pca_q = joblib.load(f'{ART}/pca_quantum.pkl')
pca_q_sc = joblib.load(f'{ART}/pca_scaler.pkl')
qsnn_params = joblib.load(f'{ART}/qsnn_params.pkl')
protos = joblib.load(f'{ART}/qsnn_prototypes.pkl')
proto_atk = protos['proto_attack']
proto_nrm = protos['proto_normal']

db_scaler = joblib.load(f'{ART}/dbscan_scaler.pkl')
db_pca = joblib.load(f'{ART}/dbscan_pca.pkl')
db_nrm = joblib.load(f'{ART}/dbscan_normal_centroids.pkl')
db_atk = joblib.load(f'{ART}/dbscan_attack_centroids.pkl')
db_cfg = joblib.load(f'{ART}/dbscan_config.pkl')

feat_df = pd.read_csv(f'{ART}/encoded_feature_names.csv')
FEATURE_NAMES = feat_df['feature'].tolist()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = CNN1D(n_features=57).to(device)
cnn_model.load_state_dict(torch.load(f'{ART}/cnn_model.pt', map_location=device))
cnn_model.eval()

dev = qml.device('lightning.qubit', wires=N_QUBITS)
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

def get_full_and_10(df):
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
    
    scaler_15 = joblib.load(f'{ART}/scaler_15.pkl')
    X_10 = scaler_15.transform(df[FEATURE_NAMES].values.astype(float)).astype(np.float32)
    return X_full, X_10

df_heldout = pd.read_csv(f'{ART}/zero_day_worms.csv')
X_full_h, X_10_h = get_full_and_10(df_heldout)

y_test = np.load(f'{ART}/y_test.npy')
y_test_cat = np.load(f'{ART}/y_test_cat.npy', allow_pickle=True)
worms_test_idx = np.where(pd.Series(y_test_cat).astype(str).str.strip().str.lower() == 'worms')[0]
X_full_t = np.load(f'{ART}/X_test_full.npy')[worms_test_idx]
X_10_t = np.load(f'{ART}/X_test_15.npy')[worms_test_idx]

def compute_detailed_trace(X_full, X_10):
    with torch.no_grad():
        x_t = torch.tensor(X_full, dtype=torch.float32).to(device)
        cnn_probs = cnn_model(x_t)[:, 1].cpu().numpy()
        cnn_embs = cnn_model.get_embedding(x_t).cpu().numpy()
    
    cnn_flags = (cnn_probs >= cnn_thresh)
    
    X_pca = pca_q.transform(X_10)
    X_q = pca_q_sc.transform(X_pca).astype(np.float32)
    
    embs_q = []
    for s in range(0, len(X_q), 500):
        chunk = X_q[s:s+500]
        embs_q.append(np.array(qsnn_circuit(chunk.T, qsnn_params)).T)
    emb_q = np.vstack(embs_q)
    
    d_atk = np.linalg.norm(emb_q - proto_atk, axis=1)
    d_nrm = np.linalg.norm(emb_q - proto_nrm, axis=1)
    neg_d = np.stack([-d_atk, -d_nrm], axis=1)
    exp_d = np.exp(neg_d - neg_d.max(axis=1, keepdims=True))
    qsnn_probs = (exp_d / exp_d.sum(axis=1, keepdims=True))[:, 0]
    
    # Direct DBSCAN projection for all samples
    emb_red = db_pca.transform(db_scaler.transform(cnn_embs))
    d_n = np.array([min(np.linalg.norm(x - c) for c in db_nrm) for x in emb_red])
    d_a = np.array([min(np.linalg.norm(x - c) for c in db_atk) for x in emb_red])
    b_th = db_cfg['benign_thresh']
    a_th = db_cfg['attack_thresh']
    
    db_cats = []
    for dn, da in zip(d_n, d_a):
        cn = dn <= b_th
        ca = da <= a_th
        if cn and not ca: db_cats.append('Benign Novel Behavior')
        elif ca and not cn: db_cats.append('Malicious Attack')
        elif cn and ca: db_cats.append('Benign Novel Behavior' if dn < da else 'Malicious Attack')
        else: db_cats.append('Zero-Day Attack')
        
    return pd.DataFrame({
        'cnn_prob': cnn_probs,
        'cnn_flag': cnn_flags,
        'qsnn_prob': qsnn_probs,
        'd_normal_dbscan': d_n,
        'd_attack_dbscan': d_a,
        'dbscan_category_if_escalated': db_cats
    })

trace_h = compute_detailed_trace(X_full_h, X_10_h)
trace_t = compute_detailed_trace(X_full_t, X_10_t)

print("=== HELD-OUT WORMS (n=130) TRACE ===")
print("Summary Statistics:")
print(f"CNN Flagged Attack (p >= {cnn_thresh:.4f}): {trace_h['cnn_flag'].sum()}/130 ({trace_h['cnn_flag'].mean()*100:.1f}%)")
print(f"CNN Cleared Normal (p < {cnn_thresh:.4f}): {(~trace_h['cnn_flag']).sum()}/130 ({(~trace_h['cnn_flag']).mean()*100:.1f}%)")
print(f"QSNN Probabilities (on 128 CNN-flagged): min={trace_h[trace_h['cnn_flag']]['qsnn_prob'].min():.4f}, mean={trace_h[trace_h['cnn_flag']]['qsnn_prob'].mean():.4f}, max={trace_h[trace_h['cnn_flag']]['qsnn_prob'].max():.4f}")
print(f"QSNN >= 0.60 (Definite Attack band): {(trace_h['cnn_flag'] & (trace_h['qsnn_prob'] >= 0.60)).sum()}/130")
print(f"QSNN < 0.45 (Definite Normal band): {(trace_h['cnn_flag'] & (trace_h['qsnn_prob'] < 0.45)).sum()}/130")
print(f"QSNN in [0.45, 0.60] (Uncertain zone): {(trace_h['cnn_flag'] & (trace_h['qsnn_prob'] >= 0.45) & (trace_h['qsnn_prob'] < 0.60)).sum()}/130")
print()
print("=== TEST SET WORMS (n=44) TRACE ===")
print(f"CNN Flagged Attack (p >= {cnn_thresh:.4f}): {trace_t['cnn_flag'].sum()}/44 ({trace_t['cnn_flag'].mean()*100:.1f}%)")
print(f"CNN Cleared Normal (p < {cnn_thresh:.4f}): {(~trace_t['cnn_flag']).sum()}/44 ({(~trace_t['cnn_flag']).mean()*100:.1f}%)")
print(f"QSNN Probabilities (on 40 CNN-flagged): min={trace_t[trace_t['cnn_flag']]['qsnn_prob'].min():.4f}, mean={trace_t[trace_t['cnn_flag']]['qsnn_prob'].mean():.4f}, max={trace_t[trace_t['cnn_flag']]['qsnn_prob'].max():.4f}")
print(f"QSNN >= 0.60 (Definite Attack band): {(trace_t['cnn_flag'] & (trace_t['qsnn_prob'] >= 0.60)).sum()}/44")
print(f"QSNN < 0.45 (Definite Normal band): {(trace_t['cnn_flag'] & (trace_t['qsnn_prob'] < 0.45)).sum()}/44")
print(f"QSNN in [0.45, 0.60] (Uncertain zone): {(trace_t['cnn_flag'] & (trace_t['qsnn_prob'] >= 0.45) & (trace_t['qsnn_prob'] < 0.60)).sum()}/44")
