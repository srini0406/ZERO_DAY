import os, sys, torch, joblib, numpy as np, pandas as pd, pennylane as qml

ART = 'artifacts'
N_QUBITS = 6

print("Loading artifacts for calibrated novelty gate...")
pca_q = joblib.load(f'{ART}/pca_quantum.pkl')
pca_q_sc = joblib.load(f'{ART}/pca_scaler.pkl')
qsnn_params = joblib.load(f'{ART}/qsnn_params.pkl')
protos = joblib.load(f'{ART}/qsnn_prototypes.pkl')
proto_atk = protos['proto_attack']
qae_params = joblib.load(f'{ART}/qae_params.pkl')

y_train = np.load(f'{ART}/y_train.npy')
X_train_15 = np.load(f'{ART}/X_train_15.npy')

# Training attacks only (Worms was already excluded during leave-one-out split in preprocessing)
train_atk_mask = (y_train == 1)
X_train_atk_15 = X_train_15[train_atk_mask]

# Subsample 2000 training attacks for fast exact threshold calibration
np.random.seed(42)
idx_sub = np.random.choice(len(X_train_atk_15), 2000, replace=False)
X_sub = X_train_atk_15[idx_sub]

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

X_pca_sub = pca_q.transform(X_sub)
X_q_sub = pca_q_sc.transform(X_pca_sub).astype(np.float32)

print("Calibrating on training attacks...")
qae_raw_train = np.array(qae_circuit(X_q_sub.T, qae_params))
qae_err_train = np.mean(1.0 - qae_raw_train.T, axis=1)
qae_th_95 = float(np.percentile(qae_err_train, 95))
qae_th_99 = float(np.percentile(qae_err_train, 99))

qsnn_emb_train = np.array(qsnn_circuit(X_q_sub.T, qsnn_params)).T
qsnn_d_atk_train = np.linalg.norm(qsnn_emb_train - proto_atk, axis=1)
qsnn_d_th_95 = float(np.percentile(qsnn_d_atk_train, 95))
qsnn_d_th_99 = float(np.percentile(qsnn_d_atk_train, 99))

print(f"\n=======================================================")
print(f"TRAINING DATA CALIBRATED THRESHOLDS (Worms excluded):")
print(f"  QAE Error 95th pct = {qae_th_95:.4f}, 99th pct = {qae_th_99:.4f}")
print(f"  QSNN Dist 95th pct = {qsnn_d_th_95:.4f}, 99th pct = {qsnn_d_th_99:.4f}")
print(f"=======================================================\n")

# Load Test & Held-out
y_test = np.load(f'{ART}/y_test.npy')
y_test_cat = np.load(f'{ART}/y_test_cat.npy', allow_pickle=True)
cnn_preds = np.load(f'{ART}/cnn_test_preds.npy')

is_test_worm = (pd.Series(y_test_cat).astype(str).str.strip().str.lower() == 'worms').values
is_test_known_atk = (y_test == 1) & (~is_test_worm)
cnn_flagged_test = (cnn_preds == 1)

qae_err_test = np.load(f'{ART}/qae_test_errors.npy')

# For test QSNN embeddings, check if already saved or compute on test attacks
print("Evaluating QSNN on test attacks...")
X_15_test = np.load(f'{ART}/X_test_15.npy')
X_pca_test = pca_q.transform(X_15_test)
X_q_test = pca_q_sc.transform(X_pca_test).astype(np.float32)

# Subdivide into 10000 chunks
qsnn_embs_test = []
for s in range(0, len(X_q_test), 5000):
    chunk = X_q_test[s:s+5000]
    qsnn_embs_test.append(np.array(qsnn_circuit(chunk.T, qsnn_params)).T)
qsnn_embs_test = np.vstack(qsnn_embs_test)
qsnn_d_atk_test = np.linalg.norm(qsnn_embs_test - proto_atk, axis=1)

# Held-out Worms
df_heldout = pd.read_csv(f'{ART}/zero_day_worms.csv')
scaler_15 = joblib.load(f'{ART}/scaler_15.pkl')
feat_df = pd.read_csv(f'{ART}/encoded_feature_names.csv')
FEATURE_NAMES = feat_df['feature'].tolist()
X_15_h = scaler_15.transform(df_heldout[FEATURE_NAMES].values.astype(float)).astype(np.float32)
X_pca_h = pca_q.transform(X_15_h)
X_q_h = pca_q_sc.transform(X_pca_h).astype(np.float32)

qae_err_h = np.mean(1.0 - np.array(qae_circuit(X_q_h.T, qae_params)).T, axis=1)
qsnn_d_atk_h = np.linalg.norm(np.array(qsnn_circuit(X_q_h.T, qsnn_params)).T - proto_atk, axis=1)

def report_gate(score_name, train_th_name, th_val, test_scores, heldout_scores):
    flagged_novel_test = (cnn_flagged_test) & (test_scores > th_val)
    n_routed = int(flagged_novel_test.sum())
    
    worms_flagged_test = int(((flagged_novel_test) & (is_test_worm)).sum())
    n_worms_test = int(is_test_worm.sum())
    
    known_atk_flagged = int(((flagged_novel_test) & (is_test_known_atk)).sum())
    n_known_atk = int((cnn_flagged_test & is_test_known_atk).sum())
    false_novel_rate = (known_atk_flagged / n_known_atk * 100) if n_known_atk > 0 else 0.0
    
    worms_flagged_h = int((heldout_scores > th_val).sum())
    n_worms_h = len(heldout_scores)
    
    print(f"=== Gate: {score_name} ({train_th_name} = {th_val:.4f}) ===")
    print(f"  Total CNN-flagged test flows routed to Novelty: {n_routed:,} / {int(cnn_flagged_test.sum()):,} ({n_routed/int(cnn_flagged_test.sum())*100:.2f}%)")
    print(f"  False-Novel Rate on Known Attacks (Test): {known_atk_flagged:,} / {n_known_atk:,} ({false_novel_rate:.2f}%)")
    print(f"  Test Worms Flagged Novel: {worms_flagged_test} / {n_worms_test} ({worms_flagged_test/n_worms_test*100:.2f}%)")
    print(f"  Held-out Worms Flagged Novel: {worms_flagged_h} / {n_worms_h} ({worms_flagged_h/n_worms_h*100:.2f}%)")
    print()

report_gate('QAE Reconstruction Error', 'Training 95th Percentile', qae_th_95, qae_err_test, qae_err_h)
report_gate('QAE Reconstruction Error', 'Training 99th Percentile', qae_th_99, qae_err_test, qae_err_h)
report_gate('QSNN Distance to Attack Prototype', 'Training 95th Percentile', qsnn_d_th_95, qsnn_d_atk_test, qsnn_d_atk_h)
report_gate('QSNN Distance to Attack Prototype', 'Training 99th Percentile', qsnn_d_th_99, qsnn_d_atk_test, qsnn_d_atk_h)
