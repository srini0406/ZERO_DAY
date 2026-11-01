import joblib, numpy as np, pennylane as qml

N_QUBITS = 6
N_VQC_LAYERS = 3
PARAMS_PER_LAYER = 18

dev = qml.device('lightning.qubit', wires=N_QUBITS)
@qml.qnode(dev, diff_method='adjoint')
def embedding_circuit(x, params):
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
    return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)])

params = joblib.load('artifacts/qsnn_params.pkl')
protos = joblib.load('artifacts/qsnn_prototypes.pkl')
p_atk, p_nrm = protos['proto_attack'], protos['proto_normal']
inter = float(np.linalg.norm(p_atk - p_nrm))

X_train = np.load('artifacts/quantum_input_train.npy')
y_train = np.load('artifacts/y_train.npy')

rng = np.random.default_rng(42)
idx = rng.choice(len(X_train), 1000, replace=False)
X_sub = X_train[idx]
y_sub = y_train[idx]

embs = np.array([embedding_circuit(x, params) for x in X_sub])
intra_atk = float(np.mean(np.linalg.norm(embs[y_sub == 1] - p_atk, axis=1)))
intra_nrm = float(np.mean(np.linalg.norm(embs[y_sub == 0] - p_nrm, axis=1)))
sep_ratio = inter / (max(intra_atk, intra_nrm) + 1e-9)

print('Baseline Inter-prototype dist:', inter)
print('Baseline Intra atk dist:', intra_atk)
print('Baseline Intra nrm dist:', intra_nrm)
print('Baseline Separation ratio:', sep_ratio)
