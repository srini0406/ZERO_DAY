import numpy as np, pandas as pd, joblib

ART = 'artifacts'
probs = np.load(f'{ART}/qsnn_test_probs.npy')
y_test = np.load(f'{ART}/y_test.npy')
y_cat = np.load(f'{ART}/y_test_cat.npy', allow_pickle=True)
cnn_preds = np.load(f'{ART}/cnn_test_preds.npy')
cnn_embs = np.load(f'{ART}/cnn_test_embeddings.npy')

db_scaler = joblib.load(f'{ART}/dbscan_scaler.pkl')
db_pca = joblib.load(f'{ART}/dbscan_pca.pkl')
db_nrm = joblib.load(f'{ART}/dbscan_normal_centroids.pkl')
db_atk = joblib.load(f'{ART}/dbscan_attack_centroids.pkl')
db_cfg = joblib.load(f'{ART}/dbscan_config.pkl')

b_th = db_cfg['benign_thresh']
a_th = db_cfg['attack_thresh']

emb_red = db_pca.transform(db_scaler.transform(cnn_embs))
d_n = np.array([min(np.linalg.norm(x - c) for c in db_nrm) for x in emb_red])
d_a = np.array([min(np.linalg.norm(x - c) for c in db_atk) for x in emb_red])

db_cats = []
for dn, da in zip(d_n, d_a):
    cn = dn <= b_th
    ca = da <= a_th
    if cn and not ca: db_cats.append('Benign Novel Behavior')
    elif ca and not cn: db_cats.append('Malicious Attack')
    elif cn and ca: db_cats.append('Benign Novel Behavior' if dn < da else 'Malicious Attack')
    else: db_cats.append('Zero-Day Attack')
db_cats = np.array(db_cats)

print('=== SENSITIVITY OF UNCERTAIN ZONE UPPER BOUND ===')
header = f"{'Upper Bound':<14} {'Total Unc':>10} {'Worms in Unc':>14} {'Worms -> ZD':>12} {'Overall ZD':>12} {'ZD Purity':>12}"
print(header)
print('-' * len(header))

for hi in [0.60, 0.65, 0.67, 0.677, 0.678, 0.68, 0.69, 0.70]:
    unc_mask = (cnn_preds == 1) & (probs >= 0.45) & (probs < hi)
    w_unc = int(((unc_mask) & (pd.Series(y_cat).astype(str).str.strip().str.lower() == 'worms')).sum())
    w_zd  = int(((unc_mask) & (pd.Series(y_cat).astype(str).str.strip().str.lower() == 'worms') & (db_cats == 'Zero-Day Attack')).sum())
    tot_zd = int(((unc_mask) & (db_cats == 'Zero-Day Attack')).sum())
    zd_true_atk = int(((unc_mask) & (db_cats == 'Zero-Day Attack') & (y_test == 1)).sum())
    purity = (zd_true_atk / tot_zd * 100) if tot_zd > 0 else 0.0
    print(f"[0.45, {hi:.3f})   {unc_mask.sum():>10,d} {w_unc:>14d} {w_zd:>12d} {tot_zd:>12,d} {purity:>11.1f}%")
