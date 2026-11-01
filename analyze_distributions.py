import numpy as np, pandas as pd, joblib

ART = 'artifacts'
probs = np.load(f'{ART}/qsnn_test_probs.npy')
y_test = np.load(f'{ART}/y_test.npy')
y_cat = np.load(f'{ART}/y_test_cat.npy', allow_pickle=True)
cnn_preds = np.load(f'{ART}/cnn_test_preds.npy')

df = pd.DataFrame({'p': probs, 'cnn': cnn_preds, 'y': y_test, 'cat': y_cat})

print('=== QSNN PROBABILITIES FOR CNN-FLAGGED FLOWS (CNN=1) ===')
for cat in sorted(df['cat'].unique()):
    sub = df[(df['cat'] == cat) & (df['cnn'] == 1)]
    if len(sub) > 0:
        p_min = sub['p'].min()
        p_10  = np.percentile(sub['p'], 10)
        p_med = sub['p'].median()
        p_90  = np.percentile(sub['p'], 90)
        p_max = sub['p'].max()
        print(f'{cat:<16}: n={len(sub):5d} | min={p_min:.4f}  p10={p_10:.4f}  med={p_med:.4f}  p90={p_90:.4f}  max={p_max:.4f}')

n_norm = int((y_test == 0).sum())
n_atk  = int((y_test == 1).sum())
print('\n=== OVERALL TEST SET PROBABILITY DISTRIBUTION (All 82,332) ===')
print(f'Normal flows (y=0, n={n_norm:,}): min={probs[y_test==0].min():.4f}, p10={np.percentile(probs[y_test==0], 10):.4f}, med={np.median(probs[y_test==0]):.4f}, p90={np.percentile(probs[y_test==0], 90):.4f}, max={probs[y_test==0].max():.4f}')
print(f'Attack flows (y=1, n={n_atk:,}): min={probs[y_test==1].min():.4f}, p10={np.percentile(probs[y_test==1], 10):.4f}, med={np.median(probs[y_test==1]):.4f}, p90={np.percentile(probs[y_test==1], 90):.4f}, max={probs[y_test==1].max():.4f}')
